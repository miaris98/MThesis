"""Off-Policy MCTS + GTrXL + EfficientZero v2 Trainer for Atari 100k Benchmark.

Combines:
1. Latent Monte Carlo Tree Search (MCTS) for test-time and collection-time lookahead planning
2. Off-policy Experience Replay buffer storing MCTS visit distributions and values
3. Multi-step GTrXL + EfficientZero v2 dynamics learning (SimSiam consistency, reward/value heads)
4. MLflow experiment tracking on port 10100
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from atari_qwen.envs.atari_wrappers import make_atari_env, make_vector_atari_envs, StickyActionEnv
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.mcts.mcts_engine import MCTSEngine
from atari_qwen.eval.evaluate import compute_hns


class MCTSReplayBuffer:
    """Off-policy circular replay buffer storing per-environment transitions to guarantee
    trajectory continuity across unroll windows."""
    def __init__(
        self,
        capacity: int = 50_000,
        num_envs: int = 8,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        action_dim: int = 4,
        unroll_steps: int = 5,
        capacity_per_env: Optional[int] = None,
        priority_alpha: float = 0.0,
    ):
        # priority_alpha > 0 enables prioritized replay (EfficientZero-style): windows are sampled
        # with probability ~ priority^alpha, priority = root value error, new data at max priority.
        self.priority_alpha = priority_alpha
        if capacity_per_env is not None:
            self.capacity = capacity_per_env
        else:
            self.capacity = max(100, capacity // num_envs)
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.unroll_steps = unroll_steps

        self.obs_buf = np.zeros((self.capacity, num_envs, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros((self.capacity, num_envs), dtype=np.int64)
        self.rew_buf = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, num_envs), dtype=np.bool_)
        self.pi_buf = np.zeros((self.capacity, num_envs, action_dim), dtype=np.float32)
        self.val_buf = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.prio_buf = np.zeros((self.capacity, num_envs), dtype=np.float32)
        self.max_prio = 1.0

        self.ptr = 0
        self.size = 0  # Steps per environment stored

    @property
    def total_transitions(self) -> int:
        return self.size * self.num_envs

    def add_batch(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        done_batch: np.ndarray,
        pi_batch: np.ndarray,
        val_batch: np.ndarray,
    ):
        self.obs_buf[self.ptr] = obs_batch
        self.act_buf[self.ptr] = act_batch
        self.rew_buf[self.ptr] = rew_batch
        self.done_buf[self.ptr] = done_batch
        self.pi_buf[self.ptr] = pi_batch
        self.val_buf[self.ptr] = val_batch
        self.prio_buf[self.ptr] = self.max_prio

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save(self, path: Path) -> None:
        """Write the whole buffer to one uncompressed .npz (resume must not refill with random data)."""
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(
            tmp,
            obs=self.obs_buf, act=self.act_buf, rew=self.rew_buf, done=self.done_buf,
            pi=self.pi_buf, val=self.val_buf, ptr=np.int64(self.ptr), size=np.int64(self.size),
            prio=self.prio_buf, max_prio=np.float64(self.max_prio),
        )
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a truncated buffer behind

    def load(self, path: Path) -> None:
        d = np.load(path)
        if d["obs"].shape != self.obs_buf.shape:
            raise ValueError(f"Replay buffer shape mismatch: file {d['obs'].shape} vs buffer {self.obs_buf.shape}")
        self.obs_buf[:] = d["obs"]
        self.act_buf[:] = d["act"]
        self.rew_buf[:] = d["rew"]
        self.done_buf[:] = d["done"]
        self.pi_buf[:] = d["pi"]
        self.val_buf[:] = d["val"]
        self.ptr = int(d["ptr"])
        self.size = int(d["size"])
        if "prio" in d.files:  # buffers saved before prioritized replay existed stay uniform-valid
            self.prio_buf[:] = d["prio"]
            self.max_prio = float(d["max_prio"])
        else:
            self.prio_buf[:self.size] = 1.0
        # The environments are freshly reset on resume, so the next stored step starts a new
        # episode. Mark the last stored step terminal so no sampled window stitches the old
        # trajectory onto the new one.
        if self.size > 0:
            self.done_buf[(self.ptr - 1) % self.capacity, :] = True

    def update_priorities(self, t_idx: np.ndarray, env_idx: np.ndarray, prios: np.ndarray) -> None:
        prios = np.abs(prios) + 1e-6
        self.prio_buf[t_idx, env_idx] = prios
        self.max_prio = max(self.max_prio, float(prios.max()))

    def _valid_starts(self, t: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Vectorised validity of window starts (t, e): the K+1 frames t..t+K must be stored,
        must not straddle the ring write pointer, and steps t..t+K-1 must contain no done."""
        K = self.unroll_steps
        offs = np.arange(K + 1)
        if self.size < self.capacity:
            ok = t < self.size - K
            idx = np.minimum(t[:, None] + offs, self.size - 1)
        else:
            idx = (t[:, None] + offs) % self.capacity
            ok = ~np.any(idx == self.ptr, axis=1)
        ok &= ~np.any(self.done_buf[idx[:, :K], e[:, None]], axis=1)
        return ok

    def sample_trajectories(self, batch_size: int, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """Samples B windows of K steps from single-env trajectories.

        Returns observations as **uint8** (normalise on the GPU): the float32 version was 4x the
        host->device traffic and was built with per-sample Python loops (see
        docs/design/atari_mcts_complexity_and_sample_efficiency.md, section 1)."""
        K = self.unroll_steps
        if self.size <= K + 1:
            raise ValueError(f"Buffer size per env ({self.size}) is too small for K={K} unrolls.")
        n = self.size if self.size < self.capacity else self.capacity

        if self.priority_alpha > 0:
            p_all = self.prio_buf[:n].astype(np.float64) ** self.priority_alpha
            p_flat = (p_all / p_all.sum()).reshape(-1)
        t_list, e_list = [], []
        got = 0
        for _ in range(50):  # oversample and reject invalid starts; a few rounds always suffice
            m = max(2 * (batch_size - got), 64)
            if self.priority_alpha > 0:
                flat = np.random.choice(p_flat.size, size=m, p=p_flat)
                t_c, e_c = flat // self.num_envs, flat % self.num_envs
            else:
                t_c = np.random.randint(0, n, size=m)
                e_c = np.random.randint(0, self.num_envs, size=m)
            ok = self._valid_starts(t_c, e_c)
            t_list.append(t_c[ok]); e_list.append(e_c[ok])
            got += int(ok.sum())
            if got >= batch_size:
                break
        t_arr = np.concatenate(t_list)[:batch_size].astype(np.int64)
        e_arr = np.concatenate(e_list)[:batch_size].astype(np.int64)
        if t_arr.size < batch_size:  # pathological (almost every window crosses a done): repeat
            reps = np.resize(np.arange(t_arr.size), batch_size)
            t_arr, e_arr = t_arr[reps], e_arr[reps]
        B = batch_size

        idx = (t_arr[:, None] + np.arange(K + 1)) % self.capacity   # (B, K+1)
        frames = self.obs_buf[idx, e_arr[:, None]]                  # (B, K+1, 4, 84, 84) uint8
        obs_0 = frames[:, 0]
        target_future_obs = frames[:, 1:]
        actions_seq = self.act_buf[idx[:, :K], e_arr[:, None]]
        rewards_seq = self.rew_buf[idx[:, :K], e_arr[:, None]].astype(np.float32)
        pi_0 = self.pi_buf[t_arr, e_arr].astype(np.float32)
        val_0 = self.val_buf[t_arr, e_arr].astype(np.float32)

        if self.priority_alpha > 0:
            # Importance-sampling correction for the non-uniform draw, normalised to max 1.
            probs = p_flat.reshape(p_all.shape)[t_arr, e_arr]
            is_w = (probs * p_flat.size) ** (-beta)
            is_w = (is_w / is_w.max()).astype(np.float32)
        else:
            is_w = np.ones((B,), dtype=np.float32)

        return {
            "obs_0": torch.from_numpy(np.ascontiguousarray(obs_0)),
            "actions_seq": torch.from_numpy(np.ascontiguousarray(actions_seq)),
            "rewards_seq": torch.from_numpy(np.ascontiguousarray(rewards_seq)),
            "target_future_obs": torch.from_numpy(np.ascontiguousarray(target_future_obs)),
            "pi_0": torch.from_numpy(pi_0),
            "val_0": torch.from_numpy(val_0),
            "is_weights": torch.from_numpy(is_w),
            "t_idx": t_arr,
            "env_idx": e_arr,
        }



def augment_obs(x: torch.Tensor, mode: str, pad: int = 4) -> torch.Tensor:
    """Random shift (replicate-pad by `pad`, random crop back, per image) and optional
    intensity scaling, on float images in [0, 1] of shape (N, C, H, W). Matches the DrQ/SPR
    shift used by EfficientZero; done with one grid_sample, so it stays on the GPU."""
    n, c, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)
    out = F.grid_sample(padded, base_grid + shift, padding_mode="zeros", align_corners=False)
    if mode == "shift_intensity":
        noise = 1.0 + 0.05 * torch.randn((n, 1, 1, 1), device=x.device, dtype=x.dtype).clamp_(-2.0, 2.0)
        out = out * noise
    return out


def evaluate_agent_mcts(
    agent: ImpalaGTrXLAgent,
    mcts_engine: MCTSEngine,
    env_id: str,
    device: torch.device,
    num_episodes: int = 10,
    use_mcts: bool = True,
    mcts_temperature: float = 0.0,
    sticky_action_p: float = 0.0,
) -> Tuple[float, float]:
    """Evaluates agent performance with optional MCTS tree lookahead across independent episode seeds."""
    agent.eval()
    scores = []
    action_counts: Dict[int, int] = {}
    action_meanings = []

    for ep in range(num_episodes):
        eval_seed = 999 + ep * 13
        eval_env_fn = make_atari_env(env_id, seed=eval_seed, idx=0, noop_max=30, clip_reward=False, episodic_life=False)
        env = eval_env_fn()
        if sticky_action_p > 0.0:
            env = StickyActionEnv(env, p=sticky_action_p)

        action_meanings = env.unwrapped.get_action_meanings()
        has_fire = "FIRE" in action_meanings and len(action_meanings) >= 2

        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        last_lives = env.unwrapped.ale.lives()
        steps = 0

        # Initial FIRE on reset for Breakout
        if has_fire:
            step_res = env.step(1)
            action_counts[1] = action_counts.get(1, 0) + 1
            if len(step_res) == 5:
                obs, r, term, trunc, _ = step_res
                done = term or trunc
            else:
                obs, r, done, _ = step_res
            ep_ret += r
            steps += 1

        while not done and steps < 27000:
            obs_t = (torch.as_tensor(obs, device=device).float() / 255.0).unsqueeze(0)
            with torch.no_grad():
                if use_mcts:
                    latent_z, pol_repr = agent.encode_observation(obs_t)
                    probs, action, _ = mcts_engine.search_single(
                        latent_z[0],
                        agent,
                        device,
                        root_policy_repr=pol_repr[0],
                        add_dirichlet=False,
                        temperature=mcts_temperature,
                    )
                else:
                    logits, _, _ = agent(obs_t)
                    action = torch.argmax(logits, dim=-1).item()

            current_lives = env.unwrapped.ale.lives()
            if has_fire and current_lives < last_lives:
                action = 1
            last_lives = current_lives

            action_counts[action] = action_counts.get(action, 0) + 1
            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, term, trunc, _ = step_res
                done = term or trunc
            else:
                obs, reward, done, _ = step_res

            ep_ret += reward
            steps += 1

        env.close()
        scores.append(ep_ret)

    agent.train()
    total_actions = sum(action_counts.values())
    if total_actions > 0 and action_meanings:
        hist_str = ", ".join(
            f"{action_meanings[a] if a < len(action_meanings) else a}:{action_counts.get(a, 0)}"
            f" ({100.0 * action_counts.get(a, 0) / total_actions:.0f}%)"
            for a in sorted(action_counts)
        )
        print(f"[EVAL ACTIONS (mcts={use_mcts})] {hist_str}", flush=True)

    return float(np.mean(scores)), float(np.std(scores))



def train_mcts_offpolicy(
    env_id: str = "BreakoutNoFrameskip-v4",
    total_steps: int = 100_000,
    num_envs: int = 8,
    num_simulations: int = 16,
    eval_simulations: int = 16,
    unroll_steps: int = 5,
    replay_ratio: float = 0.5,
    batch_size: int = 128,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    buffer_capacity: int = 50_000,
    min_replay_size: int = 1_000,
    eval_interval: int = 10_000,
    eval_episodes: int = 10,
    target_tau: float = 0.005,
    collect_with_policy_only: bool = False,
    log_dir: str = "results/100k_benchmark/S049_mcts_offpolicy",
    device_str: str = "auto",
    seed: int = 42,
    use_mlflow: bool = True,
    mlflow_port: int = 10100,
    experiment_name: str = "Atari_MCTS_OffPolicy",
    run_label: Optional[str] = None,
    cnn_kaiming_init: bool = True,
    use_gru_gating: bool = True,
    reward_loss_weight: float = 1.0,
    ez_value_loss_weight: float = 0.25,
    consistency_loss_weight: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 1.0,
    resume_from: Optional[str] = None,
    start_step: Optional[int] = None,
    ckpt_interval: int = 2_500,
    reanalyze_ratio: float = 0.0,
    priority_alpha: float = 0.0,
    priority_beta: float = 1.0,
    augment: str = "none",
):
    """Off-policy MCTS trainer integrating latent lookahead with prioritized trajectory replay."""
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
    run_name = run_label or f"mcts_offpolicy_{env_id}_s{seed}_{int(time.time())}"
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(run_dir / "tb"))

    # MLflow Setup
    mlflow_run = None
    if use_mlflow and HAS_MLFLOW:
        try:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            tracking_uri = f"http://127.0.0.1:{mlflow_port}"
            try:
                import urllib.request
                urllib.request.urlopen(f"{tracking_uri}/health", timeout=3)
            except Exception:
                # No tracking server on this box: log to the shared file store instead of
                # silently dropping MLflow tracking altogether.
                from src.config.paths import mlruns_dir
                tracking_uri = mlruns_dir().resolve().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "env_id": env_id,
                "total_steps": total_steps,
                "num_envs": num_envs,
                "num_simulations": num_simulations,
                "eval_simulations": eval_simulations,
                "unroll_steps": unroll_steps,
                "replay_ratio": replay_ratio,
                "batch_size": batch_size,
                "lr": learning_rate,
                "target_tau": target_tau,
                "seed": seed,
                "collect_with_policy_only": collect_with_policy_only,
                "reanalyze_ratio": reanalyze_ratio,
                "priority_alpha": priority_alpha,
                "priority_beta": priority_beta,
                "augment": augment,
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
                "reward_loss_weight": reward_loss_weight,
                "ez_value_loss_weight": ez_value_loss_weight,
                "consistency_loss_weight": consistency_loss_weight,
                "resume_from": resume_from or "",
            })
            print(f"✓ MLflow Tracking Active: {tracking_uri} (Run: {mlflow_run.info.run_id})", flush=True)
        except Exception as e:
            print(f"! MLflow initialization failed ({e}), continuing with local logging only.", flush=True)

    print("==================================================================")
    print("--> Initializing Off-Policy MCTS + GTrXL + EfficientZero v2 Trainer")
    print(f"--> Target Steps:     {total_steps:,} | Vector Envs: {num_envs}")
    print(f"--> MCTS Sims:        {num_simulations} (Collect) / {eval_simulations} (Eval)")
    print(f"--> Replay Ratio:     {replay_ratio} updates/step | Batch Size: {batch_size}")
    print(f"--> Unroll Steps:     {unroll_steps} | Device: {device} | Target Tau: {target_tau}")
    print("==================================================================\n", flush=True)

    # 1. Initialize Environments
    envs = make_vector_atari_envs(env_id, num_envs=num_envs, seed=seed, noop_max=30, clip_reward=True, episodic_life=True)
    action_dim = envs.single_action_space.n

    # 2. Active Agent and Target Agent
    agent = ImpalaGTrXLAgent(
        action_dim=action_dim,
        in_channels=4,
        embed_dim=256,
        depth=4,
        num_heads=4,
        ffn_dim=1024,
        unroll_steps=unroll_steps,
        bg_init=0.0,
        use_gru_gating=use_gru_gating,
        cnn_kaiming_init=cnn_kaiming_init,
    ).to(device)

    target_agent = copy.deepcopy(agent).to(device)
    target_agent.eval()
    for p in target_agent.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(agent.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_eval = -float("inf")
    resumed_step = None

    if resume_from and os.path.exists(resume_from):
        print(f"--> Loading checkpoint state from {resume_from}...", flush=True)
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            agent.load_state_dict(ckpt["model"])
            if "target_model" in ckpt:
                target_agent.load_state_dict(ckpt["target_model"])
            else:
                target_agent.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "step" in ckpt:
                resumed_step = ckpt["step"]
            if "best_eval" in ckpt:
                best_eval = ckpt["best_eval"]
            if "rng_state" in ckpt:
                try:
                    rng = ckpt["rng_state"]
                    if "torch" in rng and rng["torch"] is not None:
                        t_state = rng["torch"]
                        if isinstance(t_state, torch.Tensor) and t_state.dtype != torch.uint8:
                            t_state = t_state.to(torch.uint8)
                        elif not isinstance(t_state, torch.Tensor):
                            t_state = torch.tensor(t_state, dtype=torch.uint8)
                        torch.set_rng_state(t_state.cpu())
                    if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                        c_state = rng["cuda"]
                        if isinstance(c_state, torch.Tensor) and c_state.dtype != torch.uint8:
                            c_state = c_state.to(torch.uint8)
                        elif not isinstance(c_state, torch.Tensor):
                            c_state = torch.tensor(c_state, dtype=torch.uint8)
                        torch.cuda.set_rng_state(c_state.cpu())
                    if "numpy" in rng and rng["numpy"] is not None:
                        np.random.set_state(rng["numpy"])
                    if "random" in rng and rng["random"] is not None:
                        random.setstate(rng["random"])
                except Exception as e:
                    print(f"! Warning: Could not restore RNG state ({e}), continuing with fresh RNG.", flush=True)
            print(f"✓ Resumed full training state at step {resumed_step:,} (best eval: {best_eval:.2f})!", flush=True)
        else:
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
            agent.load_state_dict(state_dict, strict=False)
            target_agent.load_state_dict(agent.state_dict())
            print(f"✓ Resumed model weights only from {resume_from}!", flush=True)

    mcts_engine = MCTSEngine(
        action_dim=action_dim,
        num_simulations=num_simulations,
        discount=gamma,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    )
    eval_mcts_engine = MCTSEngine(
        action_dim=action_dim,
        num_simulations=eval_simulations,
        discount=gamma,
        dirichlet_eps=0.0,
    )

    buffer = MCTSReplayBuffer(
        # Never smaller than the run's budget: EfficientZero keeps every transition, and a 50k ring
        # silently discarded the first half of a 100k run.
        capacity=max(buffer_capacity, total_steps),
        num_envs=num_envs,
        obs_shape=(4, 84, 84),
        action_dim=action_dim,
        unroll_steps=unroll_steps,
        priority_alpha=priority_alpha,
    )
    # Reanalyze (EfficientZero): refresh stored MCTS policy targets with a new search from the
    # EMA target network, so old replay data is not trained toward a stale search result.
    reanalyze_engine = MCTSEngine(
        action_dim=action_dim,
        num_simulations=num_simulations,
        discount=gamma,
        dirichlet_eps=0.0,
    ) if reanalyze_ratio > 0 else None

    obs, _ = envs.reset()

    # The replay buffer is saved beside checkpoint_latest.pt; restoring it is what makes a resume
    # a continuation rather than a restart on 1,000 random transitions.
    if resume_from:
        replay_path = Path(resume_from).with_name("replay_latest.npz")
        if replay_path.exists():
            buffer.load(replay_path)
            print(f"✓ Restored replay buffer ({buffer.total_transitions:,} transitions) from {replay_path}", flush=True)
        else:
            print(f"! No replay buffer beside {resume_from}; warming up a fresh one "
                  f"(the resumed run has NOT seen its earlier data in replay).", flush=True)

    # Warmup buffer with random / initial actions
    uniform_pi = np.full((num_envs, action_dim), 1.0 / action_dim, dtype=np.float32)
    zero_val = np.zeros((num_envs,), dtype=np.float32)
    if buffer.total_transitions < min_replay_size:
        print(f"--> Warming up replay buffer with {min_replay_size} initial transitions...", flush=True)

    while buffer.total_transitions < min_replay_size:
        actions = np.random.randint(0, action_dim, size=(num_envs,))
        step_res = envs.step(actions)
        if len(step_res) == 5:
            next_obs, rews, terms, truncs, _ = step_res
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_res

        buffer.add_batch(obs, actions, rews, dones, uniform_pi, zero_val)
        obs = next_obs

    print(f"✓ Buffer warmed up! ({buffer.total_transitions:,} transitions stored)\n", flush=True)

    # The checkpoint's own step always wins: overriding it is how the 38k/5k resumes were
    # mislabelled (weights from 30k / 2.5k counted as 38k / 5k). --start-step is only honoured
    # for weight-only checkpoints that carry no step.
    if resumed_step is not None:
        if start_step is not None and start_step != resumed_step:
            print(f"! Ignoring --start-step {start_step}: checkpoint records step {resumed_step}.", flush=True)
        global_step = resumed_step
    elif start_step is not None:
        global_step = start_step
    else:
        global_step = buffer.total_transitions
    print(f"--> Training starting at step {global_step:,} / {total_steps:,}\n", flush=True)

    # Wall-clock split per 1k env steps (host-side timers; GPU work is asynchronous, so a phase's
    # GPU time can land in the next phase that synchronises - read as shares, not exact costs).
    timing = {"collect": 0.0, "sample": 0.0, "update": 0.0}
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    updates_accumulated = 0.0
    session_start_step = global_step
    start_time = time.time()
    last_ckpt_step = global_step
    search_diag: Dict[str, float] = {}

    def save_checkpoint(tag: str, extra: Optional[Dict] = None) -> None:
        state = {
            "step": global_step,
            "model": agent.state_dict(),
            "target_model": target_agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "best_eval": best_eval,
            "config": {"num_simulations": num_simulations, "seed": seed, "env_id": env_id},
        }
        if extra:
            state.update(extra)
        tmp = ckpt_dir / f"checkpoint_{tag}.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, ckpt_dir / f"checkpoint_{tag}.pt")
        torch.save(agent.state_dict(), ckpt_dir / f"model_{tag}.pt")
        if tag == "latest":
            buffer.save(ckpt_dir / "replay_latest.npz")

    while global_step < total_steps:
        # A. Collect Transitions with MCTS (or direct policy if collect_with_policy_only)
        t_c = time.perf_counter()
        obs_norm = torch.as_tensor(obs, device=device).float() / 255.0
        with torch.no_grad():
            if collect_with_policy_only:
                logits, val, _ = agent(obs_norm)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                batch_probs = probs.cpu().numpy()
                batch_actions = actions.cpu().numpy()
                batch_values = val.squeeze(-1).cpu().numpy()
            else:
                latent_z, pol_repr = agent.encode_observation(obs_norm)
                batch_probs, batch_actions, batch_values = mcts_engine.search_batch(
                    latent_z,
                    agent,
                    device,
                    root_policy_reprs=pol_repr,
                    add_dirichlet=True,
                    temperature=1.0,
                )
                # How far search moves away from the network prior. ~0 means the tree is just
                # echoing the policy head, i.e. the learned model contributes nothing.
                prior = F.softmax(agent.actor_head(pol_repr).float(), dim=-1).cpu().numpy()
                visits = np.asarray(batch_probs, dtype=np.float64)
                search_diag = {
                    "search_kl_visits_prior": float(np.mean(np.sum(
                        visits * (np.log(visits + 1e-8) - np.log(prior + 1e-8)), axis=-1))),
                    "search_visit_entropy": float(np.mean(-np.sum(visits * np.log(visits + 1e-8), axis=-1))),
                    "search_action_differs_from_prior_argmax": float(np.mean(
                        np.argmax(visits, axis=-1) != np.argmax(prior, axis=-1))),
                }

        step_res = envs.step(batch_actions)
        if len(step_res) == 5:
            next_obs, rews, terms, truncs, _ = step_res
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_res

        buffer.add_batch(obs, batch_actions, rews, dones, batch_probs, batch_values)
        obs = next_obs
        global_step += num_envs
        timing["collect"] += time.perf_counter() - t_c

        # B. Off-Policy Gradient Updates
        updates_accumulated += num_envs * replay_ratio
        num_updates = int(updates_accumulated)
        updates_accumulated -= num_updates

        loss_dict = {}
        for _ in range(num_updates):
            t_s = time.perf_counter()
            batch = buffer.sample_trajectories(batch_size=batch_size, beta=priority_beta)
            # uint8 over the bus, normalised on the GPU (4x less host->device traffic).
            b_obs_0 = batch["obs_0"].to(device, non_blocking=True).float().div_(255.0)
            b_actions = batch["actions_seq"].to(device)
            b_rewards = batch["rewards_seq"].to(device)
            target_future = batch["target_future_obs"].to(device, non_blocking=True).float().div_(255.0)
            b_obs_raw = b_obs_0
            if augment != "none":
                # Independent random shift (+intensity) for the online input and every target
                # frame, as in SPR / EfficientZero: the representation must be invariant to it.
                b_obs_0 = augment_obs(b_obs_0, augment)
                Bt, Kt = target_future.shape[:2]
                target_future = augment_obs(
                    target_future.reshape(Bt * Kt, *target_future.shape[2:]), augment
                ).reshape(target_future.shape)
            timing["sample"] += time.perf_counter() - t_s
            t_u = time.perf_counter()
            b_pi_0 = batch["pi_0"].to(device)
            b_is_w = batch["is_weights"].to(device)

            if reanalyze_engine is not None:
                n_re = int(round(b_obs_0.shape[0] * reanalyze_ratio))
                if n_re > 0:
                    with torch.no_grad():
                        re_z, re_pol = target_agent.encode_observation(b_obs_raw[:n_re])
                        re_probs, _, _ = reanalyze_engine.search_batch(
                            re_z, target_agent, device, root_policy_reprs=re_pol,
                            add_dirichlet=False, temperature=1.0,
                        )
                    b_pi_0 = b_pi_0.clone()
                    b_pi_0[:n_re] = torch.as_tensor(np.asarray(re_probs), device=device, dtype=b_pi_0.dtype)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                # 1. Forward root observation through active agent
                logits_0, value_0, latent_z0 = agent(b_obs_0)

                # 2. Encode future target observations with TARGET agent
                B_sz, K_steps = b_actions.shape
                flat_targets = target_future.reshape(B_sz * K_steps, *target_future.shape[2:])
                with torch.no_grad():
                    flat_target_z, _ = target_agent.encode_observation(flat_targets)
                    flat_target_proj = target_agent.predictor.projector(flat_target_z)
                    target_projs = flat_target_proj.view(B_sz, K_steps, -1)
                    future_values = target_agent.critic_head(flat_target_z).squeeze(-1).view(B_sz, K_steps)

                # 3. Multi-Step GAE target returns bootstrapped with target critic
                with torch.no_grad():
                    v_root = value_0.squeeze(-1)
                    values_seq = torch.cat([v_root.unsqueeze(1), future_values], dim=1)
                    advantages = torch.zeros(B_sz, K_steps, device=device, dtype=torch.float32)
                    gae = torch.zeros(B_sz, device=device, dtype=torch.float32)
                    for k in reversed(range(K_steps)):
                        delta = b_rewards[:, k] + gamma * values_seq[:, k + 1] - values_seq[:, k]
                        gae = delta + gamma * gae_lambda * gae
                        advantages[:, k] = gae
                    target_returns = advantages + values_seq[:, :K_steps]

                # 4. Multi-Step Dynamics Unroll with active agent
                unroll = agent.unroll_branches(latent_z0, b_actions)

                # 5. Losses
                # A. Policy Cross-Entropy against MCTS visit distribution
                log_probs_0 = F.log_softmax(logits_0, dim=-1)
                # Per-sample losses weighted by the prioritized-replay IS weights (all 1 when uniform).
                policy_loss = (b_is_w * -(b_pi_0 * log_probs_0).sum(dim=-1)).mean()

                # B. Value Loss (root + unrolled dynamics values)
                root_v_loss = (b_is_w * F.smooth_l1_loss(
                    value_0.squeeze(-1), target_returns[:, 0], reduction="none")).mean()
                unroll_v_loss = 0.0
                reward_loss = 0.0
                consistency_loss = 0.0

                for k in range(unroll_steps):
                    unroll_v_loss = unroll_v_loss + F.smooth_l1_loss(unroll["values"][k].squeeze(-1), target_returns[:, k])
                    reward_loss = reward_loss + F.smooth_l1_loss(unroll["rewards"][k], b_rewards[:, k])
                    consistency_loss = consistency_loss + agent.predictor.compute_consistency_loss(unroll["projections"][k], target_projs[:, k])

                unroll_v_loss = unroll_v_loss / unroll_steps
                reward_loss = reward_loss / unroll_steps
                consistency_loss = consistency_loss / unroll_steps
                total_value_loss = 0.5 * (root_v_loss + unroll_v_loss)

                # C. Entropy regularization
                probs_0 = F.softmax(logits_0, dim=-1)
                entropy = -(probs_0 * log_probs_0).sum(dim=-1).mean()

                total_loss = (
                    policy_loss
                    + ez_value_loss_weight * total_value_loss
                    + reward_loss_weight * reward_loss
                    + consistency_loss_weight * consistency_loss
                    - ent_coef * entropy
                )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            if priority_alpha > 0:
                with torch.no_grad():
                    td_err = (value_0.squeeze(-1).float() - target_returns[:, 0].float()).abs().cpu().numpy()
                buffer.update_priorities(batch["t_idx"], batch["env_idx"], td_err)

            # EMA soft-update of target network
            with torch.no_grad():
                for param, target_param in zip(agent.parameters(), target_agent.parameters()):
                    target_param.data.mul_(1.0 - target_tau).add_(param.data, alpha=target_tau)

            timing["update"] += time.perf_counter() - t_u
            loss_dict = {
                "total": total_loss.item(),
                "policy": policy_loss.item(),
                "value": total_value_loss.item(),
                "reward": reward_loss.item(),
                "consistency": consistency_loss.item(),
                "entropy": entropy.item(),
            }

        # Model-health diagnostics on the last update batch (only when about to log).
        if global_step % 1000 < num_envs and loss_dict:
            with torch.no_grad():
                # Collapse check (SimSiam): per-dim std of L2-normalised projections across the
                # batch. Healthy ~ 1/sqrt(d); a collapsed encoder that maps everything to one
                # vector gives ~0 while the consistency loss sits at a perfect -1.
                d_proj = target_projs.shape[-1]
                tgt_n = F.normalize(target_projs[:, 0].float(), dim=-1)
                onl_n = F.normalize(unroll["projections"][0].float(), dim=-1)
                loss_dict["proj_std_target"] = tgt_n.std(dim=0).mean().item()
                loss_dict["proj_std_online"] = onl_n.std(dim=0).mean().item()
                loss_dict["proj_std_healthy_ref"] = 1.0 / (d_proj ** 0.5)
                # Reward head on the sparse non-zero (brick) rewards. The reward loss alone is
                # ~0.003 because predicting 0 everywhere is almost always right.
                r_pred = torch.stack([r.float() for r in unroll["rewards"]], dim=1)
                nz = b_rewards != 0
                loss_dict["reward_nonzero_frac"] = nz.float().mean().item()
                if nz.any():
                    loss_dict["reward_recall_nonzero"] = ((r_pred[nz] * b_rewards[nz]) > 0.5).float().mean().item()
                if (~nz).any():
                    loss_dict["reward_false_pos"] = (r_pred[~nz].abs() > 0.5).float().mean().item()
                # Value vs bootstrapped target: explained variance.
                tv = target_returns[:, 0].float()
                ev = 1.0 - (tv - value_0.squeeze(-1).float()).var() / (tv.var() + 1e-8)
                loss_dict["value_explained_var"] = ev.item()
            loss_dict.update(search_diag)

        # Logging
        if global_step % 1000 < num_envs and loss_dict:
            # Steps taken in THIS process only: dividing the absolute step by session time
            # inflated SPS by the resume offset (a "37 SPS" run was really doing ~2.6).
            sps = int((global_step - session_start_step) / max(time.time() - start_time, 1e-4))
            print(
                f"Step {global_step:6d}/{total_steps:6d} | SPS: {sps:3d} | "
                f"Total: {loss_dict['total']:.4f} | Pol: {loss_dict['policy']:.4f} | "
                f"Val: {loss_dict['value']:.4f} | Rew: {loss_dict['reward']:.4f} | "
                f"Sim: {loss_dict['consistency']:.4f} | Ent: {loss_dict['entropy']:.3f}",
                flush=True,
            )
            print(
                f"    [diag] proj_std tgt/onl {loss_dict.get('proj_std_target', float('nan')):.4f}/"
                f"{loss_dict.get('proj_std_online', float('nan')):.4f} (healthy~{loss_dict.get('proj_std_healthy_ref', float('nan')):.4f}) | "
                f"rew recall {loss_dict.get('reward_recall_nonzero', float('nan')):.2f} "
                f"fp {loss_dict.get('reward_false_pos', float('nan')):.3f} "
                f"(nz {loss_dict.get('reward_nonzero_frac', float('nan')):.3f}) | "
                f"V ev {loss_dict.get('value_explained_var', float('nan')):.2f} | "
                f"search KL {loss_dict.get('search_kl_visits_prior', float('nan')):.3f} "
                f"H {loss_dict.get('search_visit_entropy', float('nan')):.3f} "
                f"argmax-changed {loss_dict.get('search_action_differs_from_prior_argmax', float('nan')):.2f}",
                flush=True,
            )
            for k, v in loss_dict.items():
                writer.add_scalar(f"losses/{k}", v, global_step)
                if mlflow_run:
                    try:
                        mlflow.log_metric(f"loss_{k}", v, step=global_step)
                    except Exception:
                        pass
            writer.add_scalar("charts/SPS", sps, global_step)

        if global_step % 1000 < num_envs and loss_dict:
            tot = max(sum(timing.values()), 1e-9)
            print("    [time/1k] " + " | ".join(
                f"{k} {v:.0f}s ({100 * v / tot:.0f}%)" for k, v in timing.items()), flush=True)
            for k, v in timing.items():
                writer.add_scalar(f"time/{k}_s_per_1k", v, global_step)
                if mlflow_run:
                    try:
                        mlflow.log_metric(f"time_{k}_s_per_1k", v, step=global_step)
                    except Exception:
                        pass
            timing = dict.fromkeys(timing, 0.0)

        # Periodic Evaluation with MCTS and Raw Policy
        if global_step % eval_interval < num_envs or global_step >= total_steps:
            print(f"\n--> Running Evaluation at Step {global_step:,} ({eval_episodes} episodes)...", flush=True)
            # 1. Evaluate with MCTS lookahead
            mean_eval_mcts, std_eval_mcts = evaluate_agent_mcts(
                agent,
                eval_mcts_engine,
                env_id,
                device,
                num_episodes=eval_episodes,
                use_mcts=True,
                mcts_temperature=0.0,
            )
            hns_mcts = compute_hns(mean_eval_mcts, env_id)

            # 2. Evaluate with raw policy (no tree search)
            mean_eval_raw, std_eval_raw = evaluate_agent_mcts(
                agent,
                eval_mcts_engine,
                env_id,
                device,
                num_episodes=eval_episodes,
                use_mcts=False,
                mcts_temperature=0.0,
            )
            hns_raw = compute_hns(mean_eval_raw, env_id)

            print(
                f"[EVALUATION] Step {global_step:6d} | "
                f"MCTS Score: {mean_eval_mcts:.2f} +/- {std_eval_mcts:.2f} "
                f"(SE {std_eval_mcts / eval_episodes ** 0.5:.2f}, HNS: {hns_mcts:.1f}%) | "
                f"Raw Policy: {mean_eval_raw:.2f} +/- {std_eval_raw:.2f} "
                f"(SE {std_eval_raw / eval_episodes ** 0.5:.2f}, HNS: {hns_raw:.1f}%) | n={eval_episodes}\n",
                flush=True,
            )

            writer.add_scalar("eval/mean_score_mcts", mean_eval_mcts, global_step)
            writer.add_scalar("eval/hns_mcts", hns_mcts, global_step)
            writer.add_scalar("eval/mean_score_raw", mean_eval_raw, global_step)
            writer.add_scalar("eval/hns_raw", hns_raw, global_step)
            if mlflow_run:
                try:
                    mlflow.log_metric("eval_score_mcts", mean_eval_mcts, step=global_step)
                    mlflow.log_metric("eval_hns_mcts", hns_mcts, step=global_step)
                    mlflow.log_metric("eval_score_raw", mean_eval_raw, step=global_step)
                    mlflow.log_metric("eval_hns_raw", hns_raw, step=global_step)
                except Exception:
                    pass

            eval_extra = {"mean_eval_mcts": mean_eval_mcts, "mean_eval_raw": mean_eval_raw}
            if mean_eval_mcts > best_eval:
                best_eval = mean_eval_mcts
                # "best" is for picking weights to report, never for resuming: resume from
                # checkpoint_latest.pt (Gate 2 was resumed from a 2.5k "best" labelled 5k).
                save_checkpoint("best", eval_extra)
                print(f"✓ Saved new best checkpoint (Score: {best_eval:.2f})", flush=True)
            save_checkpoint("latest", eval_extra)
            last_ckpt_step = global_step
            print(f"✓ Saved resumable checkpoint_latest.pt + replay buffer at step {global_step:,}", flush=True)
        elif global_step - last_ckpt_step >= ckpt_interval:
            save_checkpoint("latest")
            last_ckpt_step = global_step
            print(f"✓ Saved resumable checkpoint_latest.pt + replay buffer at step {global_step:,}", flush=True)

    envs.close()
    writer.close()
    if mlflow_run:
        try:
            mlflow.end_run()
        except Exception:
            pass

    print("\n==================================================================")
    print(f"✓ Off-Policy MCTS Training Complete! Peak Eval: {best_eval:.2f}")
    print("==================================================================\n", flush=True)
    return best_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Off-Policy MCTS Trainer for Atari 100k")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--eval-simulations", type=int, default=16)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--replay-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-replay-size", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--collect-with-policy-only", action="store_true",
                        help="Act with the raw policy during collection. NOT a search ablation: the "
                             "policy target then becomes the policy's own output, so it has no "
                             "improvement signal and drifts to uniform (policy loss == ln|A|). The "
                             "search-vs-no-search comparison is the dual MCTS/raw evaluation.")
    parser.add_argument("--reanalyze-ratio", type=float, default=0.0,
                        help="Fraction of each batch whose policy target is recomputed by a fresh target-network search")
    parser.add_argument("--priority-alpha", type=float, default=0.0, help="Prioritized replay exponent (0 = uniform)")
    parser.add_argument("--priority-beta", type=float, default=1.0, help="Importance-sampling exponent for prioritized replay")
    parser.add_argument("--augment", type=str, default="none", choices=["none", "shift", "shift_intensity"],
                        help="DrQ/SPR/EfficientZero-style augmentation of root and consistency-target frames")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=2500,
                        help="Env steps between resumable checkpoint_latest.pt + replay buffer saves")
    parser.add_argument("--log-dir", type=str, default="results/100k_benchmark/S049_mcts_offpolicy")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to model checkpoint to resume from")
    parser.add_argument("--start-step", type=int, default=None, help="Starting global step count")
    args = parser.parse_args()

    train_mcts_offpolicy(
        env_id=args.env_id,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        num_simulations=args.num_simulations,
        eval_simulations=args.eval_simulations,
        unroll_steps=args.unroll_steps,
        replay_ratio=args.replay_ratio,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_replay_size=args.min_replay_size,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        target_tau=args.target_tau,
        collect_with_policy_only=args.collect_with_policy_only,
        seed=args.seed,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        start_step=args.start_step,
        ckpt_interval=args.ckpt_interval,
        reanalyze_ratio=args.reanalyze_ratio,
        priority_alpha=args.priority_alpha,
        priority_beta=args.priority_beta,
        augment=args.augment,
        ent_coef=args.ent_coef,
        reward_loss_weight=args.reward_loss_weight,
        ez_value_loss_weight=args.value_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        run_label=args.run_label,
    )


