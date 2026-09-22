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
    """Off-policy circular replay buffer storing observations, actions, rewards, and MCTS targets."""
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        action_dim: int = 4,
        unroll_steps: int = 5,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.unroll_steps = unroll_steps

        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.bool_)
        self.pi_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.val_buf = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        done_batch: np.ndarray,
        pi_batch: np.ndarray,
        val_batch: np.ndarray,
    ):
        N = len(act_batch)
        for i in range(N):
            self.obs_buf[self.ptr] = obs_batch[i]
            self.act_buf[self.ptr] = act_batch[i]
            self.rew_buf[self.ptr] = rew_batch[i]
            self.done_buf[self.ptr] = done_batch[i]
            self.pi_buf[self.ptr] = pi_batch[i]
            self.val_buf[self.ptr] = val_batch[i]

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample_trajectories(self, batch_size: int) -> Dict[str, torch.Tensor]:
        K = self.unroll_steps
        max_idx = self.size - K - 1
        if max_idx <= 0:
            raise ValueError(f"Buffer size ({self.size}) is too small for K={K} unrolls.")

        valid_indices = []
        attempts = 0
        max_attempts = batch_size * 50

        while len(valid_indices) < batch_size and attempts < max_attempts:
            idx = np.random.randint(0, max_idx)
            # Avoid sampling windows that cross episode boundaries
            if not np.any(self.done_buf[idx : idx + K]):
                valid_indices.append(idx)
            attempts += 1

        if len(valid_indices) < batch_size:
            # Fallback: take what we have
            valid_indices.extend(valid_indices[: batch_size - len(valid_indices)])

        idx_arr = np.array(valid_indices)
        obs_0 = torch.from_numpy(self.obs_buf[idx_arr]).float() / 255.0
        actions_seq = torch.from_numpy(np.stack([self.act_buf[idx_arr + k] for k in range(K)], axis=1)).long()
        rewards_seq = torch.from_numpy(np.stack([self.rew_buf[idx_arr + k] for k in range(K)], axis=1)).float()
        pi_0 = torch.from_numpy(self.pi_buf[idx_arr]).float()
        val_0 = torch.from_numpy(self.val_buf[idx_arr]).float()

        # Stack future target observations: (B, K, 4, 84, 84)
        future_obs_list = [self.obs_buf[idx_arr + k + 1] for k in range(K)]
        target_future_obs = torch.from_numpy(np.stack(future_obs_list, axis=1)).float() / 255.0

        return {
            "obs_0": obs_0,
            "actions_seq": actions_seq,
            "rewards_seq": rewards_seq,
            "target_future_obs": target_future_obs,
            "pi_0": pi_0,
            "val_0": val_0,
        }


def evaluate_agent_mcts(
    agent: ImpalaGTrXLAgent,
    mcts_engine: MCTSEngine,
    env_id: str,
    device: torch.device,
    num_episodes: int = 5,
    use_mcts: bool = True,
    mcts_temperature: float = 0.0,
    sticky_action_p: float = 0.0,
) -> Tuple[float, float]:
    """Evaluates agent performance with optional MCTS tree lookahead."""
    eval_env_fn = make_atari_env(env_id, seed=999, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    env = eval_env_fn()
    if sticky_action_p > 0.0:
        env = StickyActionEnv(env, p=sticky_action_p)

    agent.eval()
    scores = []
    action_counts: Dict[int, int] = {}
    action_meanings = env.unwrapped.get_action_meanings()
    has_fire = "FIRE" in action_meanings and len(action_meanings) >= 2

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        stuck_counter = 0
        last_lives = env.unwrapped.ale.lives()

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

        while not done:
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
            if has_fire and (current_lives < last_lives or stuck_counter > 30):
                action = 1
                stuck_counter = 0
            last_lives = current_lives

            action_counts[action] = action_counts.get(action, 0) + 1
            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, term, trunc, _ = step_res
                done = term or trunc
            else:
                obs, reward, done, _ = step_res

            ep_ret += reward
            stuck_counter = 0 if reward != 0 else (stuck_counter + 1)

        scores.append(ep_ret)

    env.close()
    agent.train()

    total_actions = sum(action_counts.values())
    if total_actions > 0:
        hist_str = ", ".join(
            f"{action_meanings[a] if a < len(action_meanings) else a}:{action_counts.get(a, 0)}"
            f" ({100.0 * action_counts.get(a, 0) / total_actions:.0f}%)"
            for a in sorted(action_counts)
        )
        print(f"[EVAL ACTIONS] {hist_str}", flush=True)

    return float(np.mean(scores)), float(np.std(scores))


def train_mcts_offpolicy(
    env_id: str = "BreakoutNoFrameskip-v4",
    total_steps: int = 100_000,
    num_envs: int = 8,
    num_simulations: int = 35,
    eval_simulations: int = 35,
    unroll_steps: int = 5,
    replay_ratio: float = 0.5,
    batch_size: int = 128,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    buffer_capacity: int = 50_000,
    min_replay_size: int = 1_000,
    eval_interval: int = 10_000,
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
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "env_id": env_id,
                "total_steps": total_steps,
                "num_envs": num_envs,
                "num_simulations": num_simulations,
                "unroll_steps": unroll_steps,
                "replay_ratio": replay_ratio,
                "batch_size": batch_size,
                "lr": learning_rate,
                "seed": seed,
            })
            print(f"✓ MLflow Tracking Active: {tracking_uri} (Run: {mlflow_run.info.run_id})", flush=True)
        except Exception as e:
            print(f"! MLflow initialization failed ({e}), continuing with local logging only.", flush=True)

    print("==================================================================")
    print("--> Initializing Off-Policy MCTS + GTrXL + EfficientZero v2 Trainer")
    print(f"--> Target Steps:     {total_steps:,} | Vector Envs: {num_envs}")
    print(f"--> MCTS Sims:        {num_simulations} (Collect) / {eval_simulations} (Eval)")
    print(f"--> Replay Ratio:     {replay_ratio} updates/step | Batch Size: {batch_size}")
    print(f"--> Unroll Steps:     {unroll_steps} | Device: {device}")
    print("==================================================================\n", flush=True)

    # 1. Initialize Environments
    envs = make_vector_atari_envs(env_id, num_envs=num_envs, seed=seed, noop_max=30, clip_reward=True, episodic_life=True)
    action_dim = envs.single_action_space.n

    # 2. Agent & MCTS Engines
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

    optimizer = optim.AdamW(agent.parameters(), lr=learning_rate, weight_decay=1e-4)
    buffer = MCTSReplayBuffer(
        capacity=buffer_capacity,
        obs_shape=(4, 84, 84),
        action_dim=action_dim,
        unroll_steps=unroll_steps,
    )

    obs, _ = envs.reset()
    start_time = time.time()
    best_eval = -float("inf")

    # Warmup buffer with random / initial actions
    print(f"--> Warming up replay buffer with {min_replay_size} initial transitions...", flush=True)
    uniform_pi = np.full((num_envs, action_dim), 1.0 / action_dim, dtype=np.float32)
    zero_val = np.zeros((num_envs,), dtype=np.float32)

    while buffer.size < min_replay_size:
        actions = np.random.randint(0, action_dim, size=(num_envs,))
        step_res = envs.step(actions)
        if len(step_res) == 5:
            next_obs, rews, terms, truncs, _ = step_res
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_res

        buffer.add_batch(obs, actions, rews, dones, uniform_pi, zero_val)
        obs = next_obs

    print(f"✓ Buffer warmed up! ({buffer.size:,} transitions stored)\n", flush=True)

    global_step = buffer.size
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    updates_accumulated = 0.0

    while global_step < total_steps:
        # A. Collect Transitions with MCTS
        obs_norm = torch.as_tensor(obs, device=device).float() / 255.0
        with torch.no_grad():
            latent_z, pol_repr = agent.encode_observation(obs_norm)
            batch_probs, batch_actions, batch_values = mcts_engine.search_batch(
                latent_z,
                agent,
                device,
                root_policy_reprs=pol_repr,
                add_dirichlet=True,
                temperature=1.0,
            )

        step_res = envs.step(batch_actions)
        if len(step_res) == 5:
            next_obs, rews, terms, truncs, _ = step_res
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_res

        buffer.add_batch(obs, batch_actions, rews, dones, batch_probs, batch_values)
        obs = next_obs
        global_step += num_envs

        # B. Off-Policy Gradient Updates
        updates_accumulated += num_envs * replay_ratio
        num_updates = int(updates_accumulated)
        updates_accumulated -= num_updates

        loss_dict = {}
        for _ in range(num_updates):
            batch = buffer.sample_trajectories(batch_size=batch_size)
            b_obs_0 = batch["obs_0"].to(device)
            b_actions = batch["actions_seq"].to(device)
            b_rewards = batch["rewards_seq"].to(device)
            target_future = batch["target_future_obs"].to(device)
            b_pi_0 = batch["pi_0"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                # 1. Forward root observation
                logits_0, value_0, latent_z0 = agent(b_obs_0)

                # 2. Encode future target observations in one batch pass
                B_sz, K_steps = b_actions.shape
                flat_targets = target_future.reshape(B_sz * K_steps, *target_future.shape[2:])
                with torch.no_grad():
                    flat_target_z, _ = agent.encode_observation(flat_targets)
                    flat_target_proj = agent.predictor.projector(flat_target_z)
                    target_projs = flat_target_proj.view(B_sz, K_steps, -1)
                    future_values = agent.critic_head(flat_target_z).squeeze(-1).view(B_sz, K_steps)

                # 3. Multi-Step GAE target returns
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

                # 4. Multi-Step Dynamics Unroll
                unroll = agent.unroll_branches(latent_z0, b_actions)

                # 5. Losses
                # A. Policy Cross-Entropy against MCTS visit distribution
                log_probs_0 = F.log_softmax(logits_0, dim=-1)
                policy_loss = -(b_pi_0 * log_probs_0).sum(dim=-1).mean()

                # B. Value Loss (root + unrolled dynamics values)
                root_v_loss = F.smooth_l1_loss(value_0.squeeze(-1), target_returns[:, 0])
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

            loss_dict = {
                "total": total_loss.item(),
                "policy": policy_loss.item(),
                "value": total_value_loss.item(),
                "reward": reward_loss.item(),
                "consistency": consistency_loss.item(),
                "entropy": entropy.item(),
            }

        # Logging
        if global_step % 1000 < num_envs and loss_dict:
            sps = int(global_step / max(time.time() - start_time, 1e-4))
            print(
                f"Step {global_step:6d}/{total_steps:6d} | SPS: {sps:3d} | "
                f"Total: {loss_dict['total']:.4f} | Pol: {loss_dict['policy']:.4f} | "
                f"Val: {loss_dict['value']:.4f} | Rew: {loss_dict['reward']:.4f} | "
                f"Sim: {loss_dict['consistency']:.4f} | Ent: {loss_dict['entropy']:.3f}",
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

        # Periodic Evaluation with MCTS
        if global_step % eval_interval < num_envs or global_step >= total_steps:
            print(f"\n--> Running MCTS Evaluation at Step {global_step:,} ({eval_simulations} simulations)...", flush=True)
            mean_eval, std_eval = evaluate_agent_mcts(
                agent,
                eval_mcts_engine,
                env_id,
                device,
                num_episodes=5,
                use_mcts=True,
                mcts_temperature=0.0,
            )
            hns = compute_hns(mean_eval, env_id)
            print(f"[EVALUATION] Step {global_step:,} | Score: {mean_eval:.2f} +/- {std_eval:.2f} | HNS: {hns:.1f}%\n", flush=True)

            writer.add_scalar("eval/mean_score", mean_eval, global_step)
            writer.add_scalar("eval/hns", hns, global_step)
            if mlflow_run:
                try:
                    mlflow.log_metric("eval_score", mean_eval, step=global_step)
                    mlflow.log_metric("eval_hns", hns, step=global_step)
                except Exception:
                    pass

            torch.save(agent.state_dict(), ckpt_dir / "model_latest.pt")
            if mean_eval > best_eval:
                best_eval = mean_eval
                torch.save(agent.state_dict(), ckpt_dir / "model_best.pt")
                print(f"✓ Saved new best MCTS checkpoint (Score: {best_eval:.2f})", flush=True)

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
    parser.add_argument("--num-simulations", type=int, default=35)
    parser.add_argument("--eval-simulations", type=int, default=35)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--replay-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-replay-size", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default="results/100k_benchmark/S049_mcts_offpolicy")
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
        seed=args.seed,
        log_dir=args.log_dir,
    )
