"""EfficientZero V2 training recipe for Atari, with a swappable trunk (resnet = EZ-V2, gtrxl = thesis).

Matches EZ-V2's Atari config (ez/config/exp/atari_breakout.yaml) and training step
(ez/agents/base.py:update_weights, ez/worker/batch_worker.py:prepare_reward_value / _policy_reanalyze):
  * RGB 96x96, 4-frame stack, episodic life, reward clipping, 3,000-step training episodes;
  * discount 0.997 per frame -> 0.997**4 per agent step; unroll 5, 5-step bootstrapped value targets;
  * Gumbel search (16 sims, top-4, sequential halving) for acting, reanalysing *every* sampled state's
    policy (reanalyze_ratio 1.0) and search-based value estimates ("mixed" value target);
  * 601-bin h-transformed categorical value and value-prefix (LSTM) heads, SimSiam consistency (w=5);
  * prioritised replay alpha=beta=1 on |V - target|, new data at max priority;
  * SGD lr 0.2 (1% warm-up), momentum 0.9, wd 1e-4, batch 256, grad-norm 5, ~1 update per env step
    plus 20% offline updates at the end, target network hard-copied every 200 updates;
  * windows run up to (and past) episode ends with masks, so life-loss transitions are trained on.
Differences from EZ-V2 that are deliberate: synchronous single-process loop (collection and training
interleaved at a fixed update ratio instead of Ray workers), and `--trunk gtrxl`.
"""
import argparse
import copy
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from atari_qwen.envs.atari_wrappers import make_vector_atari_envs
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from atari_qwen.mcts.gumbel_mcts import GumbelMCTS, GraphedSearch
from atari_qwen.training.reanalysis_cache import ReanalysisCache
from atari_qwen.eval.evaluate import compute_hns
from atari_qwen.training import mixer_audit


class AsyncCheckpointSaver:
    """Asynchronous background worker that saves model weights and replay buffers without stalling training."""
    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            tag, ckpt_data, ckpt_path, replay_data, replay_path = item
            try:
                if ckpt_data is not None and ckpt_path is not None:
                    tmp_ckpt = Path(str(ckpt_path) + ".tmp")
                    torch.save(ckpt_data, tmp_ckpt)
                    os.replace(tmp_ckpt, ckpt_path)
                if replay_data is not None and replay_path is not None:
                    tmp_rp = Path(str(replay_path) + ".tmp.npz")
                    np.savez(tmp_rp, **replay_data)
                    os.replace(tmp_rp, replay_path)
            except Exception as e:
                print(f"[AsyncCheckpointSaver Error] {e}", file=sys.stderr, flush=True)
            finally:
                self.queue.task_done()

    def save_async(self, tag: str, model, target, opt, env_steps: int, updates: int, best: float, args_dict: dict,
                   ckpt_dir: Path, replay=None, extra=None):
        ckpt_data = {
            "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "target": {k: v.cpu().clone() for k, v in target.state_dict().items()},
            "opt": {k: v for k, v in opt.state_dict().items()},
            "env_steps": env_steps,
            "updates": updates,
            "best": best,
            "args": args_dict
        }
        if extra:
            ckpt_data.update(extra)
        ckpt_path = ckpt_dir / f"checkpoint_{tag}.pt"

        replay_data = None
        replay_path = None
        if tag == "latest" and replay is not None:
            replay_path = ckpt_dir / "replay_latest.npz"
            n = replay.size
            replay_data = {
                "frames": replay.frames[:n].cpu().numpy(),
                "act": replay.act[:n].copy(),
                "rew": replay.rew[:n].copy(),
                "done": replay.done[:n].copy(),
                "ep_start": replay.ep_start[:n].copy(),
                "pi": replay.pi[:n].copy(),
                "root_v": replay.root_v[:n].copy(),
                "prio": replay.prio[:n].copy(),
                "cur_start": replay.cur_start.copy() if hasattr(replay.cur_start, "copy") else replay.cur_start,
                "max_prio": np.float64(replay.max_prio)
            }

        self.queue.put((tag, ckpt_data, ckpt_path, replay_data, replay_path))

    def join(self):
        self.queue.join()



# ------------------------------------------------------------------------------------------------
# Replay: (T, E) grid of single frames; stacks and windows are rebuilt at sample time.
# ------------------------------------------------------------------------------------------------
class EZReplay:
    def __init__(self, steps_per_env: int, num_envs: int, frame_shape, action_dim: int, n_stack: int = 4,
                 alpha: float = 1.0, device="cpu"):
        T, E = steps_per_env, num_envs
        self.T, self.E, self.n_stack, self.alpha, self.device = T, E, n_stack, alpha, device
        # newest frame of obs_t, kept on the GPU: a 100k-step RGB 96x96 run is ~2.8 GB, and each update
        # gathers ~3k stacks (~330 MB) that would otherwise be copied on the host.
        self.frames = torch.zeros((T, E) + tuple(frame_shape), dtype=torch.uint8, device=device)
        self.act = np.zeros((T, E), dtype=np.int64)
        self.rew = np.zeros((T, E), dtype=np.float32)
        self.done = np.zeros((T, E), dtype=np.bool_)
        self.ep_start = np.zeros((T, E), dtype=np.int64)
        self.pi = np.zeros((T, E, action_dim), dtype=np.float32)
        self.root_v = np.zeros((T, E), dtype=np.float32)
        self.prio = np.zeros((T, E), dtype=np.float64)
        self.cur_start = np.zeros(E, dtype=np.int64)
        self.size = 0
        self.max_prio = 1.0

    @property
    def transitions(self) -> int:
        return self.size * self.E

    def add(self, newest_frames, actions, rewards, dones, policies, root_values):
        t = self.size
        if t >= self.T:
            raise RuntimeError("replay full: steps_per_env too small for the run")
        self.frames[t] = torch.as_tensor(newest_frames, device=self.device)
        self.act[t], self.rew[t], self.done[t] = actions, rewards, dones
        self.pi[t], self.root_v[t] = policies, root_values
        self.ep_start[t] = self.cur_start
        # EZ-V2 (replay_buffer.save_trajectory): new data enters at the buffer's *current* max priority. This
        # used a running maximum that never came down, so one early outlier error (a value head inflating
        # itself, S058a) set the entry priority of every later transition.
        self.max_prio = float(self.prio[:t].max()) if t > 0 else 1.0
        self.prio[t] = self.max_prio
        self.cur_start = np.where(dones, t + 1, self.cur_start)
        self.size += 1

    def stacks(self, t: np.ndarray, e: np.ndarray) -> torch.Tensor:
        """float (n, n_stack*C, H, W) observation stacks in [0, 1] for (t, e) pairs, padded with the
        episode's first frame (what FrameStack returns after a reset)."""
        offs = np.arange(-self.n_stack + 1, 1)
        idx = np.maximum(t[:, None] + offs, self.ep_start[t, e][:, None])       # (n, n_stack)
        ti = torch.as_tensor(idx, device=self.device)
        ei = torch.as_tensor(np.broadcast_to(e[:, None], idx.shape).copy(), device=self.device)
        f = self.frames[ti, ei]                                                   # (n, n_stack, C, H, W)
        return f.reshape(f.shape[0], -1, *f.shape[3:]).float().div_(255.0)

    def update_priorities(self, t, e, p):
        self.prio[t, e] = np.asarray(p, dtype=np.float64)

    def sample(self, B: int, horizon: int, beta: float = 1.0):
        """(t, e) window starts with horizon future steps stored, priority-sampled; IS weights.

        Prioritised (alpha > 0) sampling follows EZ-V2's _prepare_batch_context: no repeats within a batch,
        and IS weights normalised by their max and clipped to [0.1, 1]. This drew with replacement and left the
        weights unclipped, so at alpha = 1 a few high-error (reward) windows could fill many slots of one batch
        while every one of them carried a near-zero weight (S-067: the prioritised screens miscalibrated the
        reward head). Uniform sampling keeps its original draw, so alpha = 0 runs reproduce bit for bit."""
        n = self.size - horizon
        if n <= 0:
            raise ValueError("not enough data for a window")
        p = self.prio[:n].reshape(-1) ** self.alpha
        p = p / p.sum()
        if self.alpha > 0:
            flat = np.random.choice(p.size, size=B, p=p, replace=p.size < B)
        else:
            flat = np.random.choice(p.size, size=B, p=p)
        t, e = flat // self.E, flat % self.E
        w = (p[flat] * p.size) ** (-beta)
        w = w / w.max()
        if self.alpha > 0:
            w = w.clip(0.1, 1.0)
        return t, e, w.astype(np.float32)

    def save(self, path: Path):
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(tmp, frames=self.frames[:self.size].cpu().numpy(), act=self.act[:self.size], rew=self.rew[:self.size],
                 done=self.done[:self.size], ep_start=self.ep_start[:self.size], pi=self.pi[:self.size],
                 root_v=self.root_v[:self.size], prio=self.prio[:self.size], cur_start=self.cur_start,
                 max_prio=np.float64(self.max_prio))
        os.replace(tmp, path)

    def load(self, path: Path):
        d = np.load(path)
        n = d["frames"].shape[0]
        self.frames[:n] = torch.as_tensor(d["frames"], device=self.device)
        for k in ("act", "rew", "done", "ep_start", "pi", "root_v", "prio"):
            getattr(self, k)[:n] = d[k]
        self.size, self.max_prio = n, float(d["max_prio"])
        # envs restart on resume: close every open episode so no window stitches two episodes
        self.done[n - 1] = True
        self.cur_start[:] = n


# ------------------------------------------------------------------------------------------------
def augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """EZ-V2 transforms: random shift (replicate pad 4, crop; one shift per sample across the whole
    stack) then intensity 1 + 0.05 * clip(N(0,1), -2, 2)."""
    n, c, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device).float() * 2.0 / (h + 2 * pad)
    out = F.grid_sample(xp, base + shift, padding_mode="zeros", align_corners=False)
    noise = 1.0 + 0.05 * torch.randn(n, 1, 1, 1, device=x.device).clamp(-2.0, 2.0)
    return out * noise


def mixer_param_groups(model: nn.Module, weight_decay: float):
    """TODO B66: AdamW groups of the mixer as CARLA builds them for the same Qwen network (src/training/wor_optim.py,
    decay_gates_and_norms=0): weight decay on matrices, none on biases, RMSNorm gains, alpha gates, positional tables and the
    B69 policy token (CARLA's NO_DECAY_SUFFIXES)."""
    decay, no_decay = [], []
    for n, q in model.named_parameters():
        if "mixer" in n:
            (no_decay if q.ndim <= 1 or n.endswith((".pos", ".rel_table", ".policy_token")) else decay).append(q)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def carla_lr(updates: int, total_updates: int, peak: float, warmup_frac: float, eta_min: float = 1e-6) -> float:
    """TODO B66: CARLA's per-step schedule (wor_optim.build_warmup_cosine_scheduler) on the update clock: linear warm-up over
    warmup_frac of the run, then cosine decay from peak to eta_min."""
    total = max(1, total_updates)
    warm = int(warmup_frac * total)
    if warm > 0 and updates < warm:
        return peak * (updates + 1) / warm
    progress = min(1.0, max(0.0, (updates - warm) / max(1, total - warm)))
    ratio = eta_min / peak
    return peak * (ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def optim_state_cpu(opt: torch.optim.Optimizer) -> dict:
    """Deep CPU copy of an optimizer state_dict: the async saver writes it while training goes on."""
    sd = opt.state_dict()
    return {"state": {k: {n: (t.detach().cpu().clone() if torch.is_tensor(t) else t) for n, t in v.items()}
                      for k, v in sd["state"].items()},
            "param_groups": copy.deepcopy(sd["param_groups"])}


def to_input(stacks_u8: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(stacks_u8).to(device, non_blocking=True).float().div_(255.0)


@torch.no_grad()
def mirror_permutation(action_meanings) -> np.ndarray:
    """perm[a] = the action that mirrors a under a horizontal flip (LEFT <-> RIGHT; others map to themselves)."""
    idx = {m: i for i, m in enumerate(action_meanings)}
    swap = lambda m: m.replace("LEFT", "\0").replace("RIGHT", "LEFT").replace("\0", "RIGHT")
    return np.array([idx.get(swap(m), i) for i, m in enumerate(action_meanings)])


def run_search(model, mcts: GumbelMCTS, obs: torch.Tensor, add_noise: bool, chunk: int = 2048, flip_perm=None,
               graphed: GraphedSearch = None):
    """flip_perm (TODO B26 step 0, eval only): also run the root inference on the horizontally flipped frames and
    average its value and policy logits (mapped back through flip_perm) into the root's before searching."""
    vals, pols, acts = [], [], []
    for i in range(0, obs.shape[0], chunk):
        x = obs[i:i + chunk]
        s, v, p = model.initial_inference(x)
        v = model.support.vector_to_scalar(v)
        if flip_perm is not None:
            _, vf, pf = model.initial_inference(torch.flip(x, dims=[-1]))
            v = 0.5 * (v + model.support.vector_to_scalar(vf))
            p = 0.5 * (p + pf[:, torch.as_tensor(flip_perm, device=p.device)])
        rv, rp, ra = (graphed(s, v, p, add_noise) if graphed is not None
                      else mcts.search(model, s, v, p, add_noise=add_noise))
        vals.append(rv); pols.append(rp); acts.append(ra)
    return np.concatenate(vals), np.concatenate(pols), np.concatenate(acts)


PARKED_STEPS = 1000  # TODO B70: a reward-free streak this long counts as a parked ball (healthy Breakout lives score every ~50-300 steps)


def play_window(n_actions: int) -> dict:
    """TODO B70: empty play-diagnostics accumulator for one log window (see play_add / play_summary)."""
    return {"steps": 0, "acts": np.zeros(n_actions, np.int64), "max_streak": 0, "parked": 0, "pi_h": 0.0, "lives": []}


def play_add(w: dict, actions, since_reward, search_pi, ended_life_lengths) -> None:
    """One env step of all envs: executed actions, per-env reward-free streaks after the step, the acting search policy (E, A)
    and the lengths of the lives that ended on this step."""
    a = np.asarray(actions, np.int64)
    w["steps"] += len(a)
    w["acts"] += np.bincount(a, minlength=len(w["acts"]))[:len(w["acts"])]
    w["max_streak"] = max(w["max_streak"], int(np.max(since_reward)))
    w["parked"] += int(np.sum(np.asarray(since_reward) >= PARKED_STEPS))
    p = np.asarray(search_pi, np.float64)
    w["pi_h"] += float(-(p * np.log(p + 1e-8)).sum(-1).sum())
    w["lives"].extend(int(x) for x in ended_life_lengths)


def play_summary(w: dict) -> dict:
    n = max(w["steps"], 1)
    share = w["acts"] / n
    out = {"life len": f"{np.mean(w['lives']):.0f} (n {len(w['lives'])})" if w["lives"] else "none ended",
           "max rf streak": w["max_streak"], f"parked(>={PARKED_STEPS})": f"{100 * w['parked'] / n:.1f}%",
           "act share": "/".join(f"{x:.2f}" for x in share), "act-pi H": f"{w['pi_h'] / n:.3f}"}
    out["metrics"] = {"play/max_reward_free_streak": w["max_streak"], "play/parked_share": w["parked"] / n,
                      "play/search_pi_entropy": w["pi_h"] / n, **{f"play/act_share_{i}": float(x) for i, x in enumerate(share)},
                      **({"play/life_length": float(np.mean(w["lives"]))} if w["lives"] else {})}
    return out


@torch.no_grad()
def evaluate(model, mcts, env_id: str, device, episodes: int, frame_size: int, seed: int,
             max_steps: int = 27_000, sticky: float = 0.0, flip_avg: bool = False, graph: bool = False,
             resume_path=None, fire_reset: bool = False) -> np.ndarray:
    """One full-game episode (no episodic life, raw reward) per parallel env, search without noise.
    sticky > 0: ALE sticky actions with that probability (B2's robustness protocol; off for Atari-100k).
    fire_reset: press FIRE after each reset and lost life (the standard FireResetEnv). Breakout freezes until FIRE is pressed, so
    a policy that never presses it plays to max_steps (44 of 60 capped episodes in the seed-0 paper-protocol eval, 2026-10-11).
    Off = the protocol that produced the 364.25 score; on = the same protocol with the freeze removed.
    flip_avg: average the root with its mirror image (run_search's flip_perm, TODO B26 step 0).
    graph: replay the search from a CUDA graph (GraphedSearch, TODO B23; same results, far fewer launches).
    resume_path: an .npz that holds every action taken so far (the policy is stateless per step, so a game is fixed by its seed
    and its actions). It is rewritten every 500 steps; a restart replays those actions through the envs (no model calls, minutes
    not hours) and continues from there. The caller deletes it once the result file exists."""
    model.eval()
    graphed = GraphedSearch(mcts, model) if graph and device.type == "cuda" else None
    envs = make_vector_atari_envs(env_id, num_envs=episodes, seed=seed + 10_000, clip_reward=False,
                                  episodic_life=False, frame_size=frame_size, grayscale=False,
                                  fire_reset=fire_reset, max_episode_steps=max_steps, repeat_action_probability=sticky)
    obs, _ = envs.reset(seed=seed + 10_000)
    perm = None
    if flip_avg:
        from atari_qwen.envs.atari_wrappers import gym  # the module that registers the ALE envs
        probe = gym.make(env_id)
        perm = mirror_permutation(probe.unwrapped.get_action_meanings())
        probe.close()
    scores = np.zeros(episodes)
    finished = np.zeros(episodes, dtype=bool)
    actions = []                                       # one row of per-env actions per step, for the resume file
    start = 0
    meta = np.array([episodes, seed, max_steps, int(sticky * 1000), int(fire_reset)])
    if resume_path is not None and Path(resume_path).exists():
        try:
            z = np.load(resume_path)
            if np.array_equal(z["meta"], meta):
                past = z["actions"]
                t_rp = time.time()
                for step in range(len(past)):          # replay: the same env code path as the live loop, no search
                    obs, r, term, trunc, _ = envs.step(past[step].astype(np.int64))
                    scores += np.where(finished, 0.0, r)
                    finished |= np.logical_or(term, trunc)
                actions = [row for row in past]
                start = len(past)
                print(f"  eval resumed from {resume_path}: replayed {start} steps in {(time.time() - t_rp) / 60:.1f} min, "
                      f"games over {int(finished.sum())}/{episodes}", flush=True)
            else:
                print("  eval resume file does not match this protocol; starting from step 0", flush=True)
        except Exception as ex:
            print(f"  eval resume file unusable ({ex}); starting from step 0", flush=True)
            actions, start = [], 0
            scores = np.zeros(episodes); finished = np.zeros(episodes, dtype=bool)
            obs, _ = envs.reset(seed=seed + 10_000)
    t_eval = time.time()
    for step in range(start, max_steps):
        if finished.all():
            break
        # TODO B-eval-speed (tried, ruled out 2026-10-08): searching only the still-running games (instead of all `episodes`
        # every step) to skip wasted work on finished games was NOT bit-identical and was not even faster: a smaller batch
        # picks a different cuDNN conv algorithm, the tiny float rounding difference flips an argmax at some close decision,
        # and the game's subsequent trajectory diverges from there (measured: one episode's score 107 vs 377, mean -18%,
        # wall time x1.1 slower, not faster, because of extra CUDA-graph captures per batch size). Keep the full batch.
        _, _, a = run_search(model, mcts, to_input(obs, device), add_noise=False, flip_perm=perm, graphed=graphed)
        actions.append(np.asarray(a, dtype=np.int16))
        obs, r, term, trunc, _ = envs.step(a)
        scores += np.where(finished, 0.0, r)
        finished |= np.logical_or(term, trunc)
        if finished.all():
            break
        if step % 500 == 499:   # progress for ETAs: the loop ends when every game is over or at max_steps
            el = time.time() - t_eval
            rate = (step + 1 - start) / el
            print(f"  eval progress: step {step + 1}/{max_steps}, games over {int(finished.sum())}/{episodes}, "
                  f"{rate:.2f} steps/s, {el / 60:.1f} min elapsed, worst-case end in "
                  f"{(max_steps - step - 1) / rate / 60:.0f} min", flush=True)
            if resume_path is not None:
                tmp = Path(str(resume_path) + ".tmp.npz")
                np.savez(tmp, actions=np.stack(actions), meta=meta)
                os.replace(tmp, resume_path)
    envs.close()
    if graphed is not None and graphed.fallbacks:
        print(f"  graphed search: {graphed.fallbacks} of {graphed.calls} searches exceeded the depth bound (eager fallback)", flush=True)
    return scores


# ------------------------------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_label or f"ezv2_{args.trunk}_{args.env_id}_s{args.seed}_{int(time.time())}"
    run_dir = Path(args.log_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb")) if SummaryWriter else None

    mlrun = None
    if args.mlflow and HAS_MLFLOW:
        try:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            try:
                from src.config.paths import mlruns_dir
                mlflow.set_tracking_uri(mlruns_dir().resolve().as_uri())
            except Exception:
                pass
            mlflow.set_experiment(args.experiment)
            mlrun = mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: v for k, v in vars(args).items()})
        except Exception as ex:
            print(f"! MLflow disabled ({ex})", flush=True)

    def log(metrics: Dict[str, float], step: int):
        for k, v in metrics.items():
            if writer:
                writer.add_scalar(k, v, step)
        if mlrun:
            try:
                mlflow.log_metrics({k.replace("/", "_"): float(v) for k, v in metrics.items()}, step=step)
            except Exception:
                pass

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        if total_gb < 12.0:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            print(f"--> [GPU Profile] {props.name} ({total_gb:.1f} GB VRAM) - Enforcing expandable_segments:True (10GB profile)", flush=True)
        else:
            print(f"--> [GPU Profile] {props.name} ({total_gb:.1f} GB VRAM) - Standard high-headroom profile", flush=True)

    E, K, TD, H = args.num_envs, args.unroll_steps, args.td_steps, args.lstm_horizon
    gamma = args.discount ** 4                       # per agent step (frame skip 4), as EZ-V2
    envs = make_vector_atari_envs(args.env_id, num_envs=E, seed=args.seed, clip_reward=True, episodic_life=True,
                                  frame_size=args.frame_size, grayscale=False, fire_reset=False,
                                  max_episode_steps=args.train_episode_steps, same_step_autoreset=True)
    A = envs.single_action_space.n
    support = DiscreteSupport()
    model = EZV2Model(A, obs_channels=3 * 4, support=support, trunk=args.trunk, norm=args.norm,
                      state_hw=int(np.ceil(args.frame_size / 16)),
                      mixer_out_init_std=args.mixer_out_init_std, mixer_bg_init=args.mixer_bg_init,
                      simnorm=args.simnorm, mixer_pos=args.mixer_pos, mixer_size=args.mixer_size,
                      mixer_qk_norm=args.mixer_qk_norm, mixer_where=args.mixer_where,
                      mixer_no_pos=args.mixer_no_pos).to(device)
    if args.mixer_qk_norm:
        print("--> mixer: QK-norm on (per-head RMSNorm on queries and keys, TODO B67)", flush=True)
    if args.mixer_where == "repr":
        assert args.trunk != "resnet", "--mixer-where repr needs --trunk gtrxl or qwen"
        print("--> mixer: representation only, dynamics = EZ-V2 conv + ResNet (TODO B68 D1)", flush=True)
    elif args.mixer_where == "pred":
        print("--> mixer: CARLA trunk layout as the prediction (policy token -> value / policy heads); representation and dynamics "
              "= EZ-V2 conv + ResNet (TODO B69)", flush=True)
    target = copy.deepcopy(model).eval()
    # bf16 autocast only where the GPU has native bf16 (Ampere+). is_bf16_supported() also returns True for
    # emulated bf16, which ran ~4.5x slower than fp16 on a 2080 Ti (S-099); there is no GradScaler for fp16,
    # so pre-Ampere GPUs run in fp32 instead.
    bf16_native = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    amp = torch.bfloat16 if bf16_native else torch.float32
    for p in target.parameters():
        p.requires_grad_(False)
    if getattr(args, "channels_last", True) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        target = target.to(memory_format=torch.channels_last)
    mix_params = [q for n, q in model.named_parameters() if "mixer" in n]
    opt_mix = None
    if args.mixer_optimizer in ("adamw", "adamw_carla"):
        # TODO B37: scale-invariant optimiser for the GTrXL mixer only (SGD moved its q/k/v by lr x a gradient orders of magnitude below the rest of
        # the net: |Wq| stayed at init); conv trunk and heads keep EZ-V2's SGD recipe.
        assert mix_params, "--mixer-optimizer adamw needs --trunk gtrxl or qwen"
        rest_params = [q for n, q in model.named_parameters() if "mixer" not in n]
        opt = torch.optim.SGD(rest_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.mixer_optimizer == "adamw":
            mwd = 0.05 if args.mixer_weight_decay is None else args.mixer_weight_decay
            opt_mix = torch.optim.AdamW(mix_params, lr=args.mixer_lr, betas=(0.9, 0.95), weight_decay=mwd)
        else:
            # TODO B66: the recipe the same Qwen network trains with on CARLA (AdamW, default betas, CARLA's decay groups,
            # warm-up + cosine via carla_lr). Under EZ-V2's SGD (lr 0.2, momentum 0.9) the Qwen mixer's output reached 2x the
            # conv state within 8k updates and the policy prior stayed ~0.3 nats flatter than the ResNet runs' (B66 diagnosis).
            mwd = 1e-4 if args.mixer_weight_decay is None else args.mixer_weight_decay
            opt_mix = torch.optim.AdamW(mixer_param_groups(model, mwd), lr=args.mixer_lr)
    elif args.mixer_weight_decay is None:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:  # TODO B36: the GTrXL mixer's own weight decay (0 = exempt); everything else keeps --weight-decay
        mix = [q for n, q in model.named_parameters() if "mixer" in n]
        rest = [q for n, q in model.named_parameters() if "mixer" not in n]
        opt = torch.optim.SGD([{"params": rest, "weight_decay": args.weight_decay},
                               {"params": mix, "weight_decay": args.mixer_weight_decay}], lr=args.lr, momentum=0.9)
    # TODO B37 step 1: per-category audit of the mixer (relative change, absolute change and gradient norm; `mixer_audit.py`: the zero-initialised
    # tensors have no meaningful relative change, read their absolute change)
    mix_w0 = mixer_audit.snapshot(model) if args.audit_every else {}
    if (args.compile_model or args.compile_search) and device.type == "cuda":
        # each batch shape, grad mode and train / eval mode is a compiled variant of its own (the cache's 5 buckets, acting, training,
        # evaluation; repr_mixer and dyn_mixer share TokenMixer.forward). Past the default limit of 8 per function torch.compile runs
        # the rest eagerly without an error (seen in atari_qwen/scripts/b68_bottleneck.py, 2026-10-26).
        import torch._dynamo.config as dynamo_config
        setattr(dynamo_config, "recompile_limit" if hasattr(dynamo_config, "recompile_limit") else "cache_size_limit", 64)
    if args.compile_model and device.type == "cuda":
        # TODO B68 E8: fuse the training step's elementwise work (the mixers' fp32 RMSNorm kernels, gates, casts), forward and backward.
        # Compiled in place (nn.Module.compile), so the state-dict keys, the target sync and the checkpoints are unchanged; the conv /
        # BatchNorm / LSTM parts stay eager, and the acting search uses the same compiled mixers. The old flag wrapped the whole model,
        # which renamed every key (_orig_mod.) and broke the target sync. Acceptance: atari_qwen/scripts/b68_train_compile_check.py.
        assert args.trunk != "resnet", "--compile-model compiles the Qwen / GTrXL mixers"
        for mixer in (model.repr_mixer, model.dyn_mixer, model.pred_mixer):
            if mixer is not None:
                mixer.compile(dynamic=False)
        print("--> training: the online model's mixers compiled (TODO B68 E8)", flush=True)
    mcts = GumbelMCTS(A, support, num_simulations=args.num_simulations, discount=gamma, lstm_horizon=H)
    # TODO B23 step 1b: replay the acting search (B = num_envs) and the reanalyze search (B = batch x (K+1), on the target
    # network, bf16 where native) from CUDA graphs; bit-identical to eager (see GraphedSearch), far fewer kernel launches
    mk = lambda: GumbelMCTS(A, support, num_simulations=args.num_simulations, discount=gamma, lstm_horizon=H)
    graph_on = args.graph_search and device.type == "cuda"
    act_graph = GraphedSearch(mk(), model) if graph_on else None
    if args.compile_search and device.type == "cuda":
        # TODO B68 E1: fuse the elementwise work of the target network's recurrent_inference (RMSNorm's fp32 kernels, gates, casts:
        # ~67% of a 30m dynamics-mixer call). Same maths: on 1,536 real roots the compiled search is as close to an fp32 search as the
        # eager one (policy TV 0.0229 vs 0.0233, value 0.0048 vs 0.0052); 1.87x on the 30m graphed search. Only recurrent_inference:
        # inside the search it sees fixed shapes (one compile per batch / cache bucket); the root and bootstrap passes stay eager.
        # load_state_dict copies the weights in place, so the compiled code follows every target refresh without recompiling.
        target.recurrent_inference = torch.compile(target.recurrent_inference, dynamic=False)
        print("--> reanalysis search: target recurrent_inference compiled (TODO B68 E1)", flush=True)
    re_graph = GraphedSearch(mk(), target, amp_dtype=amp if bf16_native else None) if graph_on else None
    print(f"--> {run_name}: trunk={args.trunk} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"envs={E} sims={args.num_simulations} batch={args.batch_size} gamma/step={gamma:.4f}", flush=True)

    steps_per_env = args.total_steps // E + K + TD + 16
    replay = EZReplay(steps_per_env, E, (3, args.frame_size, args.frame_size), A, alpha=args.priority_alpha,
                      device=device)
    rcache = ReanalysisCache(steps_per_env * E, A, device) if args.reanalyze_cache else None
    if rcache is not None:
        print(f"--> reanalysis cache: on (TODO B68), one search per replay position per target window of "
              f"{args.target_update_interval} updates, noise hashed per (position, window)", flush=True)
    if args.reanalyze_merge:
        # TODO B68 M1: drawing batches ahead is exact only if a batch does not depend on the updates before it: uniform replay
        # (prioritised sampling reads the priorities each update writes) and noise that is a function of (position, window)
        assert rcache is not None and args.priority_alpha == 0, "--reanalyze-merge needs --reanalyze-cache and --priority-alpha 0"
        print("--> reanalysis cache: misses of the updates between two env steps (after collection: of a target window) searched in "
              "one call, from batches drawn in the same order (TODO B68 M1)", flush=True)
    # Schedules follow the FULL run (--schedule-steps, EZ-V2: 100k); a shorter --total-steps is an exact
    # prefix of it (the CARLA --stop_epoch rule). Scaling the 1% warm-up to a 10k screen gave lr 0.2
    # after 120 updates instead of 1,000 and let the value head inflate itself (S-058, S058a).
    full_online = int(args.schedule_steps * args.replay_ratio)
    warm = max(1, int(full_online * args.lr_warmup))
    online_updates = int(args.total_steps * args.replay_ratio)
    total_updates = online_updates + (int(full_online * args.offline_frac) if args.total_steps >= args.schedule_steps else 0)

    if args.mixer_optimizer == "adamw_carla":
        print(f"--> mixer: AdamW (CARLA recipe), peak lr {args.mixer_lr:g}, weight decay {mwd:g} on matrices, warm-up "
              f"{int(args.mixer_warmup_frac * total_updates)} updates then cosine to 1e-6 over {total_updates} updates", flush=True)

    env_steps, updates, best = 0, 0, -float("inf")
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); target.load_state_dict(ck["target"]); opt.load_state_dict(ck["opt"])
        if opt_mix is not None:
            if "opt_mix" in ck:
                opt_mix.load_state_dict(ck["opt_mix"])
            else:
                print("[Warning] the checkpoint has no mixer optimizer state: AdamW restarts its moments", flush=True)
        env_steps, updates, best = ck["env_steps"], ck["updates"], ck.get("best", best)
        rp = Path(args.resume_from).with_name("replay_latest.npz")
        if rp.exists():
            replay.load(rp)
        print(f"--> resumed at env step {env_steps:,}, update {updates:,} (replay {replay.transitions:,})", flush=True)

    saver = AsyncCheckpointSaver() if getattr(args, "async_checkpoint", True) else None

    def save(tag: str, extra=None):
        if opt_mix is not None:  # TODO B66: the mixer's AdamW state goes with the run (it was not saved before, so a resume reset it)
            extra = {**(extra or {}), "opt_mix": optim_state_cpu(opt_mix)}
        if saver is not None:
            saver.save_async(tag, model, target, opt, env_steps, updates, best, vars(args), ckpt_dir,
                             replay=replay if tag == "latest" else None, extra=extra)
        else:
            st = {"model": model.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(),
                  "env_steps": env_steps, "updates": updates, "best": best, "args": vars(args)}
            if extra:
                st.update(extra)
            tmp = ckpt_dir / f"checkpoint_{tag}.pt.tmp"
            torch.save(st, tmp); os.replace(tmp, ckpt_dir / f"checkpoint_{tag}.pt")
            if tag == "latest":
                replay.save(ckpt_dir / "replay_latest.npz")

    obs, _ = envs.reset(seed=args.seed)
    print(f"autocast dtype: {amp} (native bf16: {bf16_native}); graph search: {graph_on}", flush=True)
    ep_ret = np.zeros(E); recent_returns = []
    # TODO B62 fix 1: per env, agent steps since the last non-zero reward (not reset by a lost life or a new game, so an
    # always-FIRE policy that loses every life without scoring also trips it) and the random steps left in the current burst
    since_r = np.zeros(E, np.int64); burst_left = np.zeros(E, np.int64); explore_steps = 0
    explore_rng = np.random.default_rng(args.seed + 7919)
    # TODO B70 (2026-10-28): play diagnostics per log window, for why a run's return lags at the same losses (B69 pred: life return 13
    # vs 19-24 at 80-100k, 44 of 60 paper-eval games parked at the step cap): life length, longest reward-free streak, share of env
    # steps inside a streak of >= 1,000 steps ("parked"), executed-action shares and the entropy of the acting search policy
    play = play_window(A)
    life_len = np.zeros(E, np.int64)
    t0 = time.time(); last_log = env_steps; next_eval = (env_steps // args.eval_interval + 1) * args.eval_interval
    tm = {"collect": 0.0, "targets": 0.0, "update": 0.0}
    # TODO B64 (speed of the 100k Qwen run): sub-stage split of "targets" and "update". Diagnostics only: each mark syncs the
    # GPU, so the run is slower while it is on and the c/t/u shares above are not comparable to a run without it.
    prof_t = args.profile_targets and device.type == "cuda"
    tm_sub = {}
    prof = None

    def mark(key, t_):
        if not prof_t:
            return t_
        torch.cuda.synchronize(); now = time.perf_counter()
        tm_sub[key] = tm_sub.get(key, 0.0) + now - t_
        return now
    upd_credit = 0.0
    horizon = K + TD
    stepped = False
    gn_hot = 0
    nan_skips = 0                                                   # updates skipped by --nan-guard (non-finite gradient norm)
    pending = []                                                    # TODO B68 M1: batches drawn ahead; the cache holds their targets
    chl = getattr(args, "channels_last", True) and device.type == "cuda"

    def cache_keys(t_, e_, upd):
        """TODO B68: flat replay positions (row * E + env) of an update's search roots t..t+K and bootstrap states t+TD..t+TD+K, and
        whether a loss uses each (the same masks as the target code below)."""
        pos_ = t_[:, None] + np.arange(horizon + 1)
        dn_ = replay.done[pos_[:, :-1], e_[:, None]]
        ex_ = np.concatenate([np.ones((len(t_), 1), bool), np.cumprod(~dn_, axis=1).astype(bool)], axis=1)
        rec_ = (((t_ * E + e_) > (replay.transitions - args.mixed_value_threshold)) if upd >= args.mixed_value_start
                else np.ones(len(t_), bool))
        flat = pos_ * E + e_[:, None]
        return (flat[:, :K + 1].reshape(-1), ex_[:, :K + 1].reshape(-1), flat[:, TD:TD + K + 1].reshape(-1),
                (rec_[:, None] & ex_[:, TD:TD + K + 1]).reshape(-1))

    def stacks_fn(keys):
        x = replay.stacks(keys // E, keys % E)
        return x.contiguous(memory_format=torch.channels_last) if chl else x

    def ii_fn(x):
        s_, v_, p_ = target.initial_inference(x)
        return s_, support.vector_to_scalar(v_), p_

    def search_fn(s_, v_, p_, g_):
        out = re_graph(s_, v_, p_, g=g_) if re_graph is not None else mcts.search(target, s_, v_, p_, g=g_)
        return out[0], out[1]

    while updates < total_updates:
        # ---------------- collect (stops at total_steps; the rest is offline training) ----------
        if env_steps < args.total_steps:
            tc = time.perf_counter()
            model.eval()
            rv, rp_, ra = run_search(model, mcts, to_input(obs, device), add_noise=True, graphed=act_graph)
            if args.stagnation_explore > 0:
                # TODO B62 fix 1 (training only; evaluation keeps the paper protocol): after --stagnation-explore reward-free steps
                # an env takes --stagnation-burst uniformly random actions. The search targets (rp_, rv) are still stored, only the
                # executed action changes. Breaks the 2026-10-23 collapses (never FIRE: 0 presses in 8,000 steps; always FIRE).
                burst_left[(since_r >= args.stagnation_explore) & (burst_left == 0)] = args.stagnation_burst
                rnd = burst_left > 0
                if rnd.any():
                    ra = np.array(ra, copy=True)
                    ra[rnd] = explore_rng.integers(0, A, size=int(rnd.sum()))
                    explore_steps += int(rnd.sum())
                    burst_left[rnd] -= 1
                    since_r[rnd & (burst_left == 0)] = 0     # a finished burst restarts the count
            nobs, r, term, trunc, _ = envs.step(ra)
            since_r = np.where(r != 0, 0, since_r + 1)
            d = np.logical_or(term, trunc)
            replay.add(obs[:, -3:], ra, r, d, rp_, rv)
            ep_ret += r
            life_len += 1
            play_add(play, ra, since_r, rp_, life_len[d])
            life_len[d] = 0
            for i in np.where(d)[0]:
                recent_returns.append(ep_ret[i]); ep_ret[i] = 0.0
            obs = nobs
            env_steps += E
            tm["collect"] += time.perf_counter() - tc
            if replay.transitions < args.start_transitions or replay.size <= horizon + 1:
                continue  # no update credit during warm-up (a 1,000-update burst at the start otherwise)
            upd_credit += E * args.replay_ratio
            n_upd = int(upd_credit); upd_credit -= n_upd
        else:
            n_upd = 1
        offline = env_steps >= args.total_steps                     # no more data: batches can be drawn a whole window ahead (B68 M1)

        for i_upd in range(n_upd):
            if updates >= total_updates:
                break
            lr = args.lr * min(1.0, (updates + 1) / warm) * (0.1 ** (max(0, updates - warm) // args.lr_decay_steps))
            for g_ in opt.param_groups:
                g_["lr"] = lr
            if opt_mix is not None:
                m_lr = (carla_lr(updates, total_updates, args.mixer_lr, args.mixer_warmup_frac) if args.mixer_optimizer == "adamw_carla"
                        else args.mixer_lr * min(1.0, (updates + 1) / max(1, args.mixer_warmup)))
                for g_ in opt_mix.param_groups:
                    g_["lr"] = m_lr

            if args.profile_at and prof is None and updates == args.profile_at:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()

            # ---------------- targets (target network, no grad) ---------------------------------
            tt = time.perf_counter()
            B = args.batch_size
            if pending:
                t, e, isw = pending.pop(0)                          # TODO B68 M1: drawn ahead; its targets are already in the cache
            else:
                t, e, isw = replay.sample(B, horizon, beta=args.priority_beta)
                if args.reanalyze_merge:
                    # TODO B68 M1: draw now the batches of the next updates that see this same buffer, from the same RNG stream in the same
                    # order (each update gets the batch it would have drawn), and search all their misses in one call. Stops at the next
                    # env step and at the target refresh; once collection has ended, at the end of the window. (--profile-targets counts
                    # this call in t_sample.)
                    W_ = args.target_update_interval
                    L = min(W_ - updates % W_, total_updates - updates, W_ if offline else n_upd - i_upd)
                    pending = [replay.sample(B, horizon, beta=args.priority_beta) for _ in range(L - 1)]
                    ks_ = [cache_keys(tb, eb, updates + j) for j, (tb, eb, _) in enumerate([(t, e, isw)] + pending)]
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp, enabled=bf16_native):
                        rcache.fill(*(np.concatenate([k_[i] for k_ in ks_]) for i in range(4)), updates // W_,
                                    stacks_fn, ii_fn, search_fn, A)
            ks = np.arange(horizon + 1)
            pos = t[:, None] + ks                                        # (B, K+TD+1)
            dn = replay.done[pos[:, :-1], e[:, None]]                     # done after step t+k
            exists = np.concatenate([np.ones((B, 1), bool), np.cumprod(~dn, axis=1).astype(bool)], axis=1)
            rew = replay.rew[pos[:, :-1], e[:, None]] * exists[:, :-1]  # r_{t+k}, 0 past episode end
            ee = np.repeat(e, K + 1)
            root_stacks = replay.stacks(pos[:, :K + 1].reshape(-1), ee)          # states t..t+K (also the training inputs)
            if chl:
                root_stacks = root_stacks.contiguous(memory_format=torch.channels_last)
            x_root = root_stacks
            if rcache is None:                                                   # the B68 cache stacks only its misses
                boot_stacks = replay.stacks(pos[:, TD:TD + K + 1].reshape(-1), ee)   # states t+TD..t+K+TD
                if chl:
                    boot_stacks = boot_stacks.contiguous(memory_format=torch.channels_last)
            recent = ((t * E + e) > (replay.transitions - args.mixed_value_threshold)) if updates >= args.mixed_value_start else np.ones(B, bool)
            ts = mark("t_sample", tt)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp, enabled=bf16_native):
                if rcache is not None:
                    # TODO B68: search each replay position once per target window (noise hashed per position and window); repeats and the
                    # bootstrap values come from the cache. Same targets as a fresh search in this window would give.
                    srch_v, srch_pi, vb = rcache.reanalyse(*cache_keys(t, e, updates), updates // args.target_update_interval,
                                                           stacks_fn, ii_fn, search_fn, A)
                    v_boot = vb.reshape(B, K + 1).astype(np.float32)
                    ts = mark("t_search", ts)
                else:
                    v_boot = np.zeros((B, K + 1), np.float32)
                    recent_idx = np.nonzero(recent)[0]
                    if len(recent_idx) == B:
                        _, vb, _ = target.initial_inference(boot_stacks)
                        v_boot = support.vector_to_scalar(vb).float().cpu().numpy().reshape(B, K + 1)
                    elif len(recent_idx) > 0:
                        sel = torch.as_tensor(recent_idx, device=device)
                        xb = boot_stacks.view(B, K + 1, *boot_stacks.shape[1:])[sel].reshape(-1, *boot_stacks.shape[1:])
                        _, vb, _ = target.initial_inference(xb)
                        v_boot[recent_idx] = support.vector_to_scalar(vb).float().cpu().numpy().reshape(len(recent_idx), K + 1)
                    ts = mark("t_boot", ts)
                    x_root = root_stacks
                    s_re, v_re, p_re = target.initial_inference(x_root)
                    v_root = support.vector_to_scalar(v_re)
                    ts = mark("t_root", ts)
                    srch_v, srch_pi, _ = (re_graph(s_re, v_root, p_re, True) if re_graph is not None
                                          else mcts.search(target, s_re, v_root, p_re, add_noise=True))
                    ts = mark("t_search", ts)
            srch_v = srch_v.reshape(B, K + 1); srch_pi = srch_pi.reshape(B, K + 1, A)
            disc = gamma ** np.arange(TD)
            val_t = np.zeros((B, K + 1), np.float32)
            for k in range(K + 1):
                val_t[:, k] = (rew[:, k:k + TD] * disc).sum(1) + (gamma ** TD) * v_boot[:, k] * exists[:, k + TD]
            ex_k = exists[:, :K + 1]
            val_t *= ex_k
            srch_v = srch_v * ex_k
            val_t = np.where(recent[:, None], val_t, srch_v)
            if args.policy_target_smoothing > 0:
                # TODO B62 fix 2: entropy floor on the search policy targets (mix with uniform), so the prior cannot become one-hot
                # when the value is flat (2026-10-23 Qwen seed 1: policy loss 0.2, V0 0.04, never pressed FIRE)
                srch_pi = (1.0 - args.policy_target_smoothing) * srch_pi + args.policy_target_smoothing / A
            pol_t = srch_pi * ex_k[..., None]
            # value prefix: running reward sum, reset every H steps
            prefix = np.zeros((B, K), np.float32); run = np.zeros(B, np.float32)
            for k in range(K):
                if k % H == 0:
                    run = np.zeros(B, np.float32)
                run = run + rew[:, k]
                prefix[:, k] = run
            mask = exists[:, :K].astype(np.float32)                      # action t+k was taken
            cons_mask = exists[:, 1:K + 1].astype(np.float32)             # obs t+k+1 is real
            mark("t_rest", ts)
            tm["targets"] += time.perf_counter() - tt

            # ---------------- update ------------------------------------------------------------
            tu = time.perf_counter()
            model.train()
            obs_all = x_root.float().view(B, K + 1, *x_root.shape[1:])
            obs0 = augment(obs_all[:, 0])
            tgt_obs = augment(obs_all[:, 1:].reshape(B * K, *x_root.shape[1:]))
            T_ = lambda a: torch.as_tensor(a, device=device)
            val_t_, pol_t_, prefix_, mask_, cmask_, isw_ = map(T_, (val_t, pol_t, prefix, mask, cons_mask, isw))
            acts = T_(replay.act[pos[:, :K], e[:, None]])
            with torch.autocast(device_type=device.type, dtype=amp, enabled=bf16_native):
                s, v_log, p_log = model.initial_inference(obs0)
                with torch.no_grad():
                    # one call per unroll step, as EZ-V2: each train-mode pass refreshes the BatchNorm
                    # running stats, which the eval-mode target network and search depend on
                    tgt_o = tgt_obs.view(B, K, *tgt_obs.shape[1:])
                    tgt_proj = torch.stack([model.project(model.representation(tgt_o[:, k]), with_grad=False).float()
                                            for k in range(K)], dim=1)
                v0 = support.vector_to_scalar(v_log.detach())
                ce = lambda logits, target_: -(F.log_softmax(logits.float(), -1) * target_).sum(-1)
                value_loss = 0.5 * ce(v_log, support.scalar_to_vector(val_t_[:, 0]))
                policy_loss = ce(p_log, pol_t_[:, 0])
                prefix_loss = torch.zeros(B, device=device); cons_loss = torch.zeros(B, device=device)
                hid = model.init_hidden(B, device)
                for k in range(K):
                    s, vp_log, v_log, p_log, hid = model.recurrent_inference(s, acts[:, k], hid)
                    m = mask_[:, k]
                    proj = model.project(s, with_grad=True).float()
                    cons_loss = cons_loss - (F.normalize(proj, dim=-1, eps=1e-5) *
                                             F.normalize(tgt_proj[:, k], dim=-1, eps=1e-5)).sum(-1) * cmask_[:, k]
                    prefix_loss = prefix_loss + ce(vp_log, support.scalar_to_vector(prefix_[:, k])) * m
                    value_loss = value_loss + 0.5 * ce(v_log, support.scalar_to_vector(val_t_[:, k + 1])) * m
                    policy_loss = policy_loss + ce(p_log, pol_t_[:, k + 1]) * m
                    if s.requires_grad:
                        s.register_hook(lambda g: g * 0.5)
                    if (k + 1) % H == 0:
                        hid = model.init_hidden(B, device)
                loss = (args.reward_coeff * prefix_loss + args.value_coeff * value_loss
                        + args.policy_coeff * policy_loss + args.consistency_coeff * cons_loss)
                wloss = (isw_ * loss).mean() / K                           # EZ-V2 gradient_scale = 1/K
            tu2 = mark("u_fwd", tu)
            opt.zero_grad(set_to_none=True)
            wloss.backward()
            tu2 = mark("u_bwd", tu2)
            mgn = None
            if opt_mix is not None:
                mgn = nn.utils.clip_grad_norm_(mix_params, args.mixer_grad_clip)
                gn = nn.utils.clip_grad_norm_([q for n, q in model.named_parameters() if "mixer" not in n], args.max_grad_norm)
            else:
                gn = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            bad = args.nan_guard and not (bool(torch.isfinite(gn)) and (mgn is None or bool(torch.isfinite(mgn))))
            if args.audit_every and (updates + 1) % args.audit_every == 0 and mix_w0:
                _, audit_grad, _ = mixer_audit.audit(model, mix_w0)  # gradient norms of this update (before the step)
            if bad:                                  # B37 retry guard: a non-finite gradient would poison AdamW's moments for good
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                if opt_mix is not None:
                    opt_mix.zero_grad(set_to_none=True)
                if nan_skips <= 5 or nan_skips % 100 == 0:
                    print(f"[nan-guard] upd {updates}: non-finite grad norm (gn {float(gn)}, mixer {None if mgn is None else float(mgn)}), update skipped ({nan_skips} so far)", flush=True)
            else:
                opt.step()
                if opt_mix is not None:
                    opt_mix.step()
            updates += 1
            stepped = True
            if args.audit_every and updates % args.audit_every == 0 and mix_w0:
                rel, _, absd = mixer_audit.audit(model, mix_w0)
                print(f"[B37 audit] upd {updates}: |dW|/max(|W0|,1e-3) " + " ".join(f"{c} {v:.4f}" for c, v in sorted(rel.items())) +
                      " | |dW| " + " ".join(f"{c} {v:.3f}" for c, v in sorted(absd.items())) +
                      " | grad " + " ".join(f"{c} {v:.2e}" for c, v in sorted(audit_grad.items())), flush=True)
                log({**{f"audit/rel_{c}": v for c, v in rel.items()}, **{f"audit/abs_{c}": v for c, v in absd.items()},
                     **{f"audit/grad_{c}": v for c, v in audit_grad.items()}}, env_steps)
            if args.priority_alpha > 0:
                replay.update_priorities(t, e, np.abs(v0.float().cpu().numpy() - val_t[:, 0]) + 1e-6)
            if updates % args.target_update_interval == 0:
                target.load_state_dict(model.state_dict())
            if prof_t:
                mark("u_rest", tu2); tm_sub["n"] = tm_sub.get("n", 0) + 1
            if prof is not None and updates >= args.profile_at + args.profile_updates:
                prof.__exit__(None, None, None)
                for key in ("self_device_time_total", "self_cpu_time_total"):
                    print(prof.key_averages().table(sort_by=key, row_limit=25), flush=True)
                prof.export_chrome_trace(str(run_dir / f"profile_upd{args.profile_at}.json"))
                print(f"[B64 profile] updates {args.profile_at}-{updates}: trace {run_dir / f'profile_upd{args.profile_at}.json'}", flush=True)
                prof = None; args.profile_at = 0
            tm["update"] += time.perf_counter() - tu

        # ---------------- logging / eval / checkpoints -----------------------------------------
        # `stepped` guard: the losses printed below exist only once this process has run an update. With
        # num_envs * replay_ratio < 1 the first post-warm-up iterations run none (NameError).
        if stepped and (env_steps - last_log >= 1000 or (env_steps >= args.total_steps and updates % 1000 == 0)):
            last_log = env_steps
            el = time.time() - t0
            tot = max(sum(tm.values()), 1e-9)
            ret = float(np.mean(recent_returns[-20:])) if recent_returns else 0.0
            mixer_w = (sum(q.detach().norm() ** 2 for n, q in model.named_parameters()
                           if "mixer" in n and n.endswith("weight")) ** 0.5).item() if args.trunk != "resnet" else None
            cache_hits = rcache.stats() if rcache is not None else None   # TODO B68: search / bootstrap hit rates since the last line
            gn_hot = gn_hot + 1 if (args.gn_stop > 0 and float(gn) > args.gn_stop) else 0
            if gn_hot >= 3:                          # B37 retry guard: B37's lr 3e-4 runs sat at gn 1e8-1e16 for 20k steps before anyone looked
                print(f"[gn-stop] gradient norm {float(gn):.3g} > {args.gn_stop:g} on 3 consecutive log lines: stopping at env {env_steps}", flush=True)
                raise SystemExit(3)
            print(f"env {env_steps:6d}/{args.total_steps} upd {updates:6d}/{total_updates} | lr {lr:.3f} | "
                  f"loss v {value_loss.mean().item():.3f} p {policy_loss.mean().item():.3f} "
                  f"r {prefix_loss.mean().item():.3f} c {cons_loss.mean().item():.3f} | gn {float(gn):.2f} | "
                  f"train-life-ret(20) {ret:.2f} | V0 {v0.float().mean().item():.3f} tgt {val_t[:, 0].mean():.3f} | "
                  + (f"|mixer W| {mixer_w:.2f} | " if mixer_w is not None else "")
                  + (f"explore {explore_steps} | " if args.stagnation_explore > 0 else "")
                  + ("cache hit s/v {:.2f}/{:.2f} | ".format(*cache_hits) if rcache is not None else "") +
                  f"time c/t/u {100 * tm['collect'] / tot:.0f}/{100 * tm['targets'] / tot:.0f}/"
                  f"{100 * tm['update'] / tot:.0f}% | {el / 60:.1f} min", flush=True)
            if play["steps"]:
                ps = play_summary(play)
                print("[play] " + " | ".join(f"{k} {v}" for k, v in ps.items() if k != "metrics"), flush=True)
                log(ps["metrics"], env_steps)
                play = play_window(A)
            if prof_t and tm_sub.get("n"):
                n_ = tm_sub["n"]
                print(f"[B64 split] ms/update over {n_} updates: " +
                      " ".join(f"{k} {1000 * v / n_:.1f}" for k, v in tm_sub.items() if k != "n"), flush=True)
                tm_sub.clear()
            log({"loss/value": value_loss.mean().item(), "loss/policy": policy_loss.mean().item(),
                 "loss/value_prefix": prefix_loss.mean().item(), "loss/consistency": cons_loss.mean().item(),
                 "train/grad_norm": float(gn), "train/lr": lr, "train/life_return_20": ret,
                 "train/updates": updates, "train/v0_mean": v0.float().mean().item(),
                 **({"train/mixer_weight_norm": mixer_w} if mixer_w is not None else {}),
                 **({"train/reanalysis_cache_hit": cache_hits[0], "train/bootstrap_cache_hit": cache_hits[1]} if rcache is not None else {}),
                 "train/value_target_mean": float(val_t[:, 0].mean()),
                 "train/policy_target_entropy": float(-(pol_t[:, 0] * np.log(pol_t[:, 0] + 1e-8)).sum(-1).mean()),
                 "train/explore_steps": explore_steps},
                env_steps)
        done_all = updates >= total_updates
        if ((env_steps >= next_eval and env_steps <= args.total_steps) or done_all) and args.eval_mode == "deferred":
            # S-111: evals took 1-2 h each (a policy that stops pressing FIRE plays the 27k-step cap) and
            # blocked training. Save the weights here; eval_ez_checkpoints.py scores them in its own process.
            next_eval += args.eval_interval
            tag = "final" if done_all else f"env{env_steps}"
            save(f"env{env_steps}_upd{updates}", {"eval_pending": True, "eval_tag": tag})
            save("latest", {"eval_pending": True, "eval_tag": tag})
            print(f"[EVAL-DEFERRED] {tag} env {env_steps} upd {updates}: checkpoint_env{env_steps}_upd{updates}.pt",
                  flush=True)
        elif (env_steps >= next_eval and env_steps <= args.total_steps) or done_all:
            next_eval += args.eval_interval
            sc = evaluate(model, mcts, args.env_id, device, args.eval_episodes, args.frame_size, args.seed,
                          graph=args.graph_search)
            mean, se = float(sc.mean()), float(sc.std() / np.sqrt(len(sc)))
            tag = "final" if done_all else f"env{env_steps}"
            print(f"[EVAL] {tag} env {env_steps} upd {updates}: {mean:.2f} +/- {se:.2f} SE "
                  f"(HNS {compute_hns(mean, args.env_id):.1f}%) scores={sc.tolist()}", flush=True)
            log({"eval/score": mean, "eval/se": se, "eval/hns": compute_hns(mean, args.env_id)}, env_steps)
            extra = {"eval_score": mean, "eval_scores": sc.tolist()}
            if mean > best:
                best = mean
                save("best", extra)
            save(f"env{env_steps}_upd{updates}", extra)
            save("latest", extra)
        elif updates and updates % args.ckpt_interval == 0:
            save("latest")

    envs.close()
    if writer:
        writer.close()
    if act_graph is not None:  # searches deeper than the captured bound are re-run eagerly (same result, just slower)
        print(f"graphed search fallbacks: acting {act_graph.fallbacks}/{act_graph.calls}, "
              f"reanalyze {re_graph.fallbacks}/{re_graph.calls}", flush=True)
    if saver is not None:
        saver.join()
    if mlrun:
        try:
            if np.isfinite(best):  # deferred mode: eval_ez_checkpoints.py logs best_eval on the <label>_eval run
                mlflow.log_metric("best_eval", best)
            mlflow.end_run()
        except Exception:
            pass
    print(f"done: best eval {best:.2f}" if np.isfinite(best) else
          "done: evals deferred to eval_ez_checkpoints.py", flush=True)
    return best


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env-id", default="BreakoutNoFrameskip-v4")
    ap.add_argument("--trunk", default="resnet", choices=["resnet", "gtrxl", "qwen"])
    ap.add_argument("--mixer-size", default=None, choices=["small", "30m"],
                    help="TODO B61: named mixer shape (small = 128x2, 4 heads, as GTrXL; 30m = qwen30m 512x8, 8 heads, FFN 1536); "
                         "unset keeps the GTrXL mixer's 128x2")
    ap.add_argument("--mixer-no-pos", action="store_true",
                    help="TODO B70b: the policy-token trunk (--mixer-where pred) adds no learned position table (the 36 absolute position vectors)")
    ap.add_argument("--mixer-qk-norm", action="store_true",
                    help="TODO B67: Qwen3-style per-head RMSNorm on queries and keys in every mixer block (the CARLA --qk_norm block option). "
                         "Bounds the attention logits, which AdamW 3e-4 otherwise grows to ~9,000 (one-hot attention by update 8,004, B66)")
    ap.add_argument("--mixer-where", default="both", choices=["both", "repr", "pred"],
                    help="TODO B68 D1: both = mixer after the representation and after every dynamics step; repr = representation only, "
                         "the dynamics is EZ-V2's conv + ResNet (removes ~85%% of the mixer work in the search). TODO B69: pred = the CARLA "
                         "trunk's layout as the prediction function (state-cell tokens + a policy token, value / policy read from the policy "
                         "token); representation and dynamics are EZ-V2's. Qwen only")
    ap.add_argument("--norm", default="batch", choices=["batch", "heads", "all"],
                    help="batch = EZ-V2; heads/all = LayerNorm instead of BatchNorm (S-058: eval-mode BN drift)")
    ap.add_argument("--total-steps", type=int, default=100_000, help="env (agent) steps collected; < --schedule-steps = screen")
    ap.add_argument("--schedule-steps", type=int, default=100_000, help="length of the full run the lr/offline schedules are computed for")
    ap.add_argument("--num-envs", type=int, default=4, help="EZ-V2 data worker uses 4")
    ap.add_argument("--num-simulations", type=int, default=16)
    ap.add_argument("--unroll-steps", type=int, default=5)
    ap.add_argument("--td-steps", type=int, default=5)
    ap.add_argument("--lstm-horizon", type=int, default=5)
    ap.add_argument("--discount", type=float, default=0.997, help="per frame; raised to the 4th power per step")
    ap.add_argument("--frame-size", type=int, default=96)
    ap.add_argument("--train-episode-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--replay-ratio", type=float, default=1.0, help="updates per env step while collecting")
    ap.add_argument("--offline-frac", type=float, default=0.2, help="extra updates after collection (EZ-V2 20k/100k)")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr-warmup", type=float, default=0.01)
    ap.add_argument("--lr-decay-steps", type=int, default=100_000)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--mixer-out-init-std", type=float, default=0.0,
                    help="std of the GTrXL mixer's output projection at init (0 = zero-init, exact identity, the original)")
    ap.add_argument("--graph-search", action="store_true",
                    help="replay the acting, reanalyze and inline-eval searches from CUDA graphs (TODO B23 step 1b); "
                         "bit-identical to the eager search, CUDA only")
    ap.add_argument("--compile-model", action="store_true",
                    help="TODO B68 E8: torch.compile the online model's mixers in place (training forward + backward and the acting search; "
                         "state-dict keys unchanged)")
    ap.add_argument("--mixer-optimizer", choices=["sgd", "adamw", "adamw_carla"], default="sgd",
                    help="TODO B37: optimiser of the GTrXL mixer parameters (adamw: betas 0.9/0.95, --mixer-lr, --mixer-weight-decay default 0.05; "
                         "the rest of the network keeps SGD). TODO B66 adamw_carla: CARLA's recipe for the same Qwen network (default "
                         "betas, decay on matrices only, --mixer-weight-decay default 1e-4, warm-up --mixer-warmup-frac then cosine to 1e-6)")
    ap.add_argument("--mixer-lr", type=float, default=3e-4)
    ap.add_argument("--mixer-warmup", type=int, default=500, help="warm-up updates of the mixer's AdamW lr (adamw mode)")
    ap.add_argument("--mixer-warmup-frac", type=float, default=0.05,
                    help="TODO B66: warm-up fraction of the run for adamw_carla (CARLA's --warmup_frac)")
    ap.add_argument("--mixer-grad-clip", type=float, default=1.0, help="mixer-only gradient-norm clip (adamw mode)")
    ap.add_argument("--mixer-bg-init", type=float, default=2.0, help="GRU gate bias init of the mixer blocks (2 = closed, ~88%% skip; 0 = open)")
    ap.add_argument("--simnorm", type=int, default=0, help="TODO B22: SimNorm group size on the latent (0 = off; 8 = TD-MPC2)")
    ap.add_argument("--mixer-pos", default="abs", choices=["abs", "sym"],
                    help="TODO B20: GTrXL position encoding: abs = learned table (default), sym = symmetric offset bias (translation + mirror)")
    ap.add_argument("--audit-every", type=int, default=0, help="TODO B37: log the mixer's per-category relative weight change and gradient norm every N updates (0 = off)")
    ap.add_argument("--mixer-weight-decay", type=float, default=None,
                    help="weight decay of the GTrXL token mixer's parameters (default: --weight-decay). With SGD lr 0.2, "
                         "wd 1e-4 and momentum 0.9 every mixer weight shrank by 0.13 per 10k updates and was ~0 by 40k "
                         "(S-115): the mixer never trained")
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--gn-stop", type=float, default=0.0,
                    help="TODO B37 retry: stop the run when the (clipped-input) gradient norm exceeds this on 3 consecutive log lines (0 = off)")
    ap.add_argument("--nan-guard", action="store_true",
                    help="TODO B37 retry: skip an update whose gradient norm (rest or mixer) is not finite instead of stepping (counted and printed)")
    ap.add_argument("--reward-coeff", type=float, default=1.0)
    ap.add_argument("--value-coeff", type=float, default=0.5)
    ap.add_argument("--policy-coeff", type=float, default=1.0)
    ap.add_argument("--consistency-coeff", type=float, default=5.0)
    ap.add_argument("--priority-alpha", type=float, default=1.0)
    ap.add_argument("--priority-beta", type=float, default=1.0)
    ap.add_argument("--target-update-interval", type=int, default=200)
    ap.add_argument("--compile-search", action="store_true",
                    help="TODO B68 E1: torch.compile the target network's recurrent_inference used by the reanalysis search (fuses its "
                         "elementwise work; as close to an fp32 search as the eager path; 1.87x on the 30m graphed search)")
    ap.add_argument("--reanalyze-cache", action="store_true",
                    help="TODO B68: search each replay position once per target window and reuse the result (noise hashed per position "
                         "and window); 19%% of the searches of a 100k run at a 200-update window. Off = fresh noise per draw (EZ-V2)")
    ap.add_argument("--reanalyze-merge", action="store_true",
                    help="TODO B68 M1 (needs --reanalyze-cache and --priority-alpha 0): draw the batches of the updates between two env "
                         "steps (once collection has ended: of the target window) up front, in the same order, and search all their cache "
                         "misses in one call. Same batches and targets; a search's fixed cost (~95 ms on the 30m mixer) is paid once, not per update")
    ap.add_argument("--start-transitions", type=int, default=2000)
    ap.add_argument("--mixed-value-start", type=int, default=30_000, help="updates before search values replace old targets")
    ap.add_argument("--mixed-value-threshold", type=int, default=5_000, help="transitions counted as 'recent'")
    ap.add_argument("--eval-interval", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eval-mode", default="inline", choices=["inline", "deferred"],
                    help="deferred: only save checkpoint_env*_upd*.pt at eval points; score them with "
                         "eval_ez_checkpoints.py in a separate process (training does not wait)")
    ap.add_argument("--ckpt-interval", type=int, default=2_500, help="updates between checkpoint_latest saves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--log-dir", default="results/100k_benchmark/S058_ezv2_match")
    ap.add_argument("--experiment", default="Atari_EZV2_Match")
    ap.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--stagnation-explore", type=int, default=0,
                    help="TODO B62 fix 1: after this many reward-free agent steps an env acts randomly for --stagnation-burst steps "
                         "(training only; 0 = off, the EZ-V2 recipe)")
    ap.add_argument("--stagnation-burst", type=int, default=30, help="random steps per stagnation burst (B62 fix 1)")
    ap.add_argument("--policy-target-smoothing", type=float, default=0.0,
                    help="TODO B62 fix 2: mix the search policy targets with uniform by this weight (0 = off, the EZ-V2 recipe)")
    ap.add_argument("--profile-targets", action="store_true",
                    help="TODO B64: print the per-update ms of replay sampling, bootstrap values, root inference, reanalysis search, "
                         "forward and backward at each log line (syncs the GPU at every mark: diagnostics only)")
    ap.add_argument("--profile-at", type=int, default=0,
                    help="TODO B64: run torch.profiler from this update for --profile-updates updates, print the top kernels and "
                         "CPU ops and write a Chrome trace to the run directory (0 = off)")
    ap.add_argument("--profile-updates", type=int, default=20)
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="use channels_last memory format for faster Tensor Core convolutions on Ampere+")
    ap.add_argument("--async-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="save checkpoints and replay buffers in a background thread to prevent training stalls")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
