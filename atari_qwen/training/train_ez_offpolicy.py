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
import os
import random
import sys
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
from atari_qwen.mcts.gumbel_mcts import GumbelMCTS
from atari_qwen.eval.evaluate import compute_hns


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
        p = np.asarray(p, dtype=np.float64)
        self.prio[t, e] = p
        self.max_prio = max(self.max_prio, float(p.max()))

    def sample(self, B: int, horizon: int, beta: float = 1.0):
        """(t, e) window starts with horizon future steps stored, priority-sampled; IS weights."""
        n = self.size - horizon
        if n <= 0:
            raise ValueError("not enough data for a window")
        p = self.prio[:n].reshape(-1) ** self.alpha
        p = p / p.sum()
        flat = np.random.choice(p.size, size=B, p=p)
        t, e = flat // self.E, flat % self.E
        w = (p[flat] * p.size) ** (-beta)
        return t, e, (w / w.max()).astype(np.float32)

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


def to_input(stacks_u8: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(stacks_u8).to(device, non_blocking=True).float().div_(255.0)


@torch.no_grad()
def run_search(model, mcts: GumbelMCTS, obs: torch.Tensor, add_noise: bool, chunk: int = 2048):
    vals, pols, acts = [], [], []
    for i in range(0, obs.shape[0], chunk):
        s, v, p = model.initial_inference(obs[i:i + chunk])
        rv, rp, ra = mcts.search(model, s, model.support.vector_to_scalar(v), p, add_noise=add_noise)
        vals.append(rv); pols.append(rp); acts.append(ra)
    return np.concatenate(vals), np.concatenate(pols), np.concatenate(acts)


@torch.no_grad()
def evaluate(model, mcts, env_id: str, device, episodes: int, frame_size: int, seed: int,
             max_steps: int = 27_000) -> np.ndarray:
    """One full-game episode (no episodic life, raw reward) per parallel env, search without noise."""
    model.eval()
    envs = make_vector_atari_envs(env_id, num_envs=episodes, seed=seed + 10_000, clip_reward=False,
                                  episodic_life=False, frame_size=frame_size, grayscale=False,
                                  fire_reset=False, max_episode_steps=max_steps)
    obs, _ = envs.reset(seed=seed + 10_000)
    scores = np.zeros(episodes)
    finished = np.zeros(episodes, dtype=bool)
    for _ in range(max_steps):
        _, _, a = run_search(model, mcts, to_input(obs, device), add_noise=False)
        obs, r, term, trunc, _ = envs.step(a)
        scores += np.where(finished, 0.0, r)
        finished |= np.logical_or(term, trunc)
        if finished.all():
            break
    envs.close()
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

    E, K, TD, H = args.num_envs, args.unroll_steps, args.td_steps, args.lstm_horizon
    gamma = args.discount ** 4                       # per agent step (frame skip 4), as EZ-V2
    envs = make_vector_atari_envs(args.env_id, num_envs=E, seed=args.seed, clip_reward=True, episodic_life=True,
                                  frame_size=args.frame_size, grayscale=False, fire_reset=False,
                                  max_episode_steps=args.train_episode_steps, same_step_autoreset=True)
    A = envs.single_action_space.n
    support = DiscreteSupport()
    model = EZV2Model(A, obs_channels=3 * 4, support=support, trunk=args.trunk,
                      state_hw=int(np.ceil(args.frame_size / 16))).to(device)
    target = copy.deepcopy(model).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    mcts = GumbelMCTS(A, support, num_simulations=args.num_simulations, discount=gamma, lstm_horizon=H)
    print(f"--> {run_name}: trunk={args.trunk} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"envs={E} sims={args.num_simulations} batch={args.batch_size} gamma/step={gamma:.4f}", flush=True)

    steps_per_env = args.total_steps // E + K + TD + 16
    replay = EZReplay(steps_per_env, E, (3, args.frame_size, args.frame_size), A, alpha=args.priority_alpha,
                      device=device)
    # Schedules follow the FULL run (--schedule-steps, EZ-V2: 100k); a shorter --total-steps is an exact
    # prefix of it (the CARLA --stop_epoch rule). Scaling the 1% warm-up to a 10k screen gave lr 0.2
    # after 120 updates instead of 1,000 and let the value head inflate itself (S-058, S058a).
    full_online = int(args.schedule_steps * args.replay_ratio)
    warm = max(1, int(full_online * args.lr_warmup))
    online_updates = int(args.total_steps * args.replay_ratio)
    total_updates = online_updates + (int(full_online * args.offline_frac) if args.total_steps >= args.schedule_steps else 0)

    env_steps, updates, best = 0, 0, -float("inf")
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); target.load_state_dict(ck["target"]); opt.load_state_dict(ck["opt"])
        env_steps, updates, best = ck["env_steps"], ck["updates"], ck.get("best", best)
        rp = Path(args.resume_from).with_name("replay_latest.npz")
        if rp.exists():
            replay.load(rp)
        print(f"--> resumed at env step {env_steps:,}, update {updates:,} (replay {replay.transitions:,})", flush=True)

    def save(tag: str, extra=None):
        st = {"model": model.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(),
              "env_steps": env_steps, "updates": updates, "best": best, "args": vars(args)}
        if extra:
            st.update(extra)
        tmp = ckpt_dir / f"checkpoint_{tag}.pt.tmp"
        torch.save(st, tmp); os.replace(tmp, ckpt_dir / f"checkpoint_{tag}.pt")
        if tag == "latest":
            replay.save(ckpt_dir / "replay_latest.npz")

    obs, _ = envs.reset(seed=args.seed)
    amp = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    ep_ret = np.zeros(E); recent_returns = []
    t0 = time.time(); last_log = env_steps; next_eval = (env_steps // args.eval_interval + 1) * args.eval_interval
    tm = {"collect": 0.0, "targets": 0.0, "update": 0.0}
    upd_credit = 0.0
    horizon = K + TD

    while updates < total_updates:
        # ---------------- collect (stops at total_steps; the rest is offline training) ----------
        if env_steps < args.total_steps:
            tc = time.perf_counter()
            model.eval()
            rv, rp_, ra = run_search(model, mcts, to_input(obs, device), add_noise=True)
            nobs, r, term, trunc, _ = envs.step(ra)
            d = np.logical_or(term, trunc)
            replay.add(obs[:, -3:], ra, r, d, rp_, rv)
            ep_ret += r
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

        for _ in range(n_upd):
            if updates >= total_updates:
                break
            lr = args.lr * min(1.0, (updates + 1) / warm) * (0.1 ** (max(0, updates - warm) // args.lr_decay_steps))
            for g_ in opt.param_groups:
                g_["lr"] = lr

            # ---------------- targets (target network, no grad) ---------------------------------
            tt = time.perf_counter()
            B = args.batch_size
            t, e, isw = replay.sample(B, horizon, beta=args.priority_beta)
            ks = np.arange(horizon + 1)
            pos = t[:, None] + ks                                        # (B, K+TD+1)
            dn = replay.done[pos[:, :-1], e[:, None]]                     # done after step t+k
            exists = np.concatenate([np.ones((B, 1), bool), np.cumprod(~dn, axis=1).astype(bool)], axis=1)
            rew = replay.rew[pos[:, :-1], e[:, None]] * exists[:, :-1]  # r_{t+k}, 0 past episode end
            ee = np.repeat(e, K + 1)
            root_stacks = replay.stacks(pos[:, :K + 1].reshape(-1), ee)          # states t..t+K
            boot_stacks = replay.stacks(pos[:, TD:TD + K + 1].reshape(-1), ee)   # states t+TD..t+K+TD
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp):
                x_boot = boot_stacks
                _, vb, _ = target.initial_inference(x_boot)
                v_boot = support.vector_to_scalar(vb).float().cpu().numpy().reshape(B, K + 1)
                x_root = root_stacks
                s_re, v_re, p_re = target.initial_inference(x_root)
                srch_v, srch_pi, _ = mcts.search(target, s_re, support.vector_to_scalar(v_re), p_re, add_noise=True)
            srch_v = srch_v.reshape(B, K + 1); srch_pi = srch_pi.reshape(B, K + 1, A)
            disc = gamma ** np.arange(TD)
            val_t = np.zeros((B, K + 1), np.float32)
            for k in range(K + 1):
                val_t[:, k] = (rew[:, k:k + TD] * disc).sum(1) + (gamma ** TD) * v_boot[:, k] * exists[:, k + TD]
            ex_k = exists[:, :K + 1]
            val_t *= ex_k
            srch_v = srch_v * ex_k
            if updates >= args.mixed_value_start:
                recent = (t * E + e) > (replay.transitions - args.mixed_value_threshold)
                val_t = np.where(recent[:, None], val_t, srch_v)
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
            with torch.autocast(device_type=device.type, dtype=amp):
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
            opt.zero_grad(set_to_none=True)
            wloss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            updates += 1
            replay.update_priorities(t, e, np.abs(v0.float().cpu().numpy() - val_t[:, 0]) + 1e-6)
            if updates % args.target_update_interval == 0:
                target.load_state_dict(model.state_dict())
            tm["update"] += time.perf_counter() - tu

        # ---------------- logging / eval / checkpoints -----------------------------------------
        if env_steps - last_log >= 1000 or (env_steps >= args.total_steps and updates % 1000 == 0):
            last_log = env_steps
            el = time.time() - t0
            tot = max(sum(tm.values()), 1e-9)
            ret = float(np.mean(recent_returns[-20:])) if recent_returns else 0.0
            print(f"env {env_steps:6d}/{args.total_steps} upd {updates:6d}/{total_updates} | lr {lr:.3f} | "
                  f"loss v {value_loss.mean().item():.3f} p {policy_loss.mean().item():.3f} "
                  f"r {prefix_loss.mean().item():.3f} c {cons_loss.mean().item():.3f} | gn {float(gn):.2f} | "
                  f"train-life-ret(20) {ret:.2f} | V0 {v0.float().mean().item():.3f} tgt {val_t[:, 0].mean():.3f} | "
                  f"time c/t/u {100 * tm['collect'] / tot:.0f}/{100 * tm['targets'] / tot:.0f}/"
                  f"{100 * tm['update'] / tot:.0f}% | {el / 60:.1f} min", flush=True)
            log({"loss/value": value_loss.mean().item(), "loss/policy": policy_loss.mean().item(),
                 "loss/value_prefix": prefix_loss.mean().item(), "loss/consistency": cons_loss.mean().item(),
                 "train/grad_norm": float(gn), "train/lr": lr, "train/life_return_20": ret,
                 "train/updates": updates, "train/v0_mean": v0.float().mean().item(),
                 "train/value_target_mean": float(val_t[:, 0].mean()),
                 "train/policy_target_entropy": float(-(pol_t[:, 0] * np.log(pol_t[:, 0] + 1e-8)).sum(-1).mean())},
                env_steps)
        done_all = updates >= total_updates
        if (env_steps >= next_eval and env_steps <= args.total_steps) or done_all:
            next_eval += args.eval_interval
            sc = evaluate(model, mcts, args.env_id, device, args.eval_episodes, args.frame_size, args.seed)
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
    if mlrun:
        try:
            mlflow.log_metric("best_eval", best); mlflow.end_run()
        except Exception:
            pass
    print(f"done: best eval {best:.2f}", flush=True)
    return best


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env-id", default="BreakoutNoFrameskip-v4")
    ap.add_argument("--trunk", default="resnet", choices=["resnet", "gtrxl"])
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
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--reward-coeff", type=float, default=1.0)
    ap.add_argument("--value-coeff", type=float, default=0.5)
    ap.add_argument("--policy-coeff", type=float, default=1.0)
    ap.add_argument("--consistency-coeff", type=float, default=5.0)
    ap.add_argument("--priority-alpha", type=float, default=1.0)
    ap.add_argument("--priority-beta", type=float, default=1.0)
    ap.add_argument("--target-update-interval", type=int, default=200)
    ap.add_argument("--start-transitions", type=int, default=2000)
    ap.add_argument("--mixed-value-start", type=int, default=30_000, help="updates before search values replace old targets")
    ap.add_argument("--mixed-value-threshold", type=int, default=5_000, help="transitions counted as 'recent'")
    ap.add_argument("--eval-interval", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--ckpt-interval", type=int, default=2_500, help="updates between checkpoint_latest saves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--log-dir", default="results/100k_benchmark/S058_ezv2_match")
    ap.add_argument("--experiment", default="Atari_EZV2_Match")
    ap.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    ap.add_argument("--resume-from", default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
