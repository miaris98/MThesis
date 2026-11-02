"""TODO A74 (2026-11-02): the PPO learner's pieces for the maneuver head (src/models/world_on_rails/maneuver_rl.py).

The agents write one npz per route (token, action, behaviour log-prob, value, reward, dt); the learner turns them into advantages with GAE
discounted per elapsed decision unit (a committed maneuver spans `commit / period` units) and runs clipped PPO on the head alone. The
policy is anchored to the fixed prior the head starts from (E15 with p > 0.99) by a KL penalty, the way RLHF anchors to its reference: the
head moves away from E15 only where the closed-loop return pays for it. Advantages are not normalised here (the learner script decides).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.models.world_on_rails.maneuver_rl import DEFAULT_ACTION, MANEUVERS, ManeuverHead


@dataclass
class PPOConfig:
    gamma: float = 0.99                  # per 0.25 s decision unit: a ~25 s horizon
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    kl_coef: float = 0.02                # KL(pi || E15 prior) per decision
    max_grad_norm: float = 1.0


def prior_log_probs(margin: float = 7.0) -> torch.Tensor:
    """log p of the head's starting distribution (ManeuverHead.reset_to_default): the KL anchor."""
    b = torch.zeros(len(MANEUVERS))
    b[DEFAULT_ACTION] = margin
    return b.log_softmax(-1)


def gae(rewards: np.ndarray, values: np.ndarray, dt: np.ndarray, gamma: float, lam: float, last_value: float = 0.0):
    """Advantages and returns of one episode; decision t is discounted by gamma ** dt[t] (its duration in decision units). A route ends the
    episode, so the bootstrap after the last decision is last_value (0 for a finished route)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    nxt_adv = 0.0
    for t in reversed(range(T)):
        g = gamma ** float(dt[t])
        next_v = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + g * next_v - values[t]
        nxt_adv = delta + g * lam * nxt_adv
        adv[t] = nxt_adv
    return adv, adv + np.asarray(values, dtype=np.float64)


def ppo_update(head: ManeuverHead, feats: torch.Tensor, actions: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
               returns: torch.Tensor, prior_logp: torch.Tensor, cfg: PPOConfig, optimizer=None) -> Dict[str, float]:
    """Clipped PPO epochs over one batch; lr = 0 computes the statistics without stepping."""
    opt = optimizer if optimizer is not None else (torch.optim.Adam(head.parameters(), lr=cfg.lr) if cfg.lr > 0 else None)
    dev = next(head.parameters()).device
    feats, actions, old_logp, adv, returns = (x.to(dev) for x in (feats, actions, old_logp, adv, returns))
    prior_logp = prior_logp.to(dev)
    head.train()
    n = len(actions)
    sums = {"loss": 0.0, "policy": 0.0, "value_loss": 0.0, "kl_prior": 0.0, "entropy": 0.0, "clip_frac": 0.0, "approx_kl": 0.0}
    steps = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, cfg.minibatch):
            idx = perm[i:i + cfg.minibatch]
            logits, v = head(feats[idx])
            logp_all = logits.log_softmax(-1)
            logp = logp_all.gather(1, actions[idx, None]).squeeze(1)
            ratio = (logp - old_logp[idx]).exp()
            a = adv[idx]
            surr = torch.min(ratio * a, ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * a).mean()
            pi = logp_all.exp()
            kl_prior = (pi * (logp_all - prior_logp)).sum(-1).mean()
            ent = -(pi * logp_all).sum(-1).mean()
            vloss = F.mse_loss(v, returns[idx])
            loss = -surr + cfg.vf_coef * vloss + cfg.kl_coef * kl_prior - cfg.ent_coef * ent
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.max_grad_norm)
                opt.step()
            with torch.no_grad():
                sums["loss"] += float(loss); sums["policy"] += float(surr); sums["value_loss"] += float(vloss)
                sums["kl_prior"] += float(kl_prior); sums["entropy"] += float(ent)
                sums["clip_frac"] += float(((ratio - 1.0).abs() > cfg.clip).float().mean())
                sums["approx_kl"] += float((old_logp[idx] - logp).mean())
            steps += 1
    head.eval()
    return {k: v / max(steps, 1) for k, v in sums.items()}


def load_episodes(directory: str) -> List[Dict]:
    """Every finished episode in a rollout directory (agents write `<name>.npz` atomically through a `.tmp`)."""
    out = []
    for f in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        with np.load(f) as z:
            ep = {k: z[k] for k in z.files if k != "info"}
            ep["info"] = json.loads(str(z["info"])) if "info" in z.files else {}
        ep["path"] = f
        out.append(ep)
    return out


def batch_from_episodes(episodes: List[Dict], cfg: PPOConfig):
    """Concatenate episodes into one PPO batch with per-episode GAE."""
    feats, acts, logps, advs, rets = [], [], [], [], []
    for ep in episodes:
        adv, ret = gae(ep["reward"].astype(np.float64), ep["value"].astype(np.float64), ep["dt"], cfg.gamma, cfg.lam)
        feats.append(ep["feat"].astype(np.float32)); acts.append(ep["action"]); logps.append(ep["logp"])
        advs.append(adv); rets.append(ret)
    cat = lambda xs, dt: torch.as_tensor(np.concatenate(xs), dtype=dt)
    return (cat(feats, torch.float32), cat(acts, torch.long), cat(logps, torch.float32), cat(advs, torch.float32), cat(rets, torch.float32))
