"""TODO A73 (2026-11-01): offline training of the latent world model (src/models/world_on_rails/latent_world_model.py) on PDM-Lite windows.

Data: windows of K + 1 consecutive logged frames (K = 5, 1.25 s at 4 Hz, as MuZero Unplugged / ROSMO / our EZ port): the frozen pooled map of
each frame from the A59 pooled cache, and for each of the K transitions the expert's action on A71's grid and the reward of its logged step
(A72's fields in the A71 label files, build_q_labels.py).

Losses over the K-step unroll on predicted latents:
    latent  mean over ensemble heads of || s_hat_k+1 - z_k+1 ||^2, z = the EMA encoder's state of the logged frame (LAW; no pixels)
    reward  Huber(r_hat_k, r_k)
    value   Huber(V(s_hat_k), G_k), G_k = sum_j gamma^(j-k) r_j + gamma^(K-k) V_ema(z_K) (n-step return on real rewards)
    bc      CE(pi(s_hat_k), a_k) (expert data: ROSMO keeps a BC term)
Policy improvement (policy_improvement_loss): a one-step lookahead in the model over all 40 actions,
    Q_hat(s, a) = r_hat(s, a) - penalty * u(s, a) + gamma V(s'(a)),  u = ensemble spread of s'(a) (MOPO / LOMPO conservatism),
    loss = imag_weight * KL(softmax(Q_hat / T) || pi) + bc_weight * CE(pi, a_exp).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.models.world_on_rails.latent_world_model import LatentWorldModel


@dataclass(frozen=True)
class WMLossConfig:
    gamma: float = 0.98                 # per 0.25 s step (ROSMO 0.997^4)
    latent_weight: float = 1.0
    reward_weight: float = 1.0
    value_weight: float = 0.5
    unroll_bc_weight: float = 0.2
    imag_weight: float = 1.0
    bc_weight: float = 0.2              # ROSMO alpha
    penalty: float = 1.0                # MOPO lambda in {1, 5}
    temperature: float = 1.0


def _masked(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp(min=1.0)


def unroll_losses(model: LatentWorldModel, target: LatentWorldModel, feats: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                  valid: torch.Tensor, cfg: WMLossConfig) -> Dict[str, torch.Tensor]:
    """feats (B, K+1, C, gh, gw), actions (B, K) long, rewards (B, K), valid (B, K)."""
    B, K = actions.shape
    valid, rewards = valid.float(), torch.nan_to_num(rewards.float())
    with torch.no_grad():
        z = [target.encode(feats[:, k]) for k in range(K + 1)]
        acc = target.value(z[K])
        G = torch.zeros(B, K, device=feats.device)
        for k in range(K - 1, -1, -1):
            acc = rewards[:, k] + cfg.gamma * acc
            G[:, k] = acc
    s = model.encode(feats[:, 0])
    l_lat = l_rew = l_val = l_bc = 0.0
    for k in range(K):
        m = valid[:, k]
        l_val = l_val + _masked(F.smooth_l1_loss(model.value(s), G[:, k], reduction="none"), m)
        l_bc = l_bc + _masked(F.cross_entropy(model.policy_logits(s), actions[:, k], reduction="none"), m)
        s_mean, s_all, r_hat = model.step(s, actions[:, k])
        l_lat = l_lat + _masked(((s_all - z[k + 1].unsqueeze(0)) ** 2).mean((0, 2, 3)), m)
        l_rew = l_rew + _masked(F.smooth_l1_loss(r_hat, rewards[:, k], reduction="none"), m)
        s = s_mean
    out = {"latent": l_lat / K, "reward": l_rew / K, "value": l_val / K, "bc_unroll": l_bc / K}
    out["total"] = (cfg.latent_weight * out["latent"] + cfg.reward_weight * out["reward"] + cfg.value_weight * out["value"]
                    + cfg.unroll_bc_weight * out["bc_unroll"])
    return out


@torch.no_grad()
def update_target(target: nn.Module, model: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), model.parameters()):
        t.mul_(1.0 - tau).add_(s.detach(), alpha=tau)


def lookahead_q(model: LatentWorldModel, s: torch.Tensor, gamma: float, penalty: float):
    """(Q_hat (B, A), ensemble spread u (B, A)) of a one-step lookahead over every action from states s (B, T, D)."""
    B, A = s.shape[0], model.cfg.n_actions
    s_rep = s.repeat_interleave(A, 0)
    a = torch.arange(A, device=s.device).repeat(B)
    s_mean, s_all, r = model.step(s_rep, a)
    u = s_all.std(0).mean((1, 2))
    q = r - penalty * u + gamma * model.value(s_mean)
    return q.view(B, A), u.view(B, A)


def imagined_return(model: LatentWorldModel, s: torch.Tensor, horizon: int, gamma: float, penalty: float) -> torch.Tensor:
    """Penalised return of the model's greedy policy over `horizon` imagined steps plus the value bootstrap (B,)."""
    ret = torch.zeros(s.shape[0], device=s.device)
    disc = 1.0
    for _ in range(horizon):
        a = model.policy_logits(s).argmax(-1)
        s_mean, s_all, r = model.step(s, a)
        ret = ret + disc * (r - penalty * s_all.std(0).mean((1, 2)))
        disc *= gamma
        s = s_mean
    return ret + disc * model.value(s)


def policy_improvement_loss(model: LatentWorldModel, s: torch.Tensor, a_exp: torch.Tensor, cfg: WMLossConfig) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        q, _ = lookahead_q(model, s.detach(), cfg.gamma, cfg.penalty)
        tgt = F.softmax(q / cfg.temperature, -1)
    logp = F.log_softmax(model.policy_logits(s), -1)
    imag = (tgt * (torch.log(tgt.clamp(min=1e-12)) - logp)).sum(-1).mean()
    bc = F.nll_loss(logp, a_exp.long())
    return {"policy_imag": imag, "policy_bc": bc, "total": cfg.imag_weight * imag + cfg.bc_weight * bc}


class LatentWindowDataset(Dataset):
    """Windows of K + 1 consecutive frames of one route, each with the expert transition (action, reward) of its first K frames.

    Features come straight from the pooled-cache memmaps (`<prefix>.json` keys = image paths relative to data_dir), the transitions from the
    label files (counterfactual_q.load_q_transitions). A window never spans a missing frame or a frame without a transition."""

    def __init__(self, data_dir: str, prefixes: Sequence[str], tag: str, K: int = 5, routes: Sequence[str] = None):
        from src.training.counterfactual_q import load_q_transitions
        from src.training.pooled_feature_cache import cache_files
        self.K, self.prefixes = K, list(prefixes)
        self.meta, self.index = [], {}
        for pi, prefix in enumerate(self.prefixes):
            meta = json.load(open(cache_files(prefix)[0]))
            self.meta.append(meta)
            for row, key in enumerate(meta["keys"]):
                self.index[key] = (pi, row)
        self._mm = [None] * len(self.prefixes)
        by_route = defaultdict(dict)
        for key in self.index:
            route = os.path.dirname(os.path.dirname(key))
            if routes is None or route in routes:
                by_route[route][int(os.path.splitext(os.path.basename(key))[0])] = key
        self.windows: List[tuple] = []
        self.transitions: Dict[str, tuple] = {}
        for route, frames in sorted(by_route.items()):
            tr = load_q_transitions(os.path.join(data_dir, route), tag)
            ids = sorted(frames)
            for f in ids:
                keys = [frames.get(f + j) for j in range(K + 1)]
                fids = [os.path.splitext(os.path.basename(k))[0] if k else None for k in keys]
                if all(keys) and all(fid in tr for fid in fids[:K]):
                    self.windows.append(tuple(keys))
                    for k, fid in zip(keys[:K], fids[:K]):
                        self.transitions[k] = tr[fid]

    def __len__(self) -> int:
        return len(self.windows)

    def _feat(self, key: str) -> np.ndarray:
        from src.training.pooled_feature_cache import cache_files
        pi, row = self.index[key]
        if self._mm[pi] is None:                        # opened lazily in each DataLoader worker (a memmap does not pickle)
            m = self.meta[pi]
            self._mm[pi] = np.memmap(cache_files(self.prefixes[pi])[1], dtype=np.float16, mode="r",
                                     shape=(m["n"], m["channels"], m["grid"][0], m["grid"][1]))
        return np.asarray(self._mm[pi][row], dtype=np.float32)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mm"] = [None] * len(self.prefixes)
        return d

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        keys = self.windows[i]
        a = [self.transitions[k][0] for k in keys[:self.K]]
        r = [self.transitions[k][1] for k in keys[:self.K]]
        return {"feats": torch.from_numpy(np.stack([self._feat(k) for k in keys])), "actions": torch.tensor(a, dtype=torch.long),
                "rewards": torch.tensor(r, dtype=torch.float32), "valid": torch.ones(self.K)}
