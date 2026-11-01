"""TODO A72 (2026-11-01): discrete offline RL (IQL / CQL) on the logged PDM-Lite transitions, for the Qwen policy's Q head.

Action space = A71's grid: the 8 target-speed bins x 5 lateral offsets (flattened, 40). Each logged frame k carries the expert's action and the
reward of its real step k -> k+1 (counterfactual_q.logged_step_reward, A71's reward terms), and the next state is the logged frame k+1
(wor_dataset with WOR_TRANSITIONS=1). Only the expert's own transitions have a next observation, so TD runs on them alone; the other 39 actions
are shaped by the CQL penalty and, when its weight is > 0, by A71's counterfactual labels (q_critic_loss_weight).

    IQL  L_V = E[ expectile_tau( Qt(s, a_exp) - V(s) ) ]      Qt = Polyak-averaged copy of the Q head on the detached policy token
         L_Q = E[ huber( Q(s, a_exp) - (r + gamma * has_next * V(s')) ) ]      V(s') from the current V head, no gradient
    CQL  L_Q + cql_alpha * E[ logsumexp_a Q(s, a) - Q(s, a_exp) ]   (exact over the 40 actions), with L_V as in IQL for the bootstrap

Policy: IQL's extraction pi(a|s) ~ pi_BC(a|s) exp(beta A(s, a)) equals pi_BC exp(beta Q(s, a)) because V(s) cancels over actions, which is
the A71 decoder (q_critic.decode_with_q, WOR_QCRITIC="beta=..."); no separate policy head. gamma is per 0.25 s step (0.95, as the A71 labels).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OfflineRLConfig:
    mode: str = "iql"              # "iql" | "cql"
    gamma: float = 0.95            # per 0.25 s step
    tau: float = 0.7               # IQL expectile (App. B)
    cql_alpha: float = 1.0         # CQL App. F, Atari / QR-DQN range
    polyak: float = 0.005          # target Q head
    weight: float = 1.0            # scale of the whole offline-RL term in the training loss

    def __post_init__(self):
        if self.mode not in ("iql", "cql"):
            raise ValueError(f"offline RL mode must be 'iql' or 'cql', got {self.mode!r}")


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """|tau - 1(diff < 0)| * diff^2, elementwise."""
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return w * diff ** 2


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp(min=1.0)


def cql_penalty(q_sa: torch.Tensor, a: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid rows of logsumexp_a Q(s, a) - Q(s, a_data); q_sa (B, A) flattened."""
    q_a = q_sa.gather(1, a.view(-1, 1)).squeeze(1)
    return _masked_mean(torch.logsumexp(q_sa, dim=1) - q_a, valid)


def offline_rl_terms(q_sa: torch.Tensor, v: torch.Tensor, v_next: torch.Tensor, q_target_sa: torch.Tensor, batch: Dict[str, torch.Tensor],
                     cfg: OfflineRLConfig) -> Dict[str, torch.Tensor]:
    """q_sa (B, A) or (B, n_speed, n_lat); v, v_next (B,); q_target_sa like q_sa; batch: exp_a (long, flattened index), exp_r, has_next,
    exp_valid. Returns rl_v, rl_q, (rl_cql) and total (cfg.weight * sum). v_next and q_target_sa never receive gradient."""
    B = q_sa.shape[0]
    q_sa, q_target_sa = q_sa.reshape(B, -1).float(), q_target_sa.reshape(B, -1).detach().float()
    dev = q_sa.device
    a = batch["exp_a"].to(dev).long().clamp(min=0)
    valid = batch["exp_valid"].to(dev).float()
    r = torch.nan_to_num(batch["exp_r"].to(dev).float())
    cont = batch["has_next"].to(dev).float()
    q_a = q_sa.gather(1, a.view(-1, 1)).squeeze(1)
    qt_a = q_target_sa.gather(1, a.view(-1, 1)).squeeze(1)
    l_v = _masked_mean(expectile_loss(qt_a - v.float(), cfg.tau), valid)
    target = r + cfg.gamma * cont * v_next.detach().float()
    l_q = _masked_mean(F.smooth_l1_loss(q_a, target, beta=1.0, reduction="none"), valid)
    out = {"rl_v": l_v, "rl_q": l_q}
    total = l_v + l_q
    if cfg.mode == "cql":
        out["rl_cql"] = cql_penalty(q_sa, a, valid)
        total = total + cfg.cql_alpha * out["rl_cql"]
    out["total"] = cfg.weight * total
    return out


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(1.0 - tau).add_(s.detach(), alpha=tau)


def next_inputs(batch: Dict[str, torch.Tensor], device):
    """(rgb, vision_features, speed, command, route) of the next frame, from the next_* keys WOR_TRANSITIONS adds; rgb as the trainer feeds it."""
    rgb = batch.get("next_rgb")
    feats = batch.get("next_vision_features")
    return (rgb.to(device, non_blocking=True) if rgb is not None else None,
            feats.to(device, non_blocking=True) if feats is not None else None,
            batch["next_speed"].to(device), batch["next_command"].to(device), batch["next_route"].to(device))
