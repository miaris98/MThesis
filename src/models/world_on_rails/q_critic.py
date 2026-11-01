"""TODO A71 (2026-11-01): Q-critic head on the Qwen policy token, its loss, and the decision rule act() applies with it.

The head predicts Q(s, a) over the counterfactual action grid of src/training/counterfactual_q.py: the 8 target-speed bins of the speed head x
`LAT_TARGETS` lateral offsets from the route. It is trained by regression on those labels (Q distillation, as World on Rails distils its
Bellman table). At evaluation it is read only when WOR_QCRITIC is set:

    pi(v, d | s)  proportional to  pi_BC(v | s) * prior(d) * exp(beta * Q(s, v, d))

pi_BC(v) is the speed head's softmax; prior(d) favours the own lane (exp(-|d| / lat_sigma)), because the network has no lateral head of its own.
The lateral offset is chosen first: the car leaves its lane only when the lateral marginal puts at least `lat_threshold` on one offset, and a
chosen offset is then held for `hold_ticks` agent ticks (20 = 1 s at 20 Hz, the label's own commitment) so the aim point cannot flicker. The
commanded speed is the mean of pi(v | executed d), never the average over lane changes the car does not make (review 2026-11-01). The waypoints
of a shifted choice move along the route normal by their residual to the target offset (offsets are measured from the route, as in the labels),
at most the label's lateral rate times their time ahead. beta = 0 passes the plain decode through unchanged.

    WOR_QCRITIC="beta=1,lat_sigma=1,lat_threshold=0.5,lateral=1,hold_ticks=20,lat_rate=1.75"   (unset, "" or "0" = the plain decode, bit-identical)
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_on_rails.aux_heads import TARGET_SPEEDS

LAT_TARGETS = (-3.5, -1.75, 0.0, 1.75, 3.5)          # = counterfactual_q.QConfig.lat_targets
_ENV_DEFAULTS = {"beta": 1.0, "lat_sigma": 1.0, "lat_threshold": 0.5, "lateral": 1.0, "hold_ticks": 20, "lat_rate": 1.75}
WP_DT = 0.25                                          # the waypoints are the next five 4 Hz poses


class QCriticHead(nn.Module):
    """(B, embed_dim) policy token -> (B, n_speed, n_lat) Q values. Zero-initialised output, so a resumed checkpoint starts at Q = 0."""

    def __init__(self, embed_dim: int, n_speed: int = len(TARGET_SPEEDS), n_lat: int = len(LAT_TARGETS), hidden: int = 256):
        super().__init__()
        self.n_speed, self.n_lat = n_speed, n_lat
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, n_speed * n_lat))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.shape[0], self.n_speed, self.n_lat)


def q_critic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 over the frames whose label is finite; a frame with any NaN is masked out entirely. Zero (with a graph) when none is valid."""
    valid = torch.isfinite(target).flatten(1).all(-1).float()                   # masked without a host sync (no bool(valid.any()) per step)
    err = F.smooth_l1_loss(pred.float(), torch.nan_to_num(target.float()), beta=1.0, reduction="none").flatten(1).mean(-1)
    return (err * valid).sum() / valid.sum().clamp(min=1.0)


def q_critic_term(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weight: float, device) -> Optional[torch.Tensor]:
    """The weighted loss term the trainer adds, or None when the head, the labels or the weight are off."""
    if weight <= 0 or "q_sa" not in out or batch.get("target_qsa") is None:
        return None
    pred = out["q_sa"]
    target = batch["target_qsa"].to(device).float().view_as(pred).clone()
    target[batch["qsa_valid"].to(device) < 0.5] = float("nan")
    return weight * q_critic_loss(pred, target)


def parse_qcritic_env(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec or spec.strip().lower() in ("0", "off", "false", "none"):
        return None
    cfg = dict(_ENV_DEFAULTS)
    for part in spec.replace(";", ",").split(","):
        if "=" in part:
            k, v = (x.strip() for x in part.split("=", 1))
            if k not in _ENV_DEFAULTS:
                raise ValueError(f"WOR_QCRITIC: unknown key {k!r} (known: {sorted(_ENV_DEFAULTS)})")
            cfg[k] = float(v)
    return cfg


def decode_with_q(probs: np.ndarray, q: np.ndarray, bins: np.ndarray, beta: float, lat_sigma: float = 1.0, lat_threshold: float = 0.5,
                  lateral: bool = True, forced_d: Optional[float] = None) -> Tuple[float, float, Dict]:
    """(target speed in m/s, lateral offset in m, debug). The offset is chosen first (or forced, e.g. held from an earlier tick); the speed is
    the mean of pi(v | that offset)."""
    probs = np.asarray(probs, dtype=np.float64)
    if beta == 0.0:
        return float((probs * bins).sum()), 0.0, {"p_v": probs, "p_lat": None, "candidate": 0.0}
    lat = np.asarray(LAT_TARGETS)
    logits = (np.log(np.clip(probs, 1e-12, None))[:, None] - np.abs(lat)[None, :] / max(lat_sigma, 1e-6)
              + beta * (np.asarray(q, dtype=np.float64) - np.max(q)))
    joint = np.exp(logits - logits.max())
    joint /= joint.sum()
    p_lat = joint.sum(0)
    j = int(np.argmax(p_lat))
    candidate = float(lat[j]) if lateral and lat[j] != 0.0 and p_lat[j] >= lat_threshold else 0.0
    d = candidate if forced_d is None else float(forced_d)
    col = joint[:, int(np.argmin(np.abs(lat - d)))]
    p_v = col / max(col.sum(), 1e-300)
    return float((p_v * bins).sum()), d, {"p_v": p_v, "p_lat": p_lat, "candidate": candidate}


def q_decode_speed(policy, out: Dict[str, torch.Tensor], probs: np.ndarray, target_speed_kmh: float) -> Tuple[float, np.ndarray]:
    """act()'s hook before the safety layer: (target km/h, the speed posterior the rest of act() sees). A no-op unless WOR_QCRITIC is set."""
    if not getattr(policy, "_qcritic_ready", False):
        policy._qcritic_cfg, policy._qcritic_ready = parse_qcritic_env(os.environ.get("WOR_QCRITIC")), True
        if policy._qcritic_cfg is not None:
            if os.environ.get("WOR_SPEED_DECODE"):
                raise RuntimeError("WOR_QCRITIC decodes the speed as the mean of pi(v | d); it cannot be combined with WOR_SPEED_DECODE")
            print(f"[A71] Q-critic decode on: {policy._qcritic_cfg}", flush=True)
    policy._q_choice = None
    cfg = policy._qcritic_cfg
    if cfg is None:
        return target_speed_kmh, probs
    if "q_sa" not in out:
        raise RuntimeError("WOR_QCRITIC is set but this checkpoint has no Q-critic head (trained without --q_critic 1): refusing a run that "
                           "would log the switch and drive the plain decode")
    q = out["q_sa"][0].float().cpu().numpy()
    if cfg["beta"] == 0.0:
        policy._q_choice = {"q_sa": q, "q_lat": 0.0, "p_lat": None}
        return target_speed_kmh, probs
    lock = getattr(policy, "_q_lock", None) or {"d": 0.0, "left": 0}
    _, cand, dbg0 = decode_with_q(probs, q, policy.target_speed_head.bins.cpu().numpy(), cfg["beta"], cfg["lat_sigma"], cfg["lat_threshold"],
                                  bool(cfg["lateral"]))
    if lock["left"] > 0:
        lock["left"] -= 1                                       # hold the earlier choice
    elif cand != lock["d"]:
        lock = {"d": cand, "left": int(cfg["hold_ticks"]) - 1}  # a new choice, held for hold_ticks ticks including this one
    policy._q_lock = lock
    v, d, dbg = decode_with_q(probs, q, policy.target_speed_head.bins.cpu().numpy(), cfg["beta"], cfg["lat_sigma"], cfg["lat_threshold"],
                              bool(cfg["lateral"]), forced_d=lock["d"])
    policy._q_choice = {"q_sa": q, "q_lat": d, "p_lat": dbg["p_lat"], "lat_rate": cfg["lat_rate"]}
    return 3.6 * v, dbg["p_v"]


def shift_toward_lateral(wps, route, d_target: float, lat_rate: float) -> np.ndarray:
    """Each waypoint moved along the route normal by the residual to the target offset, clip(d_target - d_wp, +-lat_rate * t_k) (waypoint k is
    (k + 1) * 0.25 s ahead): offsets are measured from the route (a BC swerve already at the target is left alone, never doubled) while the
    network's own longitudinal plan and lane curvature are kept (Gemini review 2026-11-01)."""
    from src.training.counterfactual_q import RouteFrame
    wps = np.array(wps, dtype=np.float64, copy=True)
    rf = RouteFrame(np.asarray(route, dtype=np.float64), from_origin=False)
    s, d_wp = rf.project(wps[:, 0], wps[:, 1])
    t = WP_DT * (np.arange(len(wps)) + 1)
    delta = np.clip(d_target - d_wp, -lat_rate * t, lat_rate * t)
    _, _, yaw = rf.pose(s, np.zeros_like(s))
    wps[:, 0] -= delta * np.sin(yaw)
    wps[:, 1] += delta * np.cos(yaw)
    return wps.astype(np.float32)


def q_shift_lateral(policy, wps, route) -> np.ndarray:
    """act()'s hook before the controller: a chosen lateral offset re-anchors the waypoints to the route (+d = +y = right); d = 0 leaves BC."""
    choice = getattr(policy, "_q_choice", None)
    if choice is None:
        return wps
    if getattr(policy, "last_debug", None) is not None:
        policy.last_debug.update(choice)
    if choice["q_lat"] == 0.0 or route is None:
        return wps
    r = np.asarray(route.detach().cpu() if isinstance(route, torch.Tensor) else route, dtype=np.float64).reshape(-1, 2)
    wps = shift_toward_lateral(wps, r, choice["q_lat"], choice.get("lat_rate", _ENV_DEFAULTS["lat_rate"]))
    if getattr(policy, "last_debug", None) is not None:
        policy.last_debug["waypoints"] = wps                   # the clip recorder shows what the car steered on
    return wps
