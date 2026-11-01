"""TODO A71 (2026-11-01): Q-critic head on the Qwen policy token, its loss, and the decision rule act() applies with it.

The head predicts Q(s, a) over the counterfactual action grid of src/training/counterfactual_q.py: the 8 target-speed bins of the speed head x
`LAT_TARGETS` lateral offsets from the route. It is trained by regression on those labels (Q distillation, as World on Rails distils its
Bellman table). At evaluation it is read only when WOR_QCRITIC is set:

    d = argmax_d max_v Q(s, v, d) if it beats the own lane by lat_margin, else 0;   pi(v | d)  proportional to  pi_BC(v | s) * exp(beta * Q(s, v, d))

pi_BC(v) is the speed head's softmax. The lateral offset is chosen first, from each column's value V(d) = max_v Q(s, v, d): the car leaves its
lane only for an offset worth at least `lat_margin` (Q units; one step of full-speed progress = 1) more than the offset it drives, returns as
soon as its own lane is the best column, never crosses from one side to the other directly, and commits to a new offset only after
`confirm_ticks` agreeing ticks; a chosen offset is then held for `hold_ticks` agent ticks (20 = 1 s at 20 Hz). The commanded speed is the mean of
pi(v | executed d), never the average over lane changes the car does not make (review 2026-11-01): pi_BC(v) exp(beta Q(s, v, d)) in lane; in a
lane change, whose speed BC never planned (its posterior is for the in-lane plan: stop behind the parked car), the bin closest to BC's speed
among those within SPEED_TOL of the column's best (a softmax over Q drove the tied, unreachable fast bins: 13 m/s against an expert target of 8;
where the column is flat, BC decides, red lights included).
Until 2026-11-01 evening the offset was the argmax of the BC-weighted lateral marginal times an own-lane prior, which needed half the mass on one
offset: on route 2664 the car stopped behind the parked car for good (a braking BC judges every column at v = 0, where passing and stopping tie,
and two free side lanes split the mass). The waypoints
of a shifted choice move along the route normal by their residual to the target offset (offsets are measured from the route, as in the labels),
at most the label's lateral rate times their time ahead. beta = 0 passes the plain decode through unchanged.

    WOR_QCRITIC="beta=1,lat_margin=1,stay_margin=0.25,lateral=1,hold_ticks=20,confirm_ticks=5,lat_rate=1.75"   (unset, "" or "0" = the plain decode, bit-identical)
    WOR_QCRITIC_LOG=/path/decode.jsonl                                          (optional: one JSON line per tick with the decoder's choice)
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_on_rails.aux_heads import TARGET_SPEEDS

LAT_TARGETS = (-3.5, -1.75, 0.0, 1.75, 3.5)          # = counterfactual_q.QConfig.lat_targets
_ENV_DEFAULTS = {"beta": 1.0, "lat_margin": 1.0, "stay_margin": 0.25, "lateral": 1.0, "hold_ticks": 20, "confirm_ticks": 5, "lat_rate": 1.75}
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


SPEED_TOL = 0.1                                       # Q units: in a lane change, speed bins this close to their column's best count as best


def decode_with_q(probs: np.ndarray, q: np.ndarray, bins: np.ndarray, beta: float, lat_margin: float = 1.0, lateral: bool = True,
                  forced_d: Optional[float] = None, current_d: float = 0.0, stay_margin: float = 0.25) -> Tuple[float, float, Dict]:
    """(target speed in m/s, lateral offset in m, debug). The offset is chosen first from the column values, relative to the offset being driven
    (`current_d`), or forced (held from an earlier tick): a side offset needs `lat_margin` over the current one, a shifted car stays only while
    its offset beats the own lane by `stay_margin`, and a car on one side goes back to its lane before it may cross to the other. A Q with a
    non-finite value is not used: the plain decode. The speed is the mean of pi(v | offset) in
    lane and, in a lane change, the bin closest to BC's speed among those within SPEED_TOL of the column's best."""
    probs = np.asarray(probs, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if beta == 0.0 or not np.isfinite(q).all():
        return float((probs * bins).sum()), 0.0, {"p_v": probs, "p_lat": None, "v_lat": None, "candidate": 0.0, "nonfinite": beta != 0.0}
    lat = np.asarray(LAT_TARGETS)
    logits = np.log(np.clip(probs, 1e-12, None))[:, None] + beta * (q - np.max(q))
    joint = np.exp(logits - logits.max())
    joint /= joint.sum()
    v_lat = q.max(0)
    centre = int(np.argmin(np.abs(lat)))
    cur = int(np.argmin(np.abs(lat - current_d))) if lateral else centre
    j = int(np.argmax(v_lat))                           # exact ties go to the first (leftmost) offset
    if not lateral or (cur != centre and v_lat[cur] < v_lat[centre] + stay_margin):
        candidate = 0.0                                 # a shifted car whose offset no longer clearly beats its lane goes back
    elif j == cur or (j != centre and v_lat[j] - v_lat[cur] < lat_margin):
        candidate = float(lat[cur])                     # keep driving the current offset
    elif j == centre or (cur != centre and np.sign(lat[j]) != np.sign(lat[cur])):
        candidate = 0.0                                 # back to the own lane (also the way across to the other side)
    else:
        candidate = float(lat[j])
    d = candidate if forced_d is None else float(forced_d)
    k = int(np.argmin(np.abs(lat - d)))
    if k == centre:
        col = joint[:, k]
        p_v = col / max(col.sum(), 1e-300)
    else:                                               # BC's posterior is for its own in-lane plan (stop behind the parked car), and a softmax
        near = np.flatnonzero(q[:, k] >= q[:, k].max() - SPEED_TOL)     # over Q ties drives the fastest unreachable bins: of the near-best
        p_v = np.zeros(len(bins))                                         # bins, the one closest to BC's speed (BC itself where Q is flat)
        p_v[int(near[np.argmin(np.abs(bins[near] - float((probs * bins).sum())))])] = 1.0
    return float((p_v * bins).sum()), d, {"p_v": p_v, "p_lat": joint.sum(0), "v_lat": v_lat, "candidate": candidate}


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
    lock = getattr(policy, "_q_lock", None) or {"d": 0.0, "left": 0, "pend": 0.0, "n": 0}
    _, cand, _ = decode_with_q(probs, q, policy.target_speed_head.bins.cpu().numpy(), cfg["beta"], cfg["lat_margin"], bool(cfg["lateral"]),
                               current_d=lock["d"], stay_margin=cfg["stay_margin"])
    lock["pend"], lock["n"] = cand, (lock["n"] + 1 if cand == lock["pend"] else 1)
    if lock["left"] > 0:
        lock["left"] -= 1                                       # hold the earlier choice
    elif cand != lock["d"] and lock["n"] >= int(cfg["confirm_ticks"]):
        lock.update(d=cand, left=int(cfg["hold_ticks"]) - 1)    # a new choice seen confirm_ticks ticks in a row, held for hold_ticks ticks
    policy._q_lock = lock
    v, d, dbg = decode_with_q(probs, q, policy.target_speed_head.bins.cpu().numpy(), cfg["beta"], cfg["lat_margin"], bool(cfg["lateral"]),
                              forced_d=lock["d"])
    policy._q_choice = {"q_sa": q, "q_lat": d, "p_lat": dbg["p_lat"], "v_lat": dbg["v_lat"], "q_cand": cand, "q_v": v, "lat_rate": cfg["lat_rate"]}
    if os.environ.get("WOR_QCRITIC_LOG"):                       # per-tick decode record (JSON lines) for diagnosing a closed-loop route
        policy._q_tick = getattr(policy, "_q_tick", -1) + 1
        with open(os.environ["WOR_QCRITIC_LOG"], "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "tick": policy._q_tick, "q_lat": d, "q_cand": cand, "lock_left": lock["left"], "q_v": round(v, 3),
                                "bc_v": round(float((probs * policy.target_speed_head.bins.cpu().numpy()).sum()), 3),
                                "v_lat": np.round(dbg["v_lat"], 3).tolist(), "q_sa": np.round(q, 3).tolist()}) + "\n")
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
