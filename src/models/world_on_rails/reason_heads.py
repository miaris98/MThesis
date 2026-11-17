"""TODO A84: heads that predict WHY the expert slows down and whether going on would crash, from the policy token; and the eval-time danger gate.

Labels come from PDM-Lite's logs (src/training/reason_labels.py, scripts/training/build_reason_labels.py); the network still sees only the
camera, the speed and the route. Outputs (all from the trunk's policy token, the token the target-speed head reads):

  reason_cause_logits  (B, 6)  none / vehicle / vulnerable / light / stop / static: the object binding the expert's speed
  reason_dist          (B,)    log(1 + metres) to that object
  reason_closing       (B,)    closing speed / 10 m/s
  reason_brake_logit   (B,)    the expert brakes hard within 1 s
  reason_danger_logit  (B,)    proceeding at max(speed, 3 m/s) for 2 s would hit someone (the crash classifier)

Two ways to use them, one arm each:
  * multi-task ("learn the reasons"): the loss flows into the trunk, so the token the speed head reads has to carry the cause;
    with `cascade` the speed head also reads the heads' outputs through a zero-initialised projection (an arm resumed from E15 starts
    exactly at E15 and learns the coupling);
  * `detach` ("a small classifier next to the policy"): the heads read a detached token, so the driving policy is bit-identical to its
    base and the heads are a probe / critic, used only by the danger gate at evaluation.

No import from src.training here: the CARLA eval venv cannot import that package's training dependencies (2026-11-02 lesson).
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REASON_CLASSES = ("none", "vehicle", "vulnerable", "light", "stop", "static")      # = src.training.reason_labels.REASON_CLASSES (tested)
CASCADE_DIM = len(REASON_CLASSES) + 4                                               # cause probs, log distance, closing, p(brake), p(danger)
# Per-term weights inside the reason loss; chosen so each term is O(1) at init (CE over 6 classes ~1.8, BCE ~0.7, L1 on log metres ~1).
TERM_WEIGHTS = {"cause": 1.0, "dist": 0.5, "closing": 0.5, "brake": 1.0, "danger": 1.0}


class ReasonHeads(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, detach: bool = False):
        super().__init__()
        self.detach = bool(detach)
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, len(REASON_CLASSES) + 4)

    def forward(self, token: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.body(token.detach() if self.detach else token)
        o = self.out(h).float()
        k = len(REASON_CLASSES)
        return {"reason_cause_logits": o[:, :k], "reason_dist": o[:, k], "reason_closing": o[:, k + 1],
                "reason_brake_logit": o[:, k + 2], "reason_danger_logit": o[:, k + 3]}


def cascade_features(r: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, CASCADE_DIM) summary of the heads' predictions that the speed head may read (cascade arm)."""
    return torch.cat([r["reason_cause_logits"].softmax(-1), r["reason_dist"][:, None], r["reason_closing"][:, None],
                      torch.sigmoid(r["reason_brake_logit"])[:, None], torch.sigmoid(r["reason_danger_logit"])[:, None]], dim=-1)


def zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    lin = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> Optional[torch.Tensor]:
    s = m.sum()
    return None if float(s) <= 0 else (x * m).sum() / s


def reason_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], device) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Weighted sum of the five terms over the frames that carry each label (-1 / NaN = no label). None when nothing is labelled."""
    if "reason_cause_logits" not in out or "reason_cause" not in batch:
        return None, {}
    cause = batch["reason_cause"].to(device).long()
    parts, total = {}, None
    valid = cause >= 0
    if bool(valid.any()):
        parts["cause"] = F.cross_entropy(out["reason_cause_logits"][valid], cause[valid])
    bind = batch["reason_dist_valid"].to(device).float()
    t = _masked_mean((out["reason_dist"] - batch["reason_dist"].to(device).float()).abs(), bind)
    if t is not None:
        parts["dist"] = t
    cv = batch["reason_closing_valid"].to(device).float()
    t = _masked_mean((out["reason_closing"] - batch["reason_closing"].to(device).float() / 10.0).abs(), cv)
    if t is not None:
        parts["closing"] = t
    for key in ("brake", "danger"):
        y = batch[f"reason_{key}"].to(device).float()
        m = (y >= 0).float()
        t = _masked_mean(F.binary_cross_entropy_with_logits(out[f"reason_{key}_logit"], y.clamp(min=0.0), reduction="none"), m)
        if t is not None:
            parts[key] = t
    for k, v in parts.items():
        total = TERM_WEIGHTS[k] * v if total is None else total + TERM_WEIGHTS[k] * v
    return total, {k: float(v.detach()) for k, v in parts.items()}


def reason_term(out, batch, weight: float, device):
    """The trainer's hook: weight x reason_loss, or None when the weight is 0 or the batch has no label."""
    if weight <= 0:
        return None, {}
    loss, parts = reason_loss(out, batch, device)
    return (None, parts) if loss is None else (weight * loss, parts)


def danger_gate_kmh(model, out: Dict[str, torch.Tensor], speed_mps: float, target_kmh):
    """act()'s hook: records p(danger) for the clip recorder and, when WOR_DANGER_GATE is set, lets the gate lower the target speed."""
    if target_kmh is None or "reason_danger_logit" not in out:
        return target_kmh
    if not getattr(model, "_danger_ready", False):
        model._danger_gate, model._danger_ready = DangerGate.from_env(), True
    p = float(torch.sigmoid(out["reason_danger_logit"][0].float()))
    model._last_danger_p = p
    if model._danger_gate is None:
        return target_kmh
    return 3.6 * model._danger_gate.update(p, speed_mps, float(target_kmh) / 3.6)


class DangerGate:
    """Eval-time use of the danger head (WOR_DANGER_GATE="theta=0.6,off=0.4,T=4,release=3,max_release=3"); unset = no gate.

    While p(danger) >= theta (and until it falls below `off`, hysteresis) the target speed is 0, so the PID brakes. A gate that never
    opens is WoR's stall (A50's reflex lost 11 DS that way): after T s of standing with the gate still closed it releases for `release`
    s to the policy's own decode; after `max_release` releases in one stop the gate stays open until the car has driven above 3 m/s
    again (a danger that never clears is more likely a false positive than a real one, and a timeout costs the rest of the route).
    """

    def __init__(self, theta: float = 0.6, off: float = 0.4, T: float = 4.0, release: float = 3.0, max_release: int = 3, dt: float = 0.05):
        self.theta, self.off, self.T, self.release, self.max_release, self.dt = theta, min(off, theta), T, release, int(max_release), dt
        self.closed, self.standing, self.release_left, self.n_release = False, 0.0, 0.0, 0
        self.events = {"closed_ticks": 0, "releases": 0}

    @classmethod
    def from_env(cls) -> Optional["DangerGate"]:
        spec = os.environ.get("WOR_DANGER_GATE")
        if not spec:
            return None
        kv = {"theta": 0.6, "off": 0.4, "T": 4.0, "release": 3.0, "max_release": 3.0}
        for part in spec.replace(";", ",").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = float(v)
        gate = cls(kv["theta"], kv["off"], kv["T"], kv["release"], int(kv["max_release"]), float(os.environ.get("WOR_TICK_DT", "0.05")))
        print(f"[A84] danger gate on: theta={gate.theta} off={gate.off} T={gate.T} release={gate.release} max_release={gate.max_release}", flush=True)
        return gate

    def update(self, p_danger: float, speed_mps: float, target_mps: float) -> float:
        if not math.isfinite(p_danger):
            return target_mps
        self.standing = self.standing + self.dt if speed_mps < 0.5 else 0.0
        if speed_mps > 3.0:
            self.n_release = 0                      # driving again: the stop is over, the gate re-arms with a full budget
        if self.release_left > 0:
            self.release_left -= self.dt
            return target_mps
        if self.n_release >= self.max_release:      # gave up for this stop
            self.closed = False
            return target_mps
        if p_danger >= self.theta:
            self.closed = True
        elif p_danger < self.off:
            self.closed = False
        if self.closed and self.standing >= self.T and self.n_release < self.max_release:
            self.release_left, self.standing, self.closed = self.release, 0.0, False
            self.n_release += 1
            self.events["releases"] += 1
            return target_mps
        if self.closed:
            self.events["closed_ticks"] += 1
            return 0.0
        return target_mps
