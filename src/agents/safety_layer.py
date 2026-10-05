"""TODO A50 / A57 / A52 (S-125): eval-time decision layer over the target-speed posterior. No retraining, no new input: the speedometer and the policy's own 8-bin speed posterior.

Switched on per eval by environment variables (all unset = the plain mean decode, bit-identical to every earlier run):

  WOR_NEWSVENDOR="q0=0.3,T=8,qmax=0.7"  A57: while driving read the low quantile q0 of the speed posterior (a collision costs x0.6, waiting almost nothing until the 200 s cap);
                                         the quantile rises linearly to qmax over T seconds of standing (speed <= 3 m/s) and after T the plain mean decode takes over
                                         (the cost of waiting accumulates; the literal q -> 1 limit would read the fastest bin with any mass).
  WOR_CREEP_CLAMP=1                      A50: at speed < 3 m/s with stop mass P(bin 0) >= 0.3 command 0 instead of the mean, so a bimodal stop/go posterior cannot average into a
                                         creep toward the obstacle; released by the stall breaker (standing > 4 s and stop mass < 0.9), re-armed once the car drives again.
  WOR_CONTACT_REFLEX=1                   A50: a contact signature (speed drop >= 2.5 m/s within 0.25 s while the commanded brake was < 0.5) holds a full brake for 3 s, then
                                         the clamp above takes over (move again only when the stop mass is < 0.9 after 4 s of standing).
  WOR_SPEED_SCALE=0.9                    A52: matched-speed control: the plain head's target speed times a constant.
  WOR_TICK_DT=0.05                       agent tick (the Leaderboard calls the agent every simulation tick, 20 Hz).

The layer only ever lowers or holds the commanded speed relative to the chosen decode except at the stall breaker, where it falls back to the mean.
"""
from __future__ import annotations

import os
from collections import deque
from typing import Optional, Tuple

import numpy as np

SLOW_MPS = 3.0
CONTACT_DROP_MPS = 2.5
CONTACT_WINDOW_S = 0.25
CONTACT_HOLD_S = 3.0
CLAMP_STOP_MASS = 0.3
BREAKER_STANDING_S = 4.0
BREAKER_STOP_MASS = 0.9


def quantile_speed(probs: np.ndarray, bins: np.ndarray, tau: float) -> float:
    """The slowest bin whose CDF reaches tau (the same rule as aux_heads.speed_quantile, not interpolated)."""
    cdf = np.cumsum(probs)
    idx = min(int((cdf < tau - 1e-6).sum()), len(bins) - 1)
    return float(bins[idx])


def _parse_kv(spec: str, defaults: dict) -> dict:
    out = dict(defaults)
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


class SafetyLayer:
    def __init__(self, newsvendor: Optional[dict] = None, creep_clamp: bool = False, contact_reflex: bool = False, speed_scale: float = 1.0, dt: float = 0.05):
        self.nv, self.clamp_on, self.reflex_on, self.scale, self.dt = newsvendor, creep_clamp or contact_reflex, contact_reflex, float(speed_scale), float(dt)
        self.t = 0.0
        self.standing = 0.0
        self.hist: deque = deque()      # (t, speed) of the last CONTACT_WINDOW_S
        self.hold_until = -1.0
        self.clamp_active = False
        self.released = False
        self.events = {"contacts": 0, "clamp_ticks": 0, "hold_ticks": 0}

    @classmethod
    def from_env(cls) -> Optional["SafetyLayer"]:
        nv = os.environ.get("WOR_NEWSVENDOR")
        clamp = os.environ.get("WOR_CREEP_CLAMP") == "1"
        reflex = os.environ.get("WOR_CONTACT_REFLEX") == "1"
        scale = float(os.environ.get("WOR_SPEED_SCALE", "1.0"))
        if not (nv or clamp or reflex or scale != 1.0):
            return None
        layer = cls(_parse_kv(nv, {"q0": 0.3, "T": 8.0, "qmax": 0.7}) if nv else None, clamp, reflex, scale, float(os.environ.get("WOR_TICK_DT", "0.05")))
        print(f"[A50/A57] safety layer: newsvendor={layer.nv} creep_clamp={clamp} contact_reflex={reflex} speed_scale={scale}", flush=True)
        return layer

    def update(self, speed_mps: float, probs: np.ndarray, bins: np.ndarray, mean_mps: float, last_brake: float) -> Tuple[float, bool]:
        """One tick -> (target speed in m/s, force a full brake)."""
        self.t += self.dt
        probs = np.asarray(probs, dtype=np.float64)
        stop_mass = float(probs[0])
        self.standing = 0.0 if speed_mps > SLOW_MPS else self.standing + self.dt
        if speed_mps > SLOW_MPS:
            self.released = False
        target = mean_mps
        if self.nv is not None and self.standing < self.nv["T"]:
            q = self.nv["q0"] + (self.nv["qmax"] - self.nv["q0"]) * min(1.0, self.standing / self.nv["T"])
            target = quantile_speed(probs, bins, q)
        force_brake = False
        if self.reflex_on:
            self.hist.append((self.t, speed_mps))
            while self.hist and self.t - self.hist[0][0] > CONTACT_WINDOW_S:
                self.hist.popleft()
            if self.t >= self.hold_until and last_brake < 0.5 and max(s for _, s in self.hist) - speed_mps >= CONTACT_DROP_MPS:
                self.hold_until = self.t + CONTACT_HOLD_S
                self.clamp_active = True
                self.released = False
                self.events["contacts"] += 1
                self.hist.clear()
            if self.t < self.hold_until:
                force_brake = True
                self.events["hold_ticks"] += 1
                return 0.0, True
        if self.clamp_on:
            if speed_mps < SLOW_MPS and stop_mass >= CLAMP_STOP_MASS and not self.released:
                self.clamp_active = True
            if self.clamp_active and self.standing > BREAKER_STANDING_S and stop_mass < BREAKER_STOP_MASS:
                self.clamp_active, self.released = False, True       # the stall breaker: the posterior is not sure enough to stand still
                target = mean_mps
            if self.clamp_active and speed_mps >= SLOW_MPS:
                self.clamp_active = False                           # the clamp only governs slow driving
            if self.clamp_active:
                self.events["clamp_ticks"] += 1
                return 0.0, force_brake
        return (target * self.scale if target > 0 else 0.0), force_brake
