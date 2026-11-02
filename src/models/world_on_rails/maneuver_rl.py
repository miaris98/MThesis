"""TODO A74 (2026-11-02): closed-loop RL over a maneuver head on top of E15.

E15 is behaviour cloning. A71 / A72 / A73 learned values from the logs (counterfactual labels on rails, IQL, a latent world model); A71's decode
did not help the closed loop (-0.3 DS on obstacle routes, -18 on the others, 2026-11-02) and A73's dynamics failed its gate. A74 learns
*when to deviate from E15* from the closed-loop reward itself, at Atari-100k scale instead of CaRL's 300 M steps from scratch:

* actions: 9 maneuvers = lateral target {one lane left, stay, one lane right} x speed {E15's own, creep, stop}. The lateral part re-anchors E15's
  waypoints to the chosen offset from the route (A71's `shift_toward_lateral`); the speed part can only slow E15 down, never speed it up, so a
  red light E15 stops for stays a stop.
* policy: `ManeuverHead`, a small MLP on the frozen trunk's policy token (`_last_policy_repr`), zero-initialised with a bias that picks the
  default maneuver (= E15 bit for bit) with p > 0.99, and a value output for PPO.
* rollouts: `ManeuverController` decides every `period` ticks (0.25 s); a non-default choice is held `commit` ticks (a lane change takes ~2 s,
  a 0.25 s nudge would never complete one); exploration mixes in non-default maneuvers with probability eps, raised to eps_stuck after the car
  has stood still for stuck_s (the obstacle case: E15 waits behind a parked car until the route times out). The logged log-prob is the
  behaviour mixture's, so PPO's importance ratio stays correct.
* reward (`RouteTracker`, `RewardTracker`, filled in by the agent wrapper each tick): route progress in percent of the route times the product of
  the leaderboard's collision factors so far, minus an immediate penalty per collision event and a small cost per tick off the lane.

WOR_RL="head=/workspace/a74/head_latest.pth,mode=train,period=5,commit=40,eps=0.01,eps_stuck=0.1,stuck_s=4,margin=7,log=/workspace/a74/rollouts"
(unset = the plain decode, bit-identical; mode=eval acts greedily and logs nothing unless log= is given). Learner: src/training/maneuver_ppo.py.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.world_on_rails.q_critic import shift_toward_lateral

LAT_CHOICES = (-3.5, 0.0, 3.5)                      # target offset from the route, m (+ = right, q_shift_lateral's convention); one lane = 3.5 m
SPEED_MODES = ("bc", "creep", "stop")
MANEUVERS: List[Tuple[float, str]] = [(lat, spd) for lat in LAT_CHOICES for spd in SPEED_MODES]
DEFAULT_ACTION = MANEUVERS.index((0.0, "bc"))
CREEP_KMH = 5.4                                     # 1.5 m/s: close up to an obstacle or edge past it
LAT_RATE = 1.75                                     # m/s, the counterfactual labels' lateral rate (q_critic._ENV_DEFAULTS)
COLLISION_FACTORS = {"walker": 0.5, "vehicle": 0.6, "layout": 0.65}   # the leaderboard's penalty factors (pedestrian, vehicle, static)


def apply_maneuver(action: int, wps, route, target_kmh: Optional[float], lat_rate: float = LAT_RATE):
    """(waypoints, target km/h) after a maneuver. The default maneuver returns its inputs untouched (the same objects)."""
    lat, mode = MANEUVERS[int(action)]
    if lat != 0.0 and route is not None:
        r = np.asarray(route.detach().cpu() if isinstance(route, torch.Tensor) else route, dtype=np.float64).reshape(-1, 2)
        wps = shift_toward_lateral(wps, r, lat, lat_rate)
    if mode == "stop":
        target_kmh = 0.0
    elif mode == "creep":
        target_kmh = CREEP_KMH if target_kmh is None else min(float(target_kmh), CREEP_KMH)
    return wps, target_kmh


class ManeuverHead(nn.Module):
    """Policy logits over MANEUVERS and a state value, from the trunk's policy token (detached: the trunk and E15's heads stay frozen)."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.pi = nn.Linear(hidden, len(MANEUVERS))
        self.v = nn.Linear(hidden, 1)
        self.margin = 0.0
        self.meta: Dict = {}                                # {"iter": k} of the learner update that wrote it (episodes record it)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x.float())
        return self.pi(h), self.v(h).squeeze(-1)

    def reset_to_default(self, margin: float = 7.0) -> "ManeuverHead":
        """Zero output layers + a bias on the default maneuver: the same distribution for every state, p(E15) = e^m / (e^m + 8)."""
        with torch.no_grad():
            self.pi.weight.zero_()
            self.pi.bias.zero_()
            self.pi.bias[DEFAULT_ACTION] = margin
            self.v.weight.zero_()
            self.v.bias.zero_()
        self.margin = float(margin)
        return self


def save_head(head: ManeuverHead, path: str, meta: Optional[Dict] = None) -> None:
    """Atomic: the agents reload the head at every route start, so they must never read a half-written file."""
    tmp = path + ".tmp"
    torch.save({"state_dict": head.state_dict(), "in_dim": head.in_dim, "margin": head.margin, "meta": meta or {}}, tmp)
    os.replace(tmp, path)


def load_head(path: Optional[str], in_dim: int, margin: float = 7.0) -> ManeuverHead:
    """The saved head when `path` exists, else a fresh one at E15 (the first route of a run, before the learner has written anything)."""
    head = ManeuverHead(in_dim).reset_to_default(margin)
    if path and os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if int(ck["in_dim"]) != in_dim:
            raise RuntimeError(f"WOR_RL head {path}: built on a {ck['in_dim']}-dim token, this policy's is {in_dim}")
        head.load_state_dict(ck["state_dict"])
        head.margin = float(ck.get("margin", margin))
        head.meta = dict(ck.get("meta") or {})
    return head.eval()


class ManeuverController:
    """One route's RL state: picks a maneuver every `period` ticks, holds a non-default one `commit` ticks, records per decision (token,
    action, behaviour log-prob, value, reward, dt in decision units) and writes the episode with `finish`."""

    def __init__(self, head: ManeuverHead, mode: str = "train", period: int = 5, commit: int = 40, eps: float = 0.01,
                 eps_stuck: float = 0.1, stuck_s: float = 4.0, tick_s: float = 0.05, seed: Optional[int] = None):
        if mode not in ("train", "eval"):
            raise ValueError(f"WOR_RL mode must be train or eval, got {mode!r}")
        self.head = head.eval()
        self.mode, self.period, self.commit = mode, int(period), int(commit)
        self.eps, self.eps_stuck, self.stuck_ticks = float(eps), float(eps_stuck), int(round(stuck_s / tick_s))
        self.rng = np.random.default_rng(seed)
        self.decisions: List[Dict] = []
        self._action, self._left, self._held, self._still = DEFAULT_ACTION, 0, 0, 0

    @torch.no_grad()
    def step(self, feat: torch.Tensor, speed_mps: float) -> int:
        """The maneuver for this tick (call once per tick, before the controller)."""
        self._still = self._still + 1 if speed_mps < 0.1 else 0
        if self.decisions and self._left > 0:
            self._left -= 1
            self._held += 1
            return self._action
        if self.decisions:
            self.decisions[-1]["dt"] = self._held / self.period
        dev = next(self.head.parameters()).device
        logits, v = self.head(feat.reshape(1, -1).float().to(dev))
        p = logits.softmax(-1)[0].double().cpu().numpy()
        if self.mode == "eval":
            a = int(p.argmax())
            logp = float(np.log(max(p[a], 1e-12)))
        else:
            eps = self.eps_stuck if self._still >= self.stuck_ticks else self.eps
            u = np.full(len(p), eps / (len(p) - 1))
            u[DEFAULT_ACTION] = 0.0                                  # exploration only ever proposes a deviation from E15
            mix = (1.0 - eps) * p + u
            mix /= mix.sum()
            a = int(self.rng.choice(len(mix), p=mix))
            logp = float(np.log(max(mix[a], 1e-12)))
        hold = self.commit if a != DEFAULT_ACTION else self.period
        self._action, self._left, self._held = a, hold - 1, 1
        self.decisions.append({"feat": feat.reshape(-1).float().cpu().numpy().astype(np.float16), "action": a, "logp": logp,
                               "value": float(v[0]), "reward": 0.0, "dt": hold / self.period})
        return a

    def add_reward(self, r: float) -> None:
        if self.decisions:
            self.decisions[-1]["reward"] += float(r)

    def finish(self, path: Optional[str], info: Optional[Dict] = None) -> Dict:
        """Write the episode (atomic, for the learner's directory watch) and return its summary."""
        if self.decisions:
            self.decisions[-1]["dt"] = self._held / self.period
        info = dict(info or {})
        acts = [d["action"] for d in self.decisions]
        info.update({"decisions": len(self.decisions), "return": float(sum(d["reward"] for d in self.decisions)),
                     "deviations": int(sum(a != DEFAULT_ACTION for a in acts)), "mode": self.mode,
                     "head_iter": int(self.head.meta.get("iter", 0))})
        if path and self.decisions:
            arrays = {"feat": np.stack([d["feat"] for d in self.decisions]), "action": np.array(acts, np.int64),
                      "logp": np.array([d["logp"] for d in self.decisions], np.float32),
                      "value": np.array([d["value"] for d in self.decisions], np.float32),
                      "reward": np.array([d["reward"] for d in self.decisions], np.float32),
                      "dt": np.array([d["dt"] for d in self.decisions], np.float32), "info": np.array(json.dumps(info))}
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, path)
        return info


def parse_rl_env(spec: Optional[str]) -> Optional[Dict]:
    """WOR_RL="head=...,mode=train,period=5,..." -> dict; unset / "" / "0" -> None. A ';' stands for ',' (queue_A9c.sh CKE job flags)."""
    if not spec or spec.strip() in ("", "0"):
        return None
    cfg = {"head": None, "mode": "train", "period": 5, "commit": 40, "eps": 0.01, "eps_stuck": 0.1, "stuck_s": 4.0, "margin": 7.0, "log": None}
    for part in spec.replace(";", ",").split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in cfg:
            raise ValueError(f"WOR_RL: unknown key {k!r} (known: {sorted(cfg)})")
        cfg[k] = type(cfg[k])(v) if isinstance(cfg[k], (int, float)) and not isinstance(cfg[k], bool) else v.strip()
    return cfg


def controller_from_env(embed_dim: int, spec: Optional[str] = None, seed: Optional[int] = None) -> Optional[Tuple[ManeuverController, Dict]]:
    cfg = parse_rl_env(os.environ.get("WOR_RL") if spec is None else spec)
    if cfg is None:
        return None
    head = load_head(cfg["head"], embed_dim, cfg["margin"])
    ctl = ManeuverController(head, mode=cfg["mode"], period=cfg["period"], commit=cfg["commit"], eps=cfg["eps"],
                             eps_stuck=cfg["eps_stuck"], stuck_s=cfg["stuck_s"], seed=seed)
    return ctl, cfg


def rl_maneuver(policy, wps, route, target_speed_kmh: Optional[float], speed_mps: float):
    """act()'s hook before the controller: a no-op unless an RL controller is attached (`policy._rl`, set by the agent wrapper per route)."""
    ctl = getattr(policy, "_rl", None)
    if ctl is None:
        return wps, target_speed_kmh
    a = ctl.step(getattr(policy, "_last_policy_repr"), speed_mps)
    return apply_maneuver(a, wps, route, target_speed_kmh)


class RLEpisode:
    """The agent wrapper's per-route glue, kept free of CARLA so it can be tested: each tick, collisions and the ego's world position become the
    reward of the transition the previous tick's maneuver caused (call tick() before act() picks this tick's maneuver); finish() writes the
    episode into cfg["log"] (when set) under the route's save name."""

    def __init__(self, ctl: ManeuverController, cfg: Dict, route_xy, route_tag: str):
        self.ctl, self.cfg, self.route_tag = ctl, cfg, route_tag
        self.track = RouteTracker(route_xy)
        self.reward = RewardTracker(self.track.length)
        self.ticks = 0

    def tick(self, x: float, y: float, t: float, collisions=()) -> float:
        for type_id, other_id in collisions:
            kind = "vehicle" if type_id.startswith("vehicle") else "walker" if type_id.startswith("walker") else "layout"
            self.reward.collision(kind, int(other_id), t)
        prog, off = self.track.update(x, y)
        r = self.reward.step(prog, off, t)
        self.ctl.add_reward(r)
        self.ticks += 1
        return r

    def finish(self, extra: Optional[Dict] = None) -> Dict:
        info = {"route": self.route_tag, "route_length": self.track.length, "progress_m": self.track.s_max,
                "rc": 100.0 * self.track.s_max / max(self.track.length, 1e-6), "collisions": dict(self.reward.events),
                "multiplier": self.reward.multiplier, "ticks": self.ticks, "head": self.cfg.get("head"), **(extra or {})}
        path = None
        if self.cfg.get("log"):
            os.makedirs(self.cfg["log"], exist_ok=True)
            path = os.path.join(self.cfg["log"], f"{self.route_tag}_{os.getpid()}_{int(time.time() * 1000)}.npz")
        return self.ctl.finish(path, info)


class RouteTracker:
    """Progress along a route in world coordinates: the arc length of the ego's projection onto the route, searched only in a window around
    the furthest point reached (a route that passes near itself cannot jump ahead) and credited only within max_dist of the route (the
    leaderboard's route completion has the same rule). update() also returns the signed offset from the route (+ = right of travel)."""

    def __init__(self, points_xy, window_back: float = 5.0, window_ahead: float = 40.0, max_dist: float = 10.0):
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-6] if len(pts) > 1 else np.ones(len(pts), bool)
        self.pts = pts[keep]
        seg = np.diff(self.pts, axis=0)
        self.seglen = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.zeros(0)
        self.cum = np.r_[0.0, np.cumsum(self.seglen)]
        self.length = float(self.cum[-1])
        self.window_back, self.window_ahead, self.max_dist = window_back, window_ahead, max_dist
        self.s_max = 0.0

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if len(self.seglen) == 0:
            return 0.0, 0.0
        lo = max(int(np.searchsorted(self.cum, self.s_max - self.window_back, side="right")) - 1, 0)
        hi = min(max(int(np.searchsorted(self.cum, self.s_max + self.window_ahead, side="left")), lo + 1), len(self.seglen))
        a, b, L = self.pts[lo:hi], self.pts[lo + 1:hi + 1], self.seglen[lo:hi]
        ab = b - a
        t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / L ** 2, 0.0, 1.0)
        px, py = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
        dist = np.hypot(x - px, y - py)
        j = int(np.argmin(dist))
        d = float((ab[j, 0] * (y - a[j, 1]) - ab[j, 1] * (x - a[j, 0])) / L[j])
        if dist[j] > self.max_dist:
            return 0.0, d
        s = float(self.cum[lo + j] + t[j] * L[j])
        prog = max(0.0, s - self.s_max)
        self.s_max = max(self.s_max, s)
        return prog, d


class RewardTracker:
    """Per-tick reward: route progress in percent of the route (an episode's progress terms sum to its route completion) times the product of
    the leaderboard's collision factors so far, minus an immediate penalty per collision event (one per contact with an actor, debounced),
    minus a small cost per tick farther than offlane_m from the route."""

    def __init__(self, route_length: float, collision_penalty: float = 10.0, offlane_cost: float = 0.02, offlane_m: float = 2.0,
                 debounce_s: float = 2.0, progress_scale: float = 100.0):
        self.route_length = max(float(route_length), 1.0)
        self.collision_penalty, self.offlane_cost, self.offlane_m = collision_penalty, offlane_cost, offlane_m
        self.debounce_s, self.progress_scale = debounce_s, progress_scale
        self.multiplier = 1.0
        self.events = {k: 0 for k in COLLISION_FACTORS}
        self._pending = 0.0
        self._last_contact: Dict[int, float] = {}

    def collision(self, kind: str, other_id: int, t: float) -> bool:
        kind = kind if kind in COLLISION_FACTORS else "layout"
        last = self._last_contact.get(other_id)
        self._last_contact[other_id] = t
        if last is not None and t - last < self.debounce_s:
            return False
        self.events[kind] += 1
        self.multiplier *= COLLISION_FACTORS[kind]
        self._pending += self.collision_penalty
        return True

    def step(self, progress_m: float, offset_m: float, t: float) -> float:
        r = self.progress_scale * progress_m / self.route_length * self.multiplier - self._pending
        self._pending = 0.0
        if abs(offset_m) > self.offlane_m:
            r -= self.offlane_cost
        return r
