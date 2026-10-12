"""CARLA-Leaderboard-style driving metrics, computed from simulator events.

Deliberately contains no `import carla`. Every quantity here is derived from plain
scalars and string actor ids that the caller extracts from the simulator, which means
the whole scoring rule - the part that is easy to get subtly wrong and impossible to
eyeball in a 30-minute rollout - is unit-testable on a machine with no GPU and no
simulator. `eval_wor_closed_loop.py` is the thin CARLA-facing half that feeds it.

WHY THESE METRICS AND NOT SOMETHING SIMPLER
-------------------------------------------
`eval_wor.py` already counts raw collision-sensor callbacks. That number is not a
metric, for two reasons that both bite hard:

  * The collision sensor fires **once per frame of sustained contact**. A car that
    scrapes a wall for two seconds at 20 FPS logs 40 "collisions". Any score built as a
    product of per-collision penalties then underflows to zero (0.6^40 ~= 1e-9) and one
    scrape becomes indistinguishable from total failure. Debouncing per actor is not a
    refinement here, it is what makes the number mean anything at all.
  * A collision count alone cannot separate "drove the whole route and clipped a bollard"
    from "failed to leave the spawn point". Route completion is the other half, and the
    published metric multiplies them rather than reporting either alone.

So this implements the CARLA Leaderboard's Driving Score:

    driving_score = route_completion * infraction_penalty

with the standard coefficients. That choice is not aesthetic - World on Rails, the
baseline this thesis is measured against, published its numbers on this metric. A
locally-invented score would make the comparison unciteable.

WHAT IS DELIBERATELY NOT IMPLEMENTED
------------------------------------
Leaderboard 2.0 adds scenario-timeout, yield-to-emergency and minimum-speed penalties.
They are omitted: the PDM-Lite training data and the WoR baseline both predate them, and
a penalty the baseline was never scored against would bias the comparison in a direction
that is hard to argue about later. The five coefficients below are Leaderboard 1.0, which
is the version World on Rails reported.
"""
from typing import Dict, List, Optional
import math

# CARLA Leaderboard 1.0 infraction coefficients. Each occurrence multiplies the running
# penalty, so they compound: two vehicle collisions give 0.60 * 0.60 = 0.36.
PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_RED_LIGHT = 0.70
PENALTY_STOP_SIGN = 0.80

# A collision with the same actor counts again only once contact has actually CEASED for
# this long. The sensor fires every frame (0.05 s apart at 20 FPS) throughout a contact,
# so any gap longer than this means the vehicles genuinely separated and re-hit.
#
# This is a gap in the *event stream*, not a lockout since the last counted event, and
# the difference is not academic. A fixed 5 s lockout was tried first: a car wedged
# against a stationary van logged a fresh "collision" every 5.0 s for as long as it stayed
# wedged - six infractions for one continuous contact, penalty 0.6^6 = 0.047, and a route
# the policy otherwise drove to 96% completion scored 0.026. The artifact dominated the
# metric completely. Requiring the contact to end first turns that back into one event.
DEFAULT_CONTACT_GAP_S = 1.0

# Below this speed the vehicle counts as not moving, for the stuck detector.
DEFAULT_BLOCKED_SPEED_MPS = 0.1
# The leaderboard allows 90 s before declaring a route blocked. Kept as the default so
# runs are comparable to published numbers, but short smoke-test evaluations will want
# to lower it - a 600-step rollout is only 30 s and can never trigger the real threshold.
DEFAULT_BLOCKED_TIMEOUT_S = 90.0


class TerminationReason:
    """Why a route ended. Recorded per route because the aggregate is not interpretable
    without it: a mean route completion of 0.4 means something very different when the
    routes ended `COMPLETED` than when they ended `BLOCKED`."""
    COMPLETED = "completed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    ROUTE_DEVIATION = "route_deviation"
    ERROR = "error"


def classify_collision(actor_type_id: str) -> str:
    """Maps a CARLA actor blueprint id to the leaderboard's three collision classes.

    CARLA type ids are dotted blueprint paths - `vehicle.tesla.model3`,
    `walker.pedestrian.0001`, `static.prop.streetbarrier`, `traffic.speed_limit.30`.
    Anything that is not a vehicle or a walker is 'static', which matches the
    leaderboard's own bucketing: it penalises collisions with layout (poles, barriers,
    buildings) identically regardless of which prop was hit.
    """
    t = (actor_type_id or "").lower()
    if t.startswith("walker"):
        return "pedestrian"
    if t.startswith("vehicle"):
        return "vehicle"
    return "static"


_COLLISION_PENALTY = {
    "pedestrian": PENALTY_COLLISION_PEDESTRIAN,
    "vehicle": PENALTY_COLLISION_VEHICLE,
    "static": PENALTY_COLLISION_STATIC,
}


class DrivingMetrics:
    """Accumulates one route's events and computes its Driving Score.

    The caller drives this with `tick()` once per simulator step plus an `add_*` call
    per event. All state is kept here rather than in the rollout loop so that the
    rollout loop cannot accidentally define the metric differently between two runs -
    the same reasoning that put the training and validation objective in one function
    in `src/training/wor_eval.py`.
    """

    def __init__(
        self,
        route_id: str,
        total_route_points: int,
        route_length_m: float = 0.0,
        contact_gap_s: float = DEFAULT_CONTACT_GAP_S,
        blocked_speed_mps: float = DEFAULT_BLOCKED_SPEED_MPS,
        blocked_timeout_s: float = DEFAULT_BLOCKED_TIMEOUT_S,
    ):
        self.route_id = route_id
        # Guard against a 1-point route making the completion fraction divide by zero.
        self.total_route_points = max(int(total_route_points), 1)
        self.route_length_m = float(route_length_m)
        self.contact_gap_s = float(contact_gap_s)
        self.blocked_speed_mps = float(blocked_speed_mps)
        self.blocked_timeout_s = float(blocked_timeout_s)

        # Progress. Kept as the MAX index reached, never the current one: a vehicle that
        # overshoots and reverses, or that oscillates at a junction, would otherwise have
        # its completion drop back down and end up scored on where it happened to stop
        # rather than how far it got.
        self._max_route_idx = 0

        self._infractions: List[Dict] = []
        self._collision_counts = {"pedestrian": 0, "vehicle": 0, "static": 0}
        self._red_light_violations = 0
        self._stop_sign_violations = 0

        # actor_id -> time of the last collision callback SEEN for it, counted or not.
        # Updated on every event, which is what lets an ongoing contact be recognised as
        # ongoing rather than re-counted whenever a timer lapses.
        self._last_contact_t: Dict[int, float] = {}

        # Off-road accounting is distance-weighted rather than tick-weighted. A vehicle
        # stopped off the drivable lane would otherwise accumulate off-road "share"
        # forever while travelling nowhere, which reads as a worse route than one that
        # drove the same detour at speed.
        self._distance_m = 0.0
        self._distance_offroad_m = 0.0

        self._t_s = 0.0
        self._last_t_s: Optional[float] = None
        self._stopped_since: Optional[float] = None
        self._terminated: Optional[str] = None

    # ---------------------------------------------------------------- per-step update

    def tick(self, t_s: float, speed_mps: float, route_idx: int, off_road: bool = False) -> None:
        """Advances one simulator step.

        `t_s` is simulation time, not wall-clock - the run is synchronous and a laggy
        host must not be able to trip the blocked-timeout that a fast host would not.
        """
        dt = 0.0 if self._last_t_s is None else max(0.0, t_s - self._last_t_s)
        self._t_s = t_s
        self._last_t_s = t_s

        step_m = max(0.0, float(speed_mps)) * dt
        self._distance_m += step_m
        if off_road:
            self._distance_offroad_m += step_m

        if route_idx > self._max_route_idx:
            self._max_route_idx = int(route_idx)

        # Stuck detection is a *continuous* stop, so any motion resets the clock.
        if float(speed_mps) < self.blocked_speed_mps:
            if self._stopped_since is None:
                self._stopped_since = t_s
        else:
            self._stopped_since = None

    @property
    def stopped_duration_s(self) -> float:
        if self._stopped_since is None:
            return 0.0
        return max(0.0, self._t_s - self._stopped_since)

    def is_blocked(self) -> bool:
        """True once the vehicle has been stationary past the timeout. The caller ends
        the route on this; it is not scored as an infraction, it simply stops progress
        accruing, which the route-completion term already reflects."""
        return self.stopped_duration_s >= self.blocked_timeout_s

    # ------------------------------------------------------------------------- events

    def add_collision(self, t_s: float, actor_type_id: str, actor_id: int = -1) -> bool:
        """Registers a collision, returning whether it counted as a new infraction.

        Returns False while a contact is still ongoing, so the caller can log the
        distinction rather than silently dropping events it may later want to audit.
        """
        last_t = self._last_contact_t.get(actor_id)
        still_in_contact = last_t is not None and (t_s - last_t) <= self.contact_gap_s
        # Recorded even when the event does not count, so a contact that persists keeps
        # renewing itself instead of ageing out mid-scrape.
        self._last_contact_t[actor_id] = t_s
        if still_in_contact:
            return False
        kind = classify_collision(actor_type_id)
        self._collision_counts[kind] += 1
        self._infractions.append({
            "type": f"collision_{kind}",
            "t_s": float(t_s),
            "actor_type_id": actor_type_id,
            "route_idx": self._max_route_idx,
        })
        return True

    def add_red_light_violation(self, t_s: float, light_id: int = -1) -> None:
        self._red_light_violations += 1
        self._infractions.append({
            "type": "red_light", "t_s": float(t_s),
            "actor_type_id": f"traffic.traffic_light:{light_id}",
            "route_idx": self._max_route_idx,
        })

    def add_stop_sign_violation(self, t_s: float, sign_id: int = -1) -> None:
        self._stop_sign_violations += 1
        self._infractions.append({
            "type": "stop_sign", "t_s": float(t_s),
            "actor_type_id": f"traffic.stop:{sign_id}",
            "route_idx": self._max_route_idx,
        })

    def terminate(self, reason: str) -> None:
        self._terminated = reason

    # ------------------------------------------------------------------------ scoring

    @property
    def offroad_fraction(self) -> float:
        if self._distance_m <= 0.0:
            return 0.0
        return min(1.0, self._distance_offroad_m / self._distance_m)

    @property
    def raw_route_completion(self) -> float:
        """Fraction of the planned route reached, before the off-road reduction.

        The global route is sampled at a uniform spacing, so an index fraction is a
        distance fraction; that equivalence is why this can be an index ratio at all and
        it breaks if the route is ever resampled non-uniformly.
        """
        return min(1.0, self._max_route_idx / max(1, self.total_route_points - 1))

    @property
    def route_completion(self) -> float:
        """Route completion with the off-road share removed, as the leaderboard scores
        it: distance covered outside a drivable lane does not count as route progress."""
        return self.raw_route_completion * (1.0 - self.offroad_fraction)

    @property
    def infraction_penalty(self) -> float:
        """Product of every incurred coefficient. Starts at 1.0 (a clean route)."""
        p = 1.0
        for kind, count in self._collision_counts.items():
            p *= _COLLISION_PENALTY[kind] ** count
        p *= PENALTY_RED_LIGHT ** self._red_light_violations
        p *= PENALTY_STOP_SIGN ** self._stop_sign_violations
        return p

    @property
    def driving_score(self) -> float:
        return self.route_completion * self.infraction_penalty

    @property
    def termination_reason(self) -> str:
        if self._terminated is not None:
            return self._terminated
        if self.is_blocked():
            return TerminationReason.BLOCKED
        if self.raw_route_completion >= 1.0:
            return TerminationReason.COMPLETED
        return TerminationReason.TIMEOUT

    def to_dict(self) -> Dict:
        """One route's record. This is the unit the paired bootstrap resamples, so it
        carries everything needed to recompute the score without the simulator."""
        return {
            "route_id": self.route_id,
            "driving_score": round(self.driving_score, 6),
            "route_completion": round(self.route_completion, 6),
            "raw_route_completion": round(self.raw_route_completion, 6),
            "infraction_penalty": round(self.infraction_penalty, 6),
            "offroad_fraction": round(self.offroad_fraction, 6),
            "termination_reason": self.termination_reason,
            "collisions_pedestrian": self._collision_counts["pedestrian"],
            "collisions_vehicle": self._collision_counts["vehicle"],
            "collisions_static": self._collision_counts["static"],
            "red_light_violations": self._red_light_violations,
            "stop_sign_violations": self._stop_sign_violations,
            "distance_driven_m": round(self._distance_m, 3),
            "duration_s": round(self._t_s, 3),
            "total_route_points": self.total_route_points,
            "route_length_m": round(self.route_length_m, 3),
            "infractions": self._infractions,
        }


def aggregate(records: List[Dict]) -> Dict:
    """Aggregates per-route records into the headline numbers.

    The leaderboard reports the *mean* driving score over routes, not the score of the
    concatenated routes, and infractions are reported per kilometre so that a long route
    and a short one contribute comparably. Both conventions are followed here; the
    per-km normalisation in particular matters because a policy that stalls immediately
    drives almost no distance and would otherwise post a flawless infraction rate.
    """
    if not records:
        return {"num_routes": 0}

    n = len(records)
    total_km = sum(r.get("distance_driven_m", 0.0) for r in records) / 1000.0

    def _mean(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in records) / n

    def _per_km(key: str) -> Optional[float]:
        if total_km <= 0.0:
            return None
        return sum(float(r.get(key, 0.0)) for r in records) / total_km

    reasons: Dict[str, int] = {}
    for r in records:
        reason = r.get("termination_reason", TerminationReason.ERROR)
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "num_routes": n,
        "driving_score": round(_mean("driving_score"), 6),
        "route_completion": round(_mean("route_completion"), 6),
        "infraction_penalty": round(_mean("infraction_penalty"), 6),
        "offroad_fraction": round(_mean("offroad_fraction"), 6),
        "total_km": round(total_km, 4),
        "collisions_pedestrian_per_km": _per_km("collisions_pedestrian"),
        "collisions_vehicle_per_km": _per_km("collisions_vehicle"),
        "collisions_static_per_km": _per_km("collisions_static"),
        "red_light_violations_per_km": _per_km("red_light_violations"),
        "termination_reasons": reasons,
    }
