"""World on Rails (WoR) reward function from Chen et al., ICCV 2021."""
from typing import Any, Dict, Tuple
from src.envs.rewards.base import BaseReward


class WorldOnRailsReward(BaseReward):
    """
    Scalar per-step approximation of the World on Rails reward (Chen et al., ICCV 2021).

    IMPORTANT, for the thesis writeup: upstream WoR has no scalar step reward. Its reward
    (Carla-utils/PCLA/pcla_agents/wor/rails/bellman.py::BellmanUpdater.get_reward) is a
    *spatial* reward field over a 96x96 BEV label map, evaluated across a
    (speed x orientation) state lattice and a discretized action lattice, then backed up
    by value iteration to produce action-value labels for supervised distillation. It is
    not usable as-is for online PPO, which needs a scalar r(s, a). This class reproduces
    the three properties of that field that actually shape behavior:

      1. Reward is peaked at a TARGET SPEED, not proportional to speed. Upstream builds the
         waypoint reward with `to_dense(..., target_speed=_tgt_speeds, ...)` (bellman.py:151),
         where `_lerp_grids` (bellman.py:229) spreads unit reward across the speed bins
         bracketing the target and `_batch_lerp` (bellman.py:382) reads it back at the ego's
         actual speed. The net r(v) is a tent peaked at the target, falling to zero on BOTH
         sides. Reward proportional to speed - what this file used to do - is the single
         change that creates a creep-forever optimum, since any nonzero speed pays.
      2. Being on the waypoint corridor GATES the reward multiplicatively rather than
         adding a penalty for being off it (the reward field is simply zero where the
         waypoint mask is zero). An additive lane penalty makes "stand still, perfectly
         centered" score ~0, which beats any attempt at driving that risks a penalty.
      3. Off-road / collision zero out all future value multiplicatively: `free` is
         `(road>0)&(vehicle==0)&(pedestrian==0)` (bellman.py:124) and `Q = Q * free`
         (bellman.py:420). Episode termination already implements that here; the additive
         penalties below are on top of it and stay bounded relative to a full episode's
         progress so they cannot dominate.

    Known divergence from upstream: WoR's progress is measured against the target rails /
    route waypoints. This environment has no route - DrivingStateExtractor projects the ego
    onto the nearest driving lane (src/envs/driving_state.py:82), so `heading_cos` is
    lane-relative, not route-relative. Progress here therefore means "moving along whatever
    lane you are on at the target speed", not "completing a route".
    """
    NAME = "wor"
    SOURCE = "Learning to drive from a world on rails (Chen et al., ICCV 2021)"

    # Peak progress reward per SECOND of simulated time, so episode return is independent
    # of --sim-dt. At the default dt=0.1 this is 0.4/step, matching the old formulation's
    # peak, so the terminal penalties below keep the same relative weight as before.
    PROGRESS_WEIGHT_PER_SEC = 4.0
    # Half-width of the speed tent, as a fraction of the target speed. Upstream's implied
    # half-width is one bin of its speed discretization, which is narrow (~0.3 of target).
    # DELIBERATE DIVERGENCE: widened to 1.0 here. WoR computes its reward offline under
    # value iteration, so a slow agent still sees high value backed up from reachable
    # fast states; online PPO only learns from reward it actually collects, and a narrow
    # tent pays ~0 everywhere the current policy operates, leaving no gradient to climb.
    # At 1.0 the tent rises monotonically from 0 at standstill to 1 at the target and
    # decays back to 0 at twice the target, so overspeed is still discouraged.
    SPEED_TOLERANCE_FRAC = 1.0
    # How much of the progress reward the lane-centering gate can remove at the lane edge.
    # 1.0 would fully zero it, matching the hard BEV mask; kept below 1 so the gradient
    # toward the lane center stays informative instead of flat.
    LANE_GATE_STRENGTH = 0.6

    COLLISION_PENALTY = -25.0       # Terminal collision cost
    RED_LIGHT_PENALTY = -20.0       # Red light infraction cost
    OFF_ROAD_PENALTY = -20.0        # Off-road boundary termination cost
    STALL_TERMINAL_PENALTY = -15.0  # Terminal stall cost

    STALL_TIMEOUT_STEPS = 80
    # Was 2.0, which the policy exploited directly: it learned to crawl at 2.1-2.5 km/h,
    # staying above the threshold so the stall counter reset every step, and ride the
    # episode to truncation. The threshold has to sit above any speed that counts as "not
    # actually driving", not just above literal standstill. `exempt` in _track_stall
    # already covers red lights and obstacles within 10 m, so legal stops never accumulate.
    STALL_SPEED_KMH = 8.0

    def __init__(self, desired_speed: float = 25.0, **kwargs: Any):
        super().__init__(desired_speed=desired_speed)
        self._light_latched = False

    def reset_episode_tracking(self) -> None:
        """Reset event latches for a new episode."""
        super().reset_episode_tracking()
        self._light_latched = False

    def _speed_tent(self, speed_kmh: float) -> float:
        """
        r(v): 1.0 at the target speed, decaying linearly to 0 at +/- the tolerance.

        This is the scalar equivalent of WoR's target-speed lerp over its speed bins. The
        property that matters is that it is peaked rather than monotone: reward
        proportional to speed pays on every step at any nonzero speed, which is what let
        the policy settle on crawling at 2.4 km/h to ride out the episode.
        """
        tolerance = max(1e-6, self.SPEED_TOLERANCE_FRAC * self.desired_speed)
        return max(0.0, 1.0 - abs(speed_kmh - self.desired_speed) / tolerance)

    def compute_reward(
        self,
        state: Dict[str, Any],
        curriculum_factor: float = 1.0,
        dt: float = 0.05
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the step reward matching the World on Rails Markov Decision Process.
        """
        speed_kmh = float(state.get("speed_kmh", 0.0))
        heading_cos = float(state.get("heading_cos", 1.0))
        lateral_dist = float(state.get("lateral_dist", 0.0))
        lane_width = float(state.get("lane_width", 3.5))
        is_at_red_light = bool(state.get("is_at_red_light", False))
        is_collision = bool(state.get("is_collision", False))
        is_off_road = bool(state.get("is_off_road", False))
        is_junction = bool(state.get("is_junction", False))
        brake = float(state.get("brake", 0.0))
        time_step = int(state.get("time_step", 0))

        # 1. Forward progress, peaked at the target speed and aligned with the lane heading.
        r_progress_ungated = (
            self.PROGRESS_WEIGHT_PER_SEC
            * self._speed_tent(speed_kmh)
            * max(0.0, heading_cos)
            * dt
        )

        # 2. Lane-centering gates progress multiplicatively (upstream: the reward field is
        # zero wherever the waypoint mask is zero). Exempt inside junctions, where there is
        # no meaningful lane center, and off-road, where the episode is ending anyway.
        lane_gate = 1.0
        if not is_junction and not is_off_road:
            half_width = max(1.0, lane_width / 2.0)
            norm_dist = min(1.0, abs(lateral_dist) / half_width)
            lane_gate = 1.0 - self.LANE_GATE_STRENGTH * (norm_dist ** 2)

        # Split the gated progress across the two telemetry slots so they still sum to the
        # total, the way every other reward in the registry reports: r_progress carries the
        # ungated value and r_lane carries what the gate removed, as a negative lane cost.
        r_progress = r_progress_ungated
        r_lane = r_progress_ungated * (lane_gate - 1.0)

        # Zero out progress entirely while actively running a red light.
        running_red = is_at_red_light and speed_kmh >= 2.0 and brake <= 0.2
        if running_red:
            r_progress = 0.0
            r_lane = 0.0

        # 3. Red Light Violation
        r_light = 0.0
        if running_red and not self._light_latched:
            self._light_latched = True
            r_light = self.RED_LIGHT_PENALTY

        # 4. Terminal Events (Collision, Off-Road, Stall)
        r_terminal = 0.0
        is_stalled = False
        if is_collision:
            r_terminal = self.COLLISION_PENALTY
        elif is_off_road:
            r_terminal = self.OFF_ROAD_PENALTY
        else:
            # Anti-stall check
            is_stalled = self._track_stall(
                speed_kmh=speed_kmh,
                time_step=time_step,
                exempt=(is_at_red_light or float(state.get("min_obs_dist", 99.0)) < 10.0),
                timeout_steps=self.STALL_TIMEOUT_STEPS,
                stall_speed_kmh=self.STALL_SPEED_KMH
            )
            if is_stalled:
                r_terminal = self.STALL_TERMINAL_PENALTY

        # Total composite reward
        raw_reward = r_progress + r_lane + r_light + r_terminal

        return raw_reward, self._blank_info(
            r_progress=r_progress,
            r_lane=r_lane,
            r_light=r_light,
            r_obstacle=0.0,
            r_ttc=0.0,
            r_terminal=r_terminal,
            is_stalled=is_stalled,
        )
