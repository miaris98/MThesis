"""Driving state extraction: lane geometry, traffic lights, and obstacle proximity from CARLA."""
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    carla = None


class DrivingStateExtractor:
    """
    Builds the full perception state dictionary consumed by RewardCalculator.

    Uses a cached actor registry refreshed every N steps plus a single world snapshot per
    step, so obstacle scanning costs one RPC instead of one call per surrounding actor.
    """
    FRONT_CONE_LENGTH = 30.0
    FRONT_CONE_HALF_WIDTH = 3.0
    BUMPER_ALLOWANCE = 2.5
    ACTOR_REFRESH_INTERVAL = 20
    FAR_WAYPOINT_DISTANCE = 10.0
    CURVE_LOOKAHEAD_DISTANCE = 15.0
    MIN_CURVE_FACTOR = 0.4

    def __init__(self, world: Any = None, world_map: Any = None):
        self.world = world
        self.world_map = world_map
        self._actor_cache: List[Tuple[int, bool]] = []
        self._refresh_counter = 0

    def bind(self, world: Any, world_map: Any) -> None:
        """Attach the extractor to a (re)created CARLA world and drop stale actor handles."""
        self.world = world
        self.world_map = world_map
        self.reset()

    def reset(self) -> None:
        """Invalidate the cached actor registry for a fresh episode."""
        self._actor_cache = []
        self._refresh_counter = 0

    def _refresh_actor_cache(self, ego_id: int) -> None:
        """Rebuild the list of (actor_id, is_pedestrian) for vehicles and walkers except the ego."""
        if self.world is None:
            return
        cache: List[Tuple[int, bool]] = []
        for actor_filter, is_pedestrian in (("vehicle.*", False), ("walker.pedestrian.*", True)):
            try:
                for actor in self.world.get_actors().filter(actor_filter):
                    if actor.id != ego_id:
                        cache.append((actor.id, is_pedestrian))
            except Exception:
                pass
        self._actor_cache = cache

    def _lane_state(self, ego_tf: Any) -> Dict[str, Any]:
        """Project the ego onto the driving lane and derive heading, offset, and curvature."""
        lane = {
            "heading_cos": 1.0,
            "heading_cos_far": 1.0,
            "lateral_dist": 0.0,
            "curve_factor": 1.0,
            "is_junction": False,
            "off_road": False,
        }
        if self.world_map is None or ego_tf is None:
            return lane

        try:
            if carla is not None:
                wp_exact = self.world_map.get_waypoint(ego_tf.location, project_to_road=False, lane_type=carla.LaneType.Driving)
                if wp_exact is None:
                    lane["off_road"] = True

            wp = self.world_map.get_waypoint(ego_tf.location, project_to_road=True)
            if wp is None:
                return lane

            fwd = ego_tf.get_forward_vector()
            wp_fwd = wp.transform.get_forward_vector()
            wp_right = wp.transform.get_right_vector()

            heading_cos = float(max(-1.0, min(1.0, fwd.x * wp_fwd.x + fwd.y * wp_fwd.y)))
            dx = ego_tf.location.x - wp.transform.location.x
            dy = ego_tf.location.y - wp.transform.location.y
            lat_cross = abs(dx * wp_right.x + dy * wp_right.y)

            lane["heading_cos"] = heading_cos
            lane["lateral_dist"] = float(min(3.0, lat_cross))
            lane["is_junction"] = bool(wp.is_junction)

            if not wp.is_junction and lat_cross > ((wp.lane_width / 2.0) + 0.8):
                lane["off_road"] = True
            if not wp.is_junction and heading_cos < -0.2:
                lane["off_road"] = True

            far_wps = wp.next(self.FAR_WAYPOINT_DISTANCE)
            if far_wps:
                far_fwd = far_wps[0].transform.get_forward_vector()
                lane["heading_cos_far"] = float(max(-1.0, min(1.0, fwd.x * far_fwd.x + fwd.y * far_fwd.y)))

            curve_wps = wp.next(self.CURVE_LOOKAHEAD_DISTANCE)
            if curve_wps:
                curve_fwd = curve_wps[0].transform.get_forward_vector()
                cos_curve = wp_fwd.x * curve_fwd.x + wp_fwd.y * curve_fwd.y
                lane["curve_factor"] = float(max(self.MIN_CURVE_FACTOR, min(1.0, cos_curve)))
        except Exception:
            pass

        return lane

    def _is_at_red_light(self, ego: Any) -> bool:
        """Return True when the ego is inside the trigger box of a red traffic light."""
        if carla is None or ego is None:
            return False
        try:
            # CARLA returns Green when the vehicle is not affected by any traffic light.
            return ego.get_traffic_light_state() == carla.TrafficLightState.Red
        except Exception:
            return False

    def _obstacle_state(self, ego_tf: Any, ego_speed_ms: float, snapshot: Any) -> Tuple[float, bool, float]:
        """Scan the forward cone for the nearest vehicle or walker and estimate time-to-collision."""
        min_dist, is_pedestrian, ttc = 99.0, False, 99.0
        if snapshot is None or ego_tf is None or not self._actor_cache:
            return min_dist, is_pedestrian, ttc

        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        ego_loc = ego_tf.location

        for actor_id, actor_is_pedestrian in self._actor_cache:
            try:
                actor_snap = snapshot.find(actor_id)
                if actor_snap is None:
                    continue
                other_tf = actor_snap.get_transform()
                rel_x = other_tf.location.x - ego_loc.x
                rel_y = other_tf.location.y - ego_loc.y

                forward_dist = rel_x * fwd.x + rel_y * fwd.y
                if forward_dist <= 0.0 or forward_dist > self.FRONT_CONE_LENGTH:
                    continue
                if abs(rel_x * right.x + rel_y * right.y) > self.FRONT_CONE_HALF_WIDTH:
                    continue

                gap = max(0.0, forward_dist - self.BUMPER_ALLOWANCE)
                if gap >= min_dist:
                    continue

                min_dist = gap
                is_pedestrian = actor_is_pedestrian

                other_vel = actor_snap.get_velocity()
                closing_speed = ego_speed_ms - (other_vel.x * fwd.x + other_vel.y * fwd.y)
                ttc = (gap / closing_speed) if closing_speed > 0.1 else 99.0
            except Exception:
                continue

        return min_dist, is_pedestrian, min(99.0, ttc)

    def extract(
        self,
        ego: Any,
        speed_kmh: float,
        time_step: int,
        throttle: float,
        steer: float,
        brake: float,
        is_collision: bool,
        is_off_road: bool
    ) -> Dict[str, Any]:
        """Assemble the complete reward state for the current simulation step."""
        state = {
            "speed_kmh": speed_kmh, "heading_cos": 1.0, "heading_cos_far": 1.0,
            "lateral_dist": 0.0, "curve_factor": 1.0, "is_junction": False,
            "steer": steer, "throttle": throttle, "brake": brake,
            "is_at_red_light": False, "min_obs_dist": 99.0, "is_pedestrian": False,
            "ttc_seconds": 99.0, "is_collision": bool(is_collision),
            "is_off_road": bool(is_off_road), "time_step": int(time_step)
        }
        if ego is None:
            return state

        try:
            ego_tf = ego.get_transform()
        except Exception:
            return state

        lane = self._lane_state(ego_tf)
        state["heading_cos"] = lane["heading_cos"]
        state["heading_cos_far"] = lane["heading_cos_far"]
        state["lateral_dist"] = lane["lateral_dist"]
        state["curve_factor"] = lane["curve_factor"]
        state["is_junction"] = lane["is_junction"]
        state["is_off_road"] = bool(is_off_road or lane["off_road"])

        state["is_at_red_light"] = self._is_at_red_light(ego)

        if self._refresh_counter <= 0:
            self._refresh_actor_cache(getattr(ego, "id", -1))
            self._refresh_counter = self.ACTOR_REFRESH_INTERVAL
        self._refresh_counter -= 1

        snapshot = None
        try:
            snapshot = self.world.get_snapshot() if self.world is not None else None
        except Exception:
            snapshot = None

        min_dist, is_pedestrian, ttc = self._obstacle_state(ego_tf, speed_kmh / 3.6, snapshot)
        state["min_obs_dist"] = min_dist
        state["is_pedestrian"] = is_pedestrian
        state["ttc_seconds"] = ttc
        return state
