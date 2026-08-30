"""Unit tests for DrivingStateExtractor lane geometry, obstacle scanning, and TTC estimation."""
import math
import unittest


class Vec:
    """Minimal stand-in for carla.Vector3D / carla.Location."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Transform:
    """Minimal stand-in for carla.Transform with axis-aligned or yawed basis vectors."""
    def __init__(self, location, yaw_deg=0.0):
        self.location = location
        self.yaw = math.radians(yaw_deg)

    def get_forward_vector(self):
        return Vec(math.cos(self.yaw), math.sin(self.yaw), 0.0)

    def get_right_vector(self):
        return Vec(-math.sin(self.yaw), math.cos(self.yaw), 0.0)


class Waypoint:
    """Minimal stand-in for carla.Waypoint returning a fixed chain of successors."""
    def __init__(self, transform, lane_width=3.5, is_junction=False, successors=None):
        self.transform = transform
        self.lane_width = lane_width
        self.is_junction = is_junction
        self._successors = successors or {}

    def next(self, distance):
        wp = self._successors.get(distance)
        return [wp] if wp is not None else []


class Map:
    def __init__(self, waypoint):
        self.waypoint = waypoint

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        return self.waypoint


class ActorSnapshot:
    def __init__(self, transform, velocity):
        self._transform = transform
        self._velocity = velocity

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


class Snapshot:
    def __init__(self, actors):
        self.actors = actors

    def find(self, actor_id):
        return self.actors.get(actor_id)


class World:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def get_snapshot(self):
        return self._snapshot


class Ego:
    def __init__(self, transform, actor_id=1):
        self._transform = transform
        self.id = actor_id

    def get_transform(self):
        return self._transform


class TestDrivingStateExtractor(unittest.TestCase):
    """Test suite verifying lane projection, curvature, obstacle proximity, and TTC."""

    def _straight_lane_map(self):
        lane_tf = Transform(Vec(0.0, 0.0), yaw_deg=0.0)
        ahead = Waypoint(Transform(Vec(10.0, 0.0), yaw_deg=0.0))
        curve_ahead = Waypoint(Transform(Vec(15.0, 0.0), yaw_deg=0.0))
        wp = Waypoint(lane_tf, successors={10.0: ahead, 15.0: curve_ahead})
        return Map(wp)

    def test_centered_on_straight_lane(self):
        from src.envs.driving_state import DrivingStateExtractor

        extractor = DrivingStateExtractor(world=None, world_map=self._straight_lane_map())
        ego = Ego(Transform(Vec(5.0, 0.0), yaw_deg=0.0))
        state = extractor.extract(ego, speed_kmh=25.0, time_step=50, throttle=0.6, steer=0.0,
                                  brake=0.0, is_collision=False, is_off_road=False)

        self.assertAlmostEqual(state["heading_cos"], 1.0, places=5)
        self.assertAlmostEqual(state["heading_cos_far"], 1.0, places=5)
        self.assertAlmostEqual(state["lateral_dist"], 0.0, places=5)
        self.assertAlmostEqual(state["curve_factor"], 1.0, places=5)
        self.assertFalse(state["is_junction"])
        self.assertFalse(state["is_off_road"])

    def test_lateral_offset_and_off_road_detection(self):
        from src.envs.driving_state import DrivingStateExtractor

        extractor = DrivingStateExtractor(world=None, world_map=self._straight_lane_map())
        # Boundary is min(OFF_ROAD_LATERAL_LIMIT, lane_width/2 + 0.8) = min(1.8, 2.55) = 1.8 m.
        ego_inside = Ego(Transform(Vec(5.0, 1.0), yaw_deg=0.0))
        state_inside = extractor.extract(ego_inside, 25.0, 50, 0.6, 0.0, 0.0, False, False)
        self.assertAlmostEqual(state_inside["lateral_dist"], 1.0, places=5)
        self.assertAlmostEqual(state_inside["lane_width"], 3.5, places=5)
        self.assertFalse(state_inside["is_off_road"])

        # 2.0 m is inside the old lane-width rule but crosses the hard 1.8 m boundary.
        ego_boundary = Ego(Transform(Vec(5.0, 2.0), yaw_deg=0.0))
        state_boundary = extractor.extract(ego_boundary, 25.0, 50, 0.6, 0.0, 0.0, False, False)
        self.assertTrue(state_boundary["is_off_road"])

        ego_outside = Ego(Transform(Vec(5.0, 3.0), yaw_deg=0.0))
        state_outside = extractor.extract(ego_outside, 25.0, 50, 0.6, 0.0, 0.0, False, False)
        self.assertTrue(state_outside["is_off_road"])

    def test_wrong_way_heading_flags_off_road(self):
        from src.envs.driving_state import DrivingStateExtractor

        extractor = DrivingStateExtractor(world=None, world_map=self._straight_lane_map())
        ego = Ego(Transform(Vec(5.0, 0.0), yaw_deg=180.0))
        state = extractor.extract(ego, 20.0, 50, 0.6, 0.0, 0.0, False, False)

        self.assertLess(state["heading_cos"], -0.9)
        self.assertTrue(state["is_off_road"])

    def test_curve_factor_reduced_on_bend(self):
        from src.envs.driving_state import DrivingStateExtractor

        lane_tf = Transform(Vec(0.0, 0.0), yaw_deg=0.0)
        ahead = Waypoint(Transform(Vec(10.0, 0.0), yaw_deg=0.0))
        bend = Waypoint(Transform(Vec(15.0, 0.0), yaw_deg=60.0))
        wp = Waypoint(lane_tf, successors={10.0: ahead, 15.0: bend})
        extractor = DrivingStateExtractor(world=None, world_map=Map(wp))

        state = extractor.extract(Ego(Transform(Vec(5.0, 0.0))), 25.0, 50, 0.6, 0.0, 0.0, False, False)
        self.assertAlmostEqual(state["curve_factor"], math.cos(math.radians(60.0)), places=4)

    def test_obstacle_distance_and_ttc(self):
        from src.envs.driving_state import DrivingStateExtractor

        # Stationary walker 12 m directly ahead; ego closing at 10 m/s (36 km/h).
        walker_snap = ActorSnapshot(Transform(Vec(12.0, 0.0)), Vec(0.0, 0.0))
        world = World(Snapshot({7: walker_snap}))
        extractor = DrivingStateExtractor(world=world, world_map=self._straight_lane_map())
        extractor._actor_cache = [(7, True)]
        extractor._refresh_counter = 999  # keep the injected cache

        state = extractor.extract(Ego(Transform(Vec(0.0, 0.0))), 36.0, 50, 0.6, 0.0, 0.0, False, False)

        expected_gap = 12.0 - DrivingStateExtractor.BUMPER_ALLOWANCE
        self.assertAlmostEqual(state["min_obs_dist"], expected_gap, places=4)
        self.assertTrue(state["is_pedestrian"])
        self.assertAlmostEqual(state["ttc_seconds"], expected_gap / 10.0, places=3)

    def test_obstacle_behind_and_offset_ignored(self):
        from src.envs.driving_state import DrivingStateExtractor

        behind = ActorSnapshot(Transform(Vec(-10.0, 0.0)), Vec(0.0, 0.0))
        far_lateral = ActorSnapshot(Transform(Vec(10.0, 8.0)), Vec(0.0, 0.0))
        world = World(Snapshot({7: behind, 8: far_lateral}))
        extractor = DrivingStateExtractor(world=world, world_map=self._straight_lane_map())
        extractor._actor_cache = [(7, False), (8, False)]
        extractor._refresh_counter = 999

        state = extractor.extract(Ego(Transform(Vec(0.0, 0.0))), 36.0, 50, 0.6, 0.0, 0.0, False, False)

        self.assertEqual(state["min_obs_dist"], 99.0)
        self.assertEqual(state["ttc_seconds"], 99.0)
        self.assertFalse(state["is_pedestrian"])

    def test_receding_obstacle_has_no_ttc(self):
        from src.envs.driving_state import DrivingStateExtractor

        # Lead vehicle 10 m ahead travelling faster than the ego: no closing speed.
        lead = ActorSnapshot(Transform(Vec(10.0, 0.0)), Vec(20.0, 0.0))
        world = World(Snapshot({9: lead}))
        extractor = DrivingStateExtractor(world=world, world_map=self._straight_lane_map())
        extractor._actor_cache = [(9, False)]
        extractor._refresh_counter = 999

        state = extractor.extract(Ego(Transform(Vec(0.0, 0.0))), 36.0, 50, 0.6, 0.0, 0.0, False, False)

        self.assertLess(state["min_obs_dist"], 99.0)
        self.assertEqual(state["ttc_seconds"], 99.0)

    def test_missing_ego_returns_neutral_state(self):
        from src.envs.driving_state import DrivingStateExtractor

        extractor = DrivingStateExtractor(world=None, world_map=self._straight_lane_map())
        state = extractor.extract(None, 0.0, 0, 0.0, 0.0, 0.0, False, False)

        self.assertEqual(state["min_obs_dist"], 99.0)
        self.assertEqual(state["heading_cos"], 1.0)
        self.assertFalse(state["is_at_red_light"])


if __name__ == "__main__":
    unittest.main()
