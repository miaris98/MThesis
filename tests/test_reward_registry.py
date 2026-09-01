"""Contract tests every swappable reward function must satisfy, plus per-formulation behavior."""
import unittest


def base_state(**overrides):
    """A neutral driving state: centered, aligned, cruising, no hazards."""
    state = {
        "speed_kmh": 25.0, "heading_cos": 1.0, "lateral_dist": 0.0, "lane_width": 3.5,
        "is_junction": False, "steer": 0.0, "throttle": 0.5, "brake": 0.0,
        "is_at_red_light": False, "min_obs_dist": 99.0, "is_pedestrian": False,
        "ttc_seconds": 99.0, "is_collision": False, "is_off_road": False, "time_step": 50
    }
    state.update(overrides)
    return state


class TestRegistry(unittest.TestCase):
    """The registry itself: lookup, error handling, and the shared telemetry contract."""

    def test_all_registered_names_construct(self):
        from src.envs.rewards import available_rewards, make_reward
        for name in available_rewards():
            self.assertIsNotNone(make_reward(name, desired_speed=25.0))

    def test_custom_1_is_registered_and_is_the_default(self):
        from src.envs.rewards import available_rewards, make_reward, Custom1Reward
        self.assertIn("custom_1", available_rewards())
        self.assertIsInstance(make_reward(), Custom1Reward)

    def test_unknown_name_raises_with_available_list(self):
        from src.envs.rewards import make_reward
        with self.assertRaises(ValueError) as ctx:
            make_reward("does_not_exist")
        self.assertIn("custom_1", str(ctx.exception))

    def test_every_reward_emits_the_full_telemetry_schema(self):
        # The CSV schema is fixed, so runs stay comparable across reward functions.
        from src.envs.rewards import available_rewards, make_reward, REWARD_COMPONENTS
        for name in available_rewards():
            reward, info = make_reward(name).compute_reward(base_state(), 1.0, 0.05)
            self.assertIsInstance(reward, float, msg=name)
            for key in REWARD_COMPONENTS:
                self.assertIn(key, info, msg=f"{name} missing {key}")
            self.assertIn("is_stalled", info, msg=name)

    def test_every_reward_resets_episode_state(self):
        from src.envs.rewards import available_rewards, make_reward
        stalled = base_state(speed_kmh=0.0, throttle=0.0, time_step=200)
        for name in available_rewards():
            fn = make_reward(name)
            for _ in range(90):
                fn.compute_reward(stalled, 1.0, 0.05)
            self.assertGreater(fn.stalled_steps, 0, msg=name)
            fn.reset_episode_tracking()
            self.assertEqual(fn.stalled_steps, 0, msg=name)

    def test_every_reward_terminates_a_persistent_stall(self):
        from src.envs.rewards import available_rewards, make_reward
        stalled = base_state(speed_kmh=0.0, throttle=0.0, time_step=200)
        for name in available_rewards():
            fn = make_reward(name)
            info = {}
            for _ in range(200):
                _, info = fn.compute_reward(stalled, 1.0, 0.05)
            self.assertTrue(info["is_stalled"], msg=name)

    def test_no_reward_pays_for_a_legal_stop_more_than_driving(self):
        # Guards the original failure mode: idling must never beat clean forward driving.
        from src.envs.rewards import available_rewards, make_reward
        for name in available_rewards():
            idle = make_reward(name).compute_reward(
                base_state(speed_kmh=0.0, throttle=0.0, time_step=1), 1.0, 0.05)[0]
            drive = make_reward(name).compute_reward(
                base_state(speed_kmh=25.0, time_step=1), 1.0, 0.05)[0]
            self.assertGreater(drive, idle, msg=f"{name}: idling scores >= driving")


class TestCustom1Economics(unittest.TestCase):
    """Custom_1's rebalanced scale - the fix for the unwinnable-reward diagnosis."""

    def test_clean_episode_outearns_a_single_off_road(self):
        from src.envs.rewards import Custom1Reward
        fn = Custom1Reward(desired_speed=25.0)
        step, _ = fn.compute_reward(base_state(speed_kmh=20.0), 1.0, 0.05)
        # A 100-step clean episode must be worth more than one off-road event costs.
        self.assertGreater(step * 100, abs(Custom1Reward.OFF_ROAD_PENALTY))

    def test_lane_penalty_is_comparable_to_progress_at_the_lane_edge(self):
        from src.envs.rewards import Custom1Reward
        fn = Custom1Reward(desired_speed=25.0)
        _, info = fn.compute_reward(base_state(speed_kmh=20.0, lateral_dist=1.75), 1.0, 0.05)
        # Drifting to the edge must cost more than the progress earned while doing it,
        # so the gradient turns the car back before the off-road cliff rather than at it.
        self.assertLess(info["r_lane"], -info["r_progress"])

    def test_standing_still_centered_is_exactly_neutral(self):
        from src.envs.rewards import Custom1Reward
        reward, info = Custom1Reward().compute_reward(
            base_state(speed_kmh=0.0, throttle=0.0, time_step=1), 1.0, 0.05)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["r_progress"], 0.0)


class TestLiteratureRewards(unittest.TestCase):
    """Behavior specific to each published formulation."""

    def test_leaderboard_infraction_is_multiplicative_and_latched(self):
        from src.envs.rewards import LeaderboardReward
        fn = LeaderboardReward(desired_speed=25.0)
        self.assertEqual(fn.score_penalty, 1.0)
        fn.compute_reward(base_state(is_collision=True, min_obs_dist=5.0), 1.0, 0.05)
        after = fn.score_penalty
        self.assertAlmostEqual(after, LeaderboardReward.PENALTY_COLLISION_VEHICLE, places=6)
        # Latched: a multi-frame contact is charged once, not once per frame.
        fn.compute_reward(base_state(is_collision=True, min_obs_dist=5.0), 1.0, 0.05)
        self.assertAlmostEqual(fn.score_penalty, after, places=6)

    def test_leaderboard_infraction_devalues_later_progress(self):
        from src.envs.rewards import LeaderboardReward
        clean = LeaderboardReward()
        crashed = LeaderboardReward()
        crashed.compute_reward(base_state(is_collision=True, min_obs_dist=5.0), 1.0, 0.05)
        _, clean_info = clean.compute_reward(base_state(speed_kmh=20.0), 1.0, 0.05)
        _, dirty_info = crashed.compute_reward(base_state(speed_kmh=20.0), 1.0, 0.05)
        self.assertLess(dirty_info["r_progress"], clean_info["r_progress"])

    def test_leaderboard_pedestrian_is_the_harshest_collision(self):
        from src.envs.rewards import LeaderboardReward
        self.assertLess(LeaderboardReward.PENALTY_COLLISION_PEDESTRIAN,
                        LeaderboardReward.PENALTY_COLLISION_VEHICLE)
        self.assertLess(LeaderboardReward.PENALTY_COLLISION_VEHICLE,
                        LeaderboardReward.PENALTY_COLLISION_STATIC)

    def test_roach_rewards_stopping_for_a_hazard(self):
        from src.envs.rewards import RoachReward
        fn = RoachReward(desired_speed=25.0)
        # Roach rewrites the speed target to 0 at a hazard, so a correct stop scores as
        # highly as correct cruising - it is not merely un-penalized.
        stopped_at_light, _ = fn.compute_reward(
            base_state(speed_kmh=0.0, is_at_red_light=True, time_step=1), 1.0, 0.05)
        fn.reset_episode_tracking()
        cruising, _ = fn.compute_reward(base_state(speed_kmh=25.0, time_step=1), 1.0, 0.05)
        self.assertAlmostEqual(stopped_at_light, cruising, places=5)

    def test_roach_penalizes_driving_through_a_hazard(self):
        from src.envs.rewards import RoachReward
        fn = RoachReward(desired_speed=25.0)
        through, _ = fn.compute_reward(
            base_state(speed_kmh=25.0, is_at_red_light=True, time_step=1), 1.0, 0.05)
        fn.reset_episode_tracking()
        stopped, _ = fn.compute_reward(
            base_state(speed_kmh=0.0, is_at_red_light=True, time_step=1), 1.0, 0.05)
        self.assertLess(through, stopped)

    def test_interp_e2e_collision_dominates_the_weighted_sum(self):
        from src.envs.rewards import InterpE2EReward
        fn = InterpE2EReward(desired_speed=25.0)
        crash, _ = fn.compute_reward(base_state(is_collision=True), 1.0, 0.05)
        self.assertLess(crash, -100.0)

    def test_interp_e2e_penalizes_exceeding_desired_speed(self):
        from src.envs.rewards import InterpE2EReward
        fn = InterpE2EReward(desired_speed=25.0)
        _, fast = fn.compute_reward(base_state(speed_kmh=40.0), 1.0, 0.05)
        self.assertLess(fast["r_obstacle"], 0.0)

    def test_interp_e2e_step_cost_makes_idling_negative(self):
        from src.envs.rewards import InterpE2EReward
        idle, _ = InterpE2EReward().compute_reward(
            base_state(speed_kmh=0.0, throttle=0.0, time_step=1), 1.0, 0.05)
        self.assertLess(idle, 0.0)


if __name__ == "__main__":
    unittest.main()
