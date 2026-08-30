"""Unit tests for the progress-and-violations reward calculator."""
import unittest


class TestRewardCalculator(unittest.TestCase):
    """Test suite verifying dense progress reward and event-triggered violation penalties."""

    def _base_state(self, **overrides):
        state = {
            "speed_kmh": 25.0,
            "heading_cos": 1.0,
            "steer": 0.0,
            "throttle": 0.5,
            "brake": 0.0,
            "is_at_red_light": False,
            "min_obs_dist": 20.0,
            "is_pedestrian": False,
            "ttc_seconds": 99.0,
            "is_collision": False,
            "is_off_road": False,
            "time_step": 50
        }
        state.update(overrides)
        return state

    def test_standing_still_is_reward_neutral(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=0.0, throttle=0.0, time_step=10)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(reward, 0.0)
        self.assertEqual(info["r_progress"], 0.0)

    def test_forward_motion_earns_dense_progress_reward(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=36.0, heading_cos=1.0)  # 10 m/s
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0, dt=0.05)

        expected_progress = RewardCalculator.PROGRESS_PER_METER * 10.0 * 0.05
        self.assertAlmostEqual(info["r_progress"], expected_progress, places=5)
        self.assertAlmostEqual(reward, expected_progress, places=5)
        self.assertGreater(reward, 0.0)

    def test_sideways_or_backward_heading_earns_no_progress(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=20.0, heading_cos=-0.5)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(info["r_progress"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_collision_penalty(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(is_collision=True)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(info["r_terminal"], RewardCalculator.COLLISION_PENALTY)
        self.assertLess(reward, -15.0)

    def test_off_road_penalty(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(is_off_road=True)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(info["r_terminal"], RewardCalculator.OFF_ROAD_PENALTY)

    def test_curriculum_softens_terminal_penalty(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(is_off_road=True)

        r_warmup, _ = calculator.compute_reward(state, curriculum_factor=0.2)
        r_mature, _ = calculator.compute_reward(state, curriculum_factor=1.0)
        self.assertGreater(r_warmup, r_mature)

    def test_legal_red_light_stop_is_neutral(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=0.0, throttle=0.0, brake=1.0, is_at_red_light=True)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(info["r_light"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_running_red_light_is_penalized_and_blocks_progress(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=30.0, throttle=0.7, brake=0.0, is_at_red_light=True)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(info["r_light"], RewardCalculator.RED_LIGHT_VIOLATION_PENALTY)
        self.assertEqual(info["r_progress"], 0.0)
        self.assertLess(reward, 0.0)

    def test_accelerating_into_obstacle_is_penalized(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(min_obs_dist=2.0, throttle=0.8, brake=0.0)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertLess(info["r_obstacle"], 0.0)

    def test_braking_near_obstacle_is_not_penalized(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(min_obs_dist=2.0, throttle=0.0, brake=1.0)
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertEqual(info["r_obstacle"], 0.0)

    def test_pedestrian_obstacle_penalty_is_amplified(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        vehicle_state = self._base_state(min_obs_dist=2.0, throttle=0.8, is_pedestrian=False)
        ped_state = self._base_state(min_obs_dist=2.0, throttle=0.8, is_pedestrian=True)

        _, vehicle_info = calculator.compute_reward(vehicle_state, curriculum_factor=1.0)
        _, ped_info = calculator.compute_reward(ped_state, curriculum_factor=1.0)

        self.assertLess(ped_info["r_obstacle"], vehicle_info["r_obstacle"])

    def test_low_ttc_is_penalized(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(ttc_seconds=0.5)
        _, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertLess(info["r_ttc"], 0.0)

    def test_stall_triggers_after_timeout_grace_period(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=0.0, throttle=0.0, time_step=50)

        for _ in range(RewardCalculator.STALL_TIMEOUT_STEPS - 1):
            _, info = calculator.compute_reward(state, curriculum_factor=1.0)
            self.assertFalse(info["is_stalled"])

        _, info = calculator.compute_reward(state, curriculum_factor=1.0)
        self.assertTrue(info["is_stalled"])
        self.assertEqual(info["r_terminal"], RewardCalculator.STALL_PENALTY)

    def test_stall_grace_period_ignores_early_episode_steps(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=0.0, throttle=0.0, time_step=5)

        for _ in range(RewardCalculator.STALL_TIMEOUT_STEPS + 10):
            _, info = calculator.compute_reward(state, curriculum_factor=1.0)

        self.assertFalse(info["is_stalled"])

    def test_reset_episode_tracking_clears_stall_counter(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state = self._base_state(speed_kmh=0.0, throttle=0.0, time_step=50)
        for _ in range(RewardCalculator.STALL_TIMEOUT_STEPS):
            calculator.compute_reward(state, curriculum_factor=1.0)
        self.assertGreaterEqual(calculator.stalled_steps, RewardCalculator.STALL_TIMEOUT_STEPS)

        calculator.reset_episode_tracking()
        self.assertEqual(calculator.stalled_steps, 0)


if __name__ == "__main__":
    unittest.main()
