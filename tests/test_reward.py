"""Unit tests for the reward calculator and curriculum engine."""
import unittest
import numpy as np


class TestRewardCalculator(unittest.TestCase):
    """Test suite verifying multi-objective reward calculation and curriculum gating."""

    def test_reward_calculator_computations(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator(desired_speed=25.0)
        
        # Test 1: Optimal centered driving
        state = {
            "speed_kmh": 25.0,
            "heading_cos": 1.0,
            "heading_cos_far": 1.0,
            "lateral_dist": 0.0,
            "curve_factor": 1.0,
            "is_junction": False,
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
        
        reward, info = calculator.compute_reward(state, curriculum_factor=1.0)
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(info["r_lateral"], 1.0, places=2)
        self.assertGreater(info["r_speed"], 1.0)

        # Test 2: Collision penalty
        state_collision = dict(state, is_collision=True)
        reward_col, info_col = calculator.compute_reward(state_collision, curriculum_factor=1.0)
        self.assertLess(reward_col, -15.0)

    def test_curriculum_scaling(self):
        from src.envs.reward_calculator import RewardCalculator

        calculator = RewardCalculator()
        state_offroad = {
            "speed_kmh": 10.0,
            "heading_cos": 0.0,
            "heading_cos_far": 0.0,
            "lateral_dist": 2.5,
            "curve_factor": 1.0,
            "is_junction": False,
            "steer": 0.5,
            "throttle": 0.5,
            "brake": 0.0,
            "is_at_red_light": False,
            "min_obs_dist": 20.0,
            "is_pedestrian": False,
            "ttc_seconds": 99.0,
            "is_collision": False,
            "is_off_road": True,
            "time_step": 50
        }

        # With curriculum factor 0.2, terminal penalty is softer than with 1.0
        r_warmup, _ = calculator.compute_reward(state_offroad, curriculum_factor=0.2)
        r_mature, _ = calculator.compute_reward(state_offroad, curriculum_factor=1.0)
        self.assertGreater(r_warmup, r_mature)


if __name__ == "__main__":
    unittest.main()
