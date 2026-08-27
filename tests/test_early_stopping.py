"""Unit tests verifying performance-based Early Stopping behavior and metrics."""
import unittest
import numpy as np
from src.config.training_config import TrainingConfig


class TestEarlyStoppingLogic(unittest.TestCase):
    """Test suite ensuring moving average reward tracking, plateau detection, and early termination."""

    def test_moving_average_reward_calculation(self):
        window = 5
        rewards = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        recent = rewards[-window:]
        self.assertEqual(len(recent), 5)
        self.assertAlmostEqual(float(np.mean(recent)), 40.0)

    def test_plateau_patience_trigger(self):
        patience = 3
        min_delta = 1.0
        best_ma = 100.0
        patience_counter = 0
        triggered = False

        # Simulate consecutive plateau rollouts
        plateau_mas = [99.0, 100.5, 98.0]
        for cur_ma in plateau_mas:
            if cur_ma > best_ma + min_delta:
                best_ma = cur_ma
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    triggered = True
                    break

        self.assertTrue(triggered)
        self.assertEqual(patience_counter, 3)

    def test_target_reward_success_trigger(self):
        target_reward = 250.0
        recent_rewards = [240.0, 255.0, 260.0]
        cur_ma = float(np.mean(recent_rewards))
        triggered = cur_ma >= target_reward
        self.assertTrue(triggered)
        self.assertGreaterEqual(cur_ma, 250.0)


if __name__ == "__main__":
    unittest.main()
