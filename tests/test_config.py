"""Unit tests for training configuration parsing and parameter validation."""
import unittest


class TestTrainingConfig(unittest.TestCase):
    """Test suite verifying configuration objects, CLI parsing, and validation."""

    def test_default_config(self):
        from src.config.training_config import TrainingConfig

        cfg = TrainingConfig.from_args([])
        self.assertEqual(cfg.total_steps, 70000)
        self.assertEqual(cfg.policy_arch, "qwen100m")
        self.assertEqual(cfg.backbone, "resnet18")
        self.assertEqual(cfg.town, "Town10HD_Opt")
        self.assertEqual(cfg.num_envs, 2)
        self.assertEqual(cfg.get_ports(), [2000, 2004])
        self.assertTrue(cfg.early_stopping)
        self.assertEqual(cfg.early_stopping_patience, 20)

    def test_custom_args(self):
        from src.config.training_config import TrainingConfig

        custom_args = [
            "--total-steps", "1000", "--policy-arch", "qwen100m", "--backbone", "lav",
            "--town", "Town01", "--num-envs", "4", "--patience", "15", "--min-delta", "2.0"
        ]
        cfg = TrainingConfig.from_args(custom_args)
        self.assertEqual(cfg.total_steps, 1000)
        self.assertEqual(cfg.policy_arch, "qwen100m")
        self.assertEqual(cfg.backbone, "lav")
        self.assertEqual(cfg.town, "Town01")
        self.assertEqual(cfg.num_envs, 4)
        self.assertEqual(cfg.get_ports(), [2000, 2004, 2008, 2012])
        self.assertEqual(cfg.early_stopping_patience, 15)
        self.assertEqual(cfg.early_stopping_min_delta, 2.0)

    def test_explicit_ports(self):
        from src.config.training_config import TrainingConfig

        custom_args = ["--carla-ports", "2000,2004,2008", "--no-early-stopping", "--target-reward", "500.0"]
        cfg = TrainingConfig.from_args(custom_args)
        self.assertEqual(cfg.num_envs, 3)
        self.assertEqual(cfg.get_ports(), [2000, 2004, 2008])
        self.assertFalse(cfg.early_stopping)
        self.assertEqual(cfg.target_reward, 500.0)


if __name__ == "__main__":
    unittest.main()
