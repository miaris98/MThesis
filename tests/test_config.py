"""Unit tests for training configuration parsing and parameter validation."""
import unittest


class TestTrainingConfig(unittest.TestCase):
    """Test suite verifying configuration objects, CLI parsing, and validation."""

    def test_default_config(self):
        from src.config.training_config import TrainingConfig

        cfg = TrainingConfig.from_args([])
        self.assertEqual(cfg.total_steps, 50000)
        self.assertEqual(cfg.policy_arch, "qwen100m")
        self.assertEqual(cfg.backbone, "resnet18")
        self.assertEqual(cfg.town, "Town10HD_Opt")
        self.assertEqual(cfg.num_envs, 1)
        self.assertEqual(cfg.get_ports(), [2000])

    def test_custom_args(self):
        from src.config.training_config import TrainingConfig

        custom_args = ["--total-steps", "1000", "--policy-arch", "qwen100m", "--backbone", "lav", "--town", "Town01", "--num-envs", "2"]
        cfg = TrainingConfig.from_args(custom_args)
        self.assertEqual(cfg.total_steps, 1000)
        self.assertEqual(cfg.policy_arch, "qwen100m")
        self.assertEqual(cfg.backbone, "lav")
        self.assertEqual(cfg.town, "Town01")
        self.assertEqual(cfg.num_envs, 2)
        self.assertEqual(cfg.get_ports(), [2000, 2004])

    def test_explicit_ports(self):
        from src.config.training_config import TrainingConfig

        custom_args = ["--carla-ports", "2000,2004,2008"]
        cfg = TrainingConfig.from_args(custom_args)
        self.assertEqual(cfg.num_envs, 3)
        self.assertEqual(cfg.get_ports(), [2000, 2004, 2008])


if __name__ == "__main__":
    unittest.main()
