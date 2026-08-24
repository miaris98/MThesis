"""Unit tests for training configuration parsing and parameter validation."""
import unittest


class TestTrainingConfig(unittest.TestCase):
    """Test suite verifying configuration objects, CLI parsing, and validation."""

    def test_default_config(self):
        from src.config.training_config import TrainingConfig

        cfg = TrainingConfig.from_args([])
        self.assertEqual(cfg.total_steps, 50000)
        self.assertEqual(cfg.policy_arch, "qwen500m")
        self.assertEqual(cfg.backbone, "resnet18")
        self.assertEqual(cfg.town, "Town10HD_Opt")

    def test_custom_args(self):
        from src.config.training_config import TrainingConfig

        custom_args = ["--total-steps", "1000", "--policy-arch", "qwen500m", "--backbone", "lav", "--town", "Town01"]
        cfg = TrainingConfig.from_args(custom_args)
        self.assertEqual(cfg.total_steps, 1000)
        self.assertEqual(cfg.policy_arch, "qwen500m")
        self.assertEqual(cfg.backbone, "lav")
        self.assertEqual(cfg.town, "Town01")


if __name__ == "__main__":
    unittest.main()
