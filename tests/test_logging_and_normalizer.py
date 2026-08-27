"""Unit tests for telemetry loggers, running normalizers, and hardware monitor."""
import os
import shutil
import tempfile
import unittest
import numpy as np

try:
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
except Exception:
    pass


class TestLoggingAndNormalizer(unittest.TestCase):
    """Test suite verifying reward normalizers, telemetry loggers, and hardware probes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_running_mean_std(self):
        from src.logging.normalizer import RunningMeanStd

        normalizer = RunningMeanStd()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            normalizer.update(v)

        self.assertAlmostEqual(normalizer.mean, float(np.mean(values)), delta=1.0)
        self.assertGreater(normalizer.std, 0.0)

        # Test array updates
        batch_normalizer = RunningMeanStd()
        batch_normalizer.update(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        self.assertAlmostEqual(batch_normalizer.mean, float(np.mean(values)), delta=1.0)
        self.assertGreater(batch_normalizer.std, 0.0)

    def test_hardware_monitor(self):
        from src.logging.hardware_monitor import HardwareMonitor

        metrics = HardwareMonitor.get_metrics()
        self.assertIn("gpu_mem_used_mb", metrics)
        self.assertIn("sys_cpu_pct", metrics)
        self.assertIn("sys_ram_used_gb", metrics)

    def test_csv_logger(self):
        from src.logging.csv_logger import CSVTelemetryLogger

        csv_path = os.path.join(self.temp_dir, "test_telemetry.csv")
        logger = CSVTelemetryLogger(csv_path)

        dummy_row = {
            "global_step": 1,
            "episode": 1,
            "step_in_ep": 1,
            "speed_kmh": 20.5,
            "action_throttle": 0.5,
            "action_steer": 0.0,
            "action_brake": 0.0,
            "raw_reward": 1.25,
            "normalized_reward": 1.1,
            "curriculum_alpha": 0.5,
            "r_speed": 1.0,
            "r_heading": 0.5,
            "r_lateral": 0.9,
            "r_boundary": 0.0,
            "r_steer": 0.0,
            "r_comfort": 0.0,
            "r_wrong_way": 0.0,
            "r_light": 0.0,
            "r_obstacle": 0.0,
            "r_ttc": 0.0,
            "r_idle": 0.0,
            "r_stall": 0.0,
            "loss_policy": -0.0123,
            "loss_value": 0.4567,
            "loss_entropy": 3.456,
            "loss_approx_kl": 0.005,
            "loss_clip_fraction": 0.12,
            "loss_explained_variance": 0.85,
            "sps": 120.5,
            "fps": 482.0,
            "gpu_mem_used_mb": 1024.0,
            "gpu_mem_pct": 10.0,
            "sys_cpu_pct": 15.0,
            "sys_ram_used_gb": 4.0,
            "is_collision": False,
            "is_off_road": False,
            "termination_reason": ""
        }
        logger.log_step(dummy_row)
        logger.close()

        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # Header + 1 row


if __name__ == "__main__":
    unittest.main()
