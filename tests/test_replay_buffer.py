"""Unit tests for the off-policy replay buffer used by SAC."""
import unittest
import numpy as np
import torch


class TestReplayBuffer(unittest.TestCase):
    """Test suite verifying circular insertion, capacity wrapping, and batch sampling."""

    def _buffer(self, capacity=10, visual_dim=4):
        from src.training.replay_buffer import ReplayBuffer
        return ReplayBuffer(capacity=capacity, visual_dim=visual_dim, action_dim=3,
                            device=torch.device("cpu"))

    def _add(self, buf, n=2, value=1.0, reward=0.5, done=0.0):
        buf.add(
            visual=torch.full((n, buf.visual_dim), value),
            speed=torch.full((n, 1), 10.0),
            action=torch.zeros((n, 3)),
            reward=np.full(n, reward, dtype=np.float32),
            next_visual=torch.full((n, buf.visual_dim), value + 1.0),
            next_speed=torch.full((n, 1), 12.0),
            done=np.full(n, done, dtype=np.float32)
        )

    def test_starts_empty(self):
        buf = self._buffer()
        self.assertEqual(len(buf), 0)
        with self.assertRaises(ValueError):
            buf.sample(4)

    def test_add_advances_length_by_num_envs(self):
        buf = self._buffer()
        self._add(buf, n=2)
        self.assertEqual(len(buf), 2)
        self._add(buf, n=2)
        self.assertEqual(len(buf), 4)

    def test_wraps_at_capacity_without_growing(self):
        buf = self._buffer(capacity=6)
        for _ in range(5):
            self._add(buf, n=2)
        self.assertEqual(len(buf), 6)
        self.assertTrue(buf.full)

    def test_stored_values_round_trip(self):
        buf = self._buffer(capacity=4, visual_dim=3)
        self._add(buf, n=2, value=7.0, reward=-2.5, done=1.0)
        vis, spd, act, rew, next_vis, next_spd, done = buf.sample(2)

        self.assertEqual(vis.shape, (2, 3))
        self.assertEqual(act.shape, (2, 3))
        self.assertTrue(torch.allclose(vis, torch.full((2, 3), 7.0)))
        self.assertTrue(torch.allclose(next_vis, torch.full((2, 3), 8.0)))
        self.assertTrue(torch.allclose(rew, torch.full((2,), -2.5)))
        self.assertTrue(torch.allclose(done, torch.ones(2)))

    def test_sample_is_capped_by_current_size(self):
        buf = self._buffer(capacity=100)
        self._add(buf, n=2)
        vis = buf.sample(64)[0]
        self.assertEqual(vis.shape[0], 2)

    def test_memory_estimate_scales_with_capacity(self):
        small = self._buffer(capacity=100, visual_dim=512)
        large = self._buffer(capacity=1000, visual_dim=512)
        self.assertAlmostEqual(large.memory_mb / small.memory_mb, 10.0, places=4)


if __name__ == "__main__":
    unittest.main()
