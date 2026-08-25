"""Unit tests for RolloutBuffer and Generalized Advantage Estimation (GAE) with single and multi-env isolation."""
import unittest
import torch
import numpy as np


class TestRolloutBuffer(unittest.TestCase):
    """Test suite verifying RolloutBuffer storage, GAE calculation, and multi-env trajectory isolation."""

    def test_single_env_rollout_buffer_gae(self):
        from src.training.rollout_buffer import RolloutBuffer

        buffer_size = 10
        device = torch.device("cpu")
        buffer = RolloutBuffer(buffer_size=buffer_size, gamma=0.99, gae_lambda=0.95, device=device, num_envs=1)

        for i in range(buffer_size):
            buffer.add(
                obs_vis=torch.randn(1, 128),
                speed=torch.tensor([[20.0]]),
                action=torch.tensor([[0.5, 0.0, 0.0]]),
                log_prob=torch.tensor([-0.5]),
                reward=1.0,
                done=False if i < buffer_size - 1 else True,
                value=torch.tensor([0.8])
            )

        self.assertEqual(len(buffer), buffer_size)
        self.assertEqual(buffer.total_transitions, buffer_size)

        next_value = torch.tensor([0.5])
        b_vis, b_spd, b_act, b_logp, b_adv, b_ret, b_val = buffer.compute_returns_and_advantages(
            next_value=next_value,
            next_done=True
        )

        self.assertEqual(b_adv.shape, (buffer_size,))
        self.assertEqual(b_ret.shape, (buffer_size,))
        self.assertEqual(b_act.shape, (buffer_size, 3))

    def test_multi_env_rollout_buffer_isolation(self):
        from src.training.rollout_buffer import RolloutBuffer

        buffer_size = 8
        num_envs = 2
        device = torch.device("cpu")
        buffer = RolloutBuffer(buffer_size=buffer_size, gamma=0.99, gae_lambda=0.95, device=device, num_envs=num_envs)

        # Env 0 terminates at step 3, Env 1 does NOT terminate at step 3
        for i in range(buffer_size):
            dones = np.array([True if i == 3 else False, False], dtype=bool)
            rewards = np.array([1.0, 2.0], dtype=np.float32)
            values = torch.tensor([0.5, 1.0])
            buffer.add(
                obs_vis=torch.randn(num_envs, 64),
                speed=torch.ones((num_envs, 1)) * 20.0,
                action=torch.zeros((num_envs, 3)),
                log_prob=torch.tensor([-0.2, -0.3]),
                reward=rewards,
                done=dones,
                value=values
            )

        self.assertEqual(len(buffer), buffer_size)
        self.assertEqual(buffer.total_transitions, buffer_size * num_envs)

        next_value = torch.tensor([0.4, 0.9])
        next_done = np.array([False, False])
        b_vis, b_spd, b_act, b_logp, b_adv, b_ret, b_val = buffer.compute_returns_and_advantages(
            next_value=next_value,
            next_done=next_done
        )

        # Output tensors should be flattened to (buffer_size * num_envs)
        total_samples = buffer_size * num_envs
        self.assertEqual(b_adv.shape, (total_samples,))
        self.assertEqual(b_ret.shape, (total_samples,))
        self.assertEqual(b_act.shape, (total_samples, 3))
        self.assertEqual(b_spd.shape, (total_samples, 1))
        self.assertEqual(b_vis.shape, (total_samples, 64))


if __name__ == "__main__":
    unittest.main()
