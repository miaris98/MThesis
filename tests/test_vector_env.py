"""Unit tests for Vectorized CARLA Environment wrappers and dummy step logic."""
import unittest
import numpy as np


class MockCarlaEnv:
    """Mock single CARLA environment for testing vector environment orchestration."""
    def __init__(self, env_id: int = 0):
        self.env_id = env_id
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return {
            "image": np.zeros((256, 768, 3), dtype=np.uint8),
            "speed": np.array([0.0], dtype=np.float32)
        }, {"env_id": self.env_id}

    def step(self, action):
        self.step_count += 1
        speed = float(self.step_count * 5.0)
        obs = {
            "image": np.zeros((256, 768, 3), dtype=np.uint8),
            "speed": np.array([speed], dtype=np.float32)
        }
        reward = 1.0
        terminated = bool(self.step_count >= 5)
        truncated = False
        info = {
            "env_id": self.env_id,
            "speed_kmh": speed,
            "is_collision": False,
            "is_off_road": False
        }
        return obs, reward, terminated, truncated, info

    def set_curriculum_factor(self, factor):
        self.curriculum_factor = factor

    def close(self):
        pass


class TestVectorEnv(unittest.TestCase):
    """Test suite verifying vector environment step, auto-reset, and observation batching."""

    def test_dummy_vector_env(self):
        from src.envs.vector_carla_env import DummyCarlaVectorEnv

        num_envs = 3
        env_fns = [lambda i=i: MockCarlaEnv(env_id=i) for i in range(num_envs)]
        vec_env = DummyCarlaVectorEnv(env_fns)

        obs, infos = vec_env.reset()
        self.assertEqual(obs["image"].shape, (num_envs, 256, 768, 3))
        self.assertEqual(obs["speed"].shape, (num_envs, 1))
        self.assertEqual(len(infos), num_envs)

        # Step vectorized environment
        actions = np.zeros((num_envs, 3), dtype=np.float32)
        for step_idx in range(6):
            obs, rewards, term, trunc, infos = vec_env.step(actions)
            self.assertEqual(obs["image"].shape, (num_envs, 256, 768, 3))
            self.assertEqual(obs["speed"].shape, (num_envs, 1))
            self.assertEqual(rewards.shape, (num_envs,))
            self.assertEqual(term.shape, (num_envs,))
            self.assertEqual(trunc.shape, (num_envs,))
            self.assertEqual(len(infos), num_envs)

            if step_idx == 4:
                # Step 5 reached: all envs should terminate and auto-reset
                self.assertTrue(np.all(term))
                for info in infos:
                    self.assertIn("terminal_observation", info)

        vec_env.close()


if __name__ == "__main__":
    unittest.main()
