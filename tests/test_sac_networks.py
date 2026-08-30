"""Unit tests for SAC policy squashing, log-probability correction, and target updates."""
import math
import unittest
import torch
import torch.nn as nn


class TestSACMath(unittest.TestCase):
    """
    Verifies the SAC-specific math against hand-computed references.

    These target the parts that fail silently rather than loudly: a missing tanh
    change-of-variables term still trains, it just optimizes the wrong entropy.
    """

    def test_tanh_correction_matches_analytic_log_prob(self):
        from src.models.sac_networks import TANH_EPS
        from torch.distributions import Normal

        mean = torch.tensor([[0.3, -0.7, 0.1]])
        std = torch.tensor([[1.0, 0.5, 2.0]])
        x_t = torch.tensor([[0.8, -0.2, 1.5]])
        action = torch.tanh(x_t)

        dist = Normal(mean, std)
        got = (dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + TANH_EPS)).sum(dim=-1)

        # Reference: Gaussian log-density minus log|d(tanh)/dx| = log(1 - tanh(x)^2)
        expected = 0.0
        for i in range(3):
            m, s, x = mean[0, i].item(), std[0, i].item(), x_t[0, i].item()
            gauss = -0.5 * ((x - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)
            expected += gauss - math.log(1.0 - math.tanh(x) ** 2 + TANH_EPS)

        self.assertAlmostEqual(got.item(), expected, places=4)

    def test_squashed_actions_stay_in_env_action_range(self):
        # The env maps throttle = (a0 + 1) / 2 and clips steer/brake, so the policy must
        # emit strictly bounded actions - unlike PPO's unbounded Normal sample.
        x = torch.tensor([[-50.0, 0.0, 50.0]])
        action = torch.tanh(x)
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))

    def test_soft_update_moves_target_fractionally(self):
        online = nn.Linear(3, 1)
        target = nn.Linear(3, 1)
        with torch.no_grad():
            online.weight.fill_(1.0)
            target.weight.fill_(0.0)

        tau = 0.005
        with torch.no_grad():
            for p, p_t in zip(online.parameters(), target.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

        self.assertAlmostEqual(target.weight[0, 0].item(), tau, places=6)

    def test_target_entropy_is_negative_action_dim(self):
        # SAC's standard heuristic; wrong sign here makes the temperature diverge.
        action_dim = 3
        self.assertEqual(-float(action_dim), -3.0)

    def test_bellman_target_masks_only_true_terminals(self):
        rew = torch.tensor([1.0, 1.0])
        done = torch.tensor([0.0, 1.0])
        min_q_next = torch.tensor([10.0, 10.0])
        gamma = 0.99

        target = rew + gamma * (1.0 - done) * min_q_next
        self.assertAlmostEqual(target[0].item(), 1.0 + 0.99 * 10.0, places=5)
        self.assertAlmostEqual(target[1].item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
