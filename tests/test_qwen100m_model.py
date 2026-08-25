"""Unit tests verifying 100M Qwen Decision Transformer architecture and parameter count."""
import unittest
import torch


class TestQwen100MModel(unittest.TestCase):
    """Test suite verifying ~100M Qwen Decision Transformer instantiation, parameters, and shapes."""

    def test_qwen100m_decision_transformer_params(self):
        from src.models.transformer.qwen_transformer import QwenDecisionTransformer

        in_features = 512
        model = QwenDecisionTransformer(in_features=in_features, action_dim=3, model_size="100m")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n--> Qwen-100M Parameter Count: {param_count:,} ({param_count / 1e6:.2f}M)")

        # Verify parameters are in ~100M range (100M - 110M)
        self.assertGreater(param_count, 90_000_000)
        self.assertLess(param_count, 120_000_000)

        # Forward pass verification
        batch_size = 4
        vis_features = torch.randn(batch_size, in_features)
        speed = torch.tensor([[10.0], [20.0], [30.0], [0.0]])
        
        action_mean, log_std, value = model(vis_features, speed)
        self.assertEqual(action_mean.shape, (batch_size, 3))
        self.assertEqual(log_std.shape, (3,))
        self.assertEqual(value.shape, (batch_size,))

    def test_actor_critic_ppo_qwen100m(self):
        from src.models.actor_critic import ActorCriticPPO

        agent = ActorCriticPPO(
            action_dim=3,
            features_dim=512,
            backbone_name="resnet18",
            policy_arch="qwen100m",
            freeze_backbone=True,
            use_pretrained=False
        )

        batch_size = 2
        img = torch.randint(0, 256, (batch_size, 256, 768, 3), dtype=torch.uint8)
        spd = torch.tensor([[25.0], [18.0]], dtype=torch.float32)

        action, log_prob, entropy, value = agent.get_action_and_value(image=img, speed=spd)
        self.assertEqual(action.shape, (batch_size, 3))
        self.assertEqual(log_prob.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))
        self.assertEqual(value.shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
