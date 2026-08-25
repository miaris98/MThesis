"""Unit tests for neural network architectures, backbones, and transformer components."""
import unittest
import torch
import numpy as np


class TestModels(unittest.TestCase):
    """Test suite verifying all neural network models and transformer architectures."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.img_height = 256
        self.img_width = 768  # 3 cameras stitched: 256 x 768
        self.dummy_image = torch.randint(0, 256, (self.batch_size, self.img_height, self.img_width, 3), dtype=torch.uint8).to(self.device)
        self.dummy_speed = torch.tensor([[20.0], [15.0]], dtype=torch.float32).to(self.device)

    def test_transformer_layers(self):
        from src.models.transformer.layers import RMSNorm, SwiGLU, QwenAttentionWithSkip, QwenTransformerBlock

        dim = 128
        x = torch.randn(self.batch_size, 10, dim).to(self.device)
        
        # Test RMSNorm
        norm = RMSNorm(dim).to(self.device)
        norm_out = norm(x)
        self.assertEqual(norm_out.shape, x.shape)

        # Test SwiGLU
        ffn = SwiGLU(dim, hidden_dim=256).to(self.device)
        ffn_out = ffn(x)
        self.assertEqual(ffn_out.shape, x.shape)

        # Test Attention with Skip
        attn = QwenAttentionWithSkip(dim=dim, num_heads=4).to(self.device)
        attn_out = attn(x)
        self.assertEqual(attn_out.shape, x.shape)

        # Test Transformer Block
        block = QwenTransformerBlock(dim=dim, num_heads=4, ffn_dim=256).to(self.device)
        block_out = block(x)
        self.assertEqual(block_out.shape, x.shape)

    def test_qwen_decision_transformer(self):
        from src.models.transformer.qwen_transformer import QwenDecisionTransformer

        in_features = 1536
        vis_features = torch.randn(self.batch_size, in_features).to(self.device)
        speed = self.dummy_speed

        model = QwenDecisionTransformer(in_features=in_features, action_dim=3, model_size="500m").to(self.device)
        action_mean, log_std, value = model(vis_features, speed)

        self.assertEqual(action_mean.shape, (self.batch_size, 3))
        self.assertEqual(log_std.shape, (3,))
        self.assertEqual(value.shape, (self.batch_size,))

    def test_cnn_feature_extractor(self):
        from src.models.backbones.cnn_backbone import CNNFeatureExtractor

        extractor = CNNFeatureExtractor(in_channels=3, features_dim=512).to(self.device)
        vis_feat = extractor.extract_visual_features(self.dummy_image)
        self.assertEqual(vis_feat.shape[0], self.batch_size)
        
        out = extractor.forward_with_visual_features(vis_feat, self.dummy_speed)
        self.assertEqual(out.shape, (self.batch_size, 512))

    def test_resnet_feature_extractor(self):
        from src.models.backbones.resnet_backbone import PretrainedVisionFeatureExtractor

        extractor = PretrainedVisionFeatureExtractor(backbone_name="resnet18", features_dim=512, freeze_backbone=True).to(self.device)
        vis_feat = extractor.extract_visual_features(self.dummy_image)
        self.assertEqual(vis_feat.shape[0], self.batch_size)
        self.assertEqual(vis_feat.shape[1], 512 * 3)

        out = extractor.forward_with_visual_features(vis_feat, self.dummy_speed)
        self.assertEqual(out.shape, (self.batch_size, 512))

    def test_actor_critic_ppo(self):
        from src.models.actor_critic import ActorCriticPPO

        agent = ActorCriticPPO(
            action_dim=3,
            features_dim=512,
            backbone_name="resnet18",
            policy_arch="qwen500m",
            freeze_backbone=True,
            use_pretrained=True
        ).to(self.device)

        action, log_prob, entropy, value = agent.get_action_and_value(
            image=self.dummy_image,
            speed=self.dummy_speed
        )

        self.assertEqual(action.shape, (self.batch_size, 3))
        self.assertEqual(log_prob.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertEqual(value.shape, (self.batch_size,))


if __name__ == "__main__":
    unittest.main()
