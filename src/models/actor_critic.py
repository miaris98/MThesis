"""Unified ActorCriticPPO model combining modular vision encoders with transformer/MLP decision heads."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.backbones import (
    CNNFeatureExtractor,
    PretrainedVisionFeatureExtractor,
    ERFNetFeatureExtractor
)
from src.models.transformer import QwenDecisionTransformer


class ActorCriticPPO(nn.Module):
    """
    PPO Actor-Critic Policy Network with modular Vision Encoder and Decision Architecture.
    Supports 500M/900M Qwen Decision Transformers, ResNet/LAV/ERFNet backbones, and feature caching.
    """
    def __init__(
        self, 
        action_dim: int = 3, 
        features_dim: int = 512, 
        backbone_name: str = "lav", 
        policy_arch: str = "qwen500m", 
        freeze_backbone: bool = True, 
        use_pretrained: bool = True, 
        weights_path: Optional[str] = None
    ):
        super().__init__()
        self.policy_arch = policy_arch
        
        # 1. Vision Feature Extractor
        if use_pretrained:
            if backbone_name == "erfnet":
                self.encoder = ERFNetFeatureExtractor(
                    features_dim=features_dim,
                    freeze_backbone=freeze_backbone,
                    weights_path=weights_path
                )
            else:
                self.encoder = PretrainedVisionFeatureExtractor(
                    backbone_name=backbone_name,
                    features_dim=features_dim,
                    freeze_backbone=freeze_backbone,
                    weights_path=weights_path
                )
        else:
            self.encoder = CNNFeatureExtractor(in_channels=3, features_dim=features_dim)

        vis_dim = getattr(self.encoder, 'visual_feature_dim', features_dim)

        # 2. PPO Decision Policy Architecture
        if policy_arch in ["qwen100m", "qwen500m", "qwen900m", "qwen", "transformer"]:
            if "900m" in policy_arch:
                model_size = "900m"
            elif "500m" in policy_arch:
                model_size = "500m"
            else:
                model_size = "100m"
            print(f"--> Initializing {model_size.upper()} Parameter Qwen-Style Decision Transformer...")
            self.decision_net = QwenDecisionTransformer(
                in_features=vis_dim,
                action_dim=action_dim,
                model_size=model_size
            )
            param_count = sum(p.numel() for p in self.decision_net.parameters())
            print(f"✓ {model_size.upper()} Qwen Decision Transformer initialized! Policy Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
        else:
            self.decision_net = None
            self.actor_mean = nn.Sequential(
                nn.Linear(features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh()
            )
            with torch.no_grad():
                self.actor_mean[2].bias.data[0] = 0.5
                self.actor_mean[2].bias.data[1] = 0.0
                self.actor_mean[2].bias.data[2] = -0.5
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            self.critic = nn.Sequential(
                nn.Linear(features_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract multi-camera visual features using vision backbone."""
        return self.encoder.extract_visual_features(image)

    def get_action_and_value(
        self, 
        image: Optional[torch.Tensor] = None, 
        speed: Optional[torch.Tensor] = None, 
        action: Optional[torch.Tensor] = None, 
        deterministic: bool = False, 
        visual_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters, sample action, log_prob, entropy, and state-value."""
        if visual_features is None:
            visual_features = self.encoder.extract_visual_features(image)

        if self.decision_net is not None:
            action_mean, actor_log_std, value = self.decision_net(visual_features, speed)
            action_std = torch.exp(torch.clamp(actor_log_std, -2.0, 0.5))
        else:
            features = self.encoder.forward_with_visual_features(visual_features, speed)
            action_mean = self.actor_mean(features)
            action_std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
            value = self.critic(features).squeeze(-1)

        dist = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return action, log_prob, entropy, value

    def train(self, mode: bool = True):
        """Set training mode while keeping frozen vision backbone strictly in eval mode."""
        super().train(mode)
        if getattr(self.encoder, 'freeze_backbone', False):
            if hasattr(self.encoder, 'backbone'):
                self.encoder.backbone.eval()
            else:
                self.encoder.eval()
        return self
