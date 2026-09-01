"""Soft Actor-Critic networks: tanh-squashed Gaussian policy and twin Q critics over cached features."""
import copy
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.qwen_sac_heads import QwenSACActor, QwenQCritic
from src.models.backbones import (
    CNNFeatureExtractor,
    PretrainedVisionFeatureExtractor,
    ERFNetFeatureExtractor
)

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


def _mlp(in_dim: int, out_dim: int, hidden: Tuple[int, int] = (512, 256)) -> nn.Sequential:
    """Two-hidden-layer ReLU MLP used for both the policy and the Q heads."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden[0]), nn.ReLU(),
        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        nn.Linear(hidden[1], out_dim)
    )


class SACActorCritic(nn.Module):
    """
    SAC policy and twin critics sharing the PPO vision backbone.

    The backbone is frozen and used only through `extract_visual_features`, so the replay
    buffer can store cached visual embeddings instead of raw frames and the encoder never
    re-runs during updates. Actor and critics each own their fusion layer rather than
    sharing one, which keeps critic gradients from leaking into the policy's representation.
    """
    def __init__(
        self,
        action_dim: int = 3,
        features_dim: int = 512,
        backbone_name: str = "lav",
        freeze_backbone: bool = True,
        use_pretrained: bool = True,
        weights_path: Optional[str] = None,
        policy_arch: str = "mlp"
    ):
        super().__init__()
        self.action_dim = action_dim
        self.policy_arch = str(policy_arch).lower()

        if use_pretrained:
            if backbone_name == "erfnet":
                self.encoder = ERFNetFeatureExtractor(
                    features_dim=features_dim, freeze_backbone=freeze_backbone, weights_path=weights_path
                )
            else:
                self.encoder = PretrainedVisionFeatureExtractor(
                    backbone_name=backbone_name, features_dim=features_dim,
                    freeze_backbone=freeze_backbone, weights_path=weights_path
                )
        else:
            self.encoder = CNNFeatureExtractor(in_channels=3, features_dim=features_dim)

        self.visual_dim = int(getattr(self.encoder, 'visual_feature_dim', features_dim))
        obs_dim = self.visual_dim + 1  # visual embedding + normalized speed scalar

        self.is_qwen = "qwen" in self.policy_arch
        if self.is_qwen:
            # Independent trunks: actor + twin critics, each a full Qwen transformer.
            self.actor = QwenSACActor(self.visual_dim, action_dim, self.policy_arch)
            self.q1 = QwenQCritic(self.visual_dim, action_dim, self.policy_arch)
            self.q2 = QwenQCritic(self.visual_dim, action_dim, self.policy_arch)
        else:
            self.actor = _mlp(obs_dim, 2 * action_dim)
            self.q1 = _mlp(obs_dim + action_dim, 1)
            self.q2 = _mlp(obs_dim + action_dim, 1)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad = False

        if not self.is_qwen:
            # Bias the initial policy toward rolling forward rather than braking, matching the
            # action mapping in CameraEasyCarlaEnv: throttle = (a0+1)/2, steer = a1, brake = a2.
            with torch.no_grad():
                self.actor[-1].bias.data[0] = 0.5
                self.actor[-1].bias.data[1] = 0.0
                self.actor[-1].bias.data[2] = -0.5

        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.q1.parameters()) * 2
        total_trainable = actor_params + critic_params
        print(f"SAC networks [{self.policy_arch}] | actor {actor_params/1e6:.1f}M + twin critics "
              f"{critic_params/1e6:.1f}M = {total_trainable/1e6:.1f}M trainable | visual_dim={self.visual_dim}")
        if self.is_qwen:
            # Adam keeps 2 moments per parameter; targets add another full critic pair.
            optim_gb = (total_trainable * 16 + critic_params * 4) / (1024 ** 3)
            print(f"  Estimated weights+optimizer+target footprint: ~{optim_gb:.1f} GB before activations.")

    def train(self, mode: bool = True):
        """
        Set training mode while pinning the frozen vision backbone to eval.

        Critical for replay-based learning: torch.no_grad() stops gradients but does NOT stop
        BatchNorm from updating its running statistics. Left in train mode the encoder would
        drift, and features already stored in the replay buffer would no longer match features
        computed for the same observation later - silently poisoning every Bellman backup.
        """
        super().train(mode)
        if getattr(self.encoder, 'freeze_backbone', False):
            if hasattr(self.encoder, 'backbone'):
                self.encoder.backbone.eval()
            else:
                self.encoder.eval()
        return self

    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Run the frozen vision backbone; the only place raw frames are ever touched."""
        return self.encoder.extract_visual_features(image)

    @staticmethod
    def _obs(visual_features: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Concatenate cached visual features with the normalized speed scalar."""
        speed_x = speed.float().view(visual_features.shape[0], 1) / 50.0
        return torch.cat([visual_features.float(), speed_x], dim=1)

    def sample_action(
        self, visual_features: torch.Tensor, speed: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a tanh-squashed Gaussian action with its exact log-probability.

        The tanh change-of-variables correction is what keeps log_prob valid after squashing;
        without it the entropy term and therefore the temperature tuning are both wrong.
        """
        if self.is_qwen:
            mean, log_std = self.actor(visual_features, speed)
        else:
            mean, log_std = self.actor(self._obs(visual_features, speed)).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            return action, torch.zeros(action.shape[0], device=action.device)

        dist = Normal(mean, std)
        x_t = dist.rsample()  # reparameterized: gradients flow through the sample
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + TANH_EPS)
        return action, log_prob.sum(dim=-1)

    def q_values(
        self, visual_features: torch.Tensor, speed: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Twin online Q estimates for the given state-action pair."""
        if self.is_qwen:
            return self.q1(visual_features, speed, action), self.q2(visual_features, speed, action)
        x = torch.cat([self._obs(visual_features, speed), action.float()], dim=1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def target_q_values(
        self, visual_features: torch.Tensor, speed: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Twin target-network Q estimates, used only inside the no-grad Bellman backup."""
        if self.is_qwen:
            return (self.q1_target(visual_features, speed, action),
                    self.q2_target(visual_features, speed, action))
        x = torch.cat([self._obs(visual_features, speed), action.float()], dim=1)
        return self.q1_target(x).squeeze(-1), self.q2_target(x).squeeze(-1)

    def soft_update_targets(self, tau: float = 0.005) -> None:
        """Polyak-average the online critics into the targets."""
        with torch.no_grad():
            for online, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, p_targ in zip(online.parameters(), target.parameters()):
                    p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def actor_parameters(self):
        """Trainable policy parameters."""
        return self.actor.parameters()

    def critic_parameters(self):
        """Trainable twin-critic parameters."""
        return list(self.q1.parameters()) + list(self.q2.parameters())
