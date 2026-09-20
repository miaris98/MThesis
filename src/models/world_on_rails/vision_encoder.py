"""Vision encoders the WoR policy heads sit on.

Split out of `wor_policy.py`, which had grown past this repo's 500-line module limit once the
CARLA-pretrained path was added. The split is along a real seam rather than an arbitrary cut:
everything here answers "what turns pixels into a feature map", and nothing here knows what a
waypoint or a rail is.

`PretrainedVisionEncoder` (torchvision / ImageNet) and `CarlaPretrainedEncoder` (timm / CARLA,
in carla_encoder.py) are the two implementations; `build_vision_encoder` is the single entry
point that picks between them, and is what keeps the `cnn` and `qwen*` arms on identical vision.

Both names are re-exported from `wor_policy` so existing imports keep working.
"""
from typing import Optional
import os
import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedVisionEncoder(nn.Module):
    """
    Multi-Camera / Wide-RGB Vision Backbone initialized with ImageNet or CARLA-domain weights.
    Supports ResNet-18, ResNet-34, ResNet-50, and ConvNeXt.
    """
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        # Frozen by default: this project trains the driving policy on top of a
        # pretrained perception stack and treats training the vision model itself as
        # out of scope (the PPO/SAC path in config/training_config.py already
        # defaults the same way).
        self.freeze_backbone = freeze_backbone

        if "resnet18" in self.backbone_name:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and weights_path is None) else None
            base = models.resnet18(weights=weights)
            self.out_channels = 512
        elif "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if (pretrained and weights_path is None) else None
            base = models.resnet50(weights=weights)
            self.out_channels = 2048
        else:  # Default: resnet34
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if (pretrained and weights_path is None) else None
            base = models.resnet34(weights=weights)
            self.out_channels = 512

        # Extract spatial feature extractor (retaining 2D spatial dimensions)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if weights_path is not None:
            self._load_custom_weights(weights_path)

        if self.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Keeps a frozen backbone in eval mode.

        requires_grad=False and torch.no_grad() stop the *weights* changing, but
        BatchNorm still updates running_mean/running_var on every forward pass while
        in train mode. A "frozen" backbone would therefore keep drifting its
        normalization statistics toward the training data - which both contradicts
        using a fixed pretrained perception stack and makes its features
        non-deterministic across epochs.
        """
        return super().train(False) if self.freeze_backbone else super().train(mode)

    def _load_custom_weights(self, path: str):
        """Loads domain-specific CARLA weights (e.g., LAV, TransFuser++, WoR, TCP, Roach, or PCLA)."""
        if not os.path.exists(path):
            print(f"[Warning] Specified weights path does not exist: {path}")
            return

        print(f"--> Loading CARLA-domain pretrained vision weights from: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint.get("state_dict_bev", checkpoint)))

        prefixes = [
            "image_encoder.", "encoder.image_encoder.", "encoder.backbone.", "encoder.",
            "backbone.", "perception.", "bev_planner.", "rgb_encoder.", "camera_encoder.",
            "bev_encoder.", "model.", "net.", "policy.encoder.",
            # The original WoR paper's own CameraModel (pcla_agents/wor/rails/models/
            # main_model.py) names its ResNet34 backbone "backbone_wide" - its
            # checkpoints are otherwise a plain torchvision-style ResNet, so this
            # prefix alone is enough for a near-full match (unlike RegNet-backboned
            # sources like TransFuser++/garage_2, which can only ever partially match
            # regardless of prefix stripping since the layers themselves differ).
            "backbone_wide.", "image_model.backbone_wide."
        ]

        filtered = {}
        for k, v in state_dict.items():
            clean_k = k
            # Repeated passes, not one ordered pass. A compound prefix is only fully stripped
            # in a single pass when its outer component happens to sort before its inner one in
            # this list, and TCP's keys ("model.perception.conv1.weight") are the counterexample:
            # "perception." is checked before "model.", so one pass leaves "perception.conv1.weight"
            # and load_state_dict matches 0 of 218 tensors - silently, under strict=False, leaving
            # an ImageNet backbone behind a run config that claims CARLA weights. Terminates
            # because every strip shortens the key and no prefix is empty.
            stripped = True
            while stripped:
                stripped = False
                for prefix in prefixes:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                        stripped = True
            filtered[clean_k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        matched_keys = len(self.state_dict()) - len(missing)
        print(f"✓ Matched & loaded {matched_keys}/{len(self.state_dict())} vision backbone parameters from CARLA checkpoint.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB tensor of shape (B, 3, H, W) or (B, H, W, 3) in range [0, 255] or [0.0, 1.0].
        Returns:
            Spatial feature map of shape (B, out_channels, H/32, W/32).
        """
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1.0:
            x = x / 255.0

        x_norm = (x - self.mean) / self.std

        if self.freeze_backbone:
            with torch.no_grad():
                h = self.relu(self.bn1(self.conv1(x_norm)))
                h = self.maxpool(h)
                h = self.layer1(h)
                h = self.layer2(h)
                h = self.layer3(h)
                h = self.layer4(h)
        else:
            h = self.relu(self.bn1(self.conv1(x_norm)))
            h = self.maxpool(h)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            h = self.layer4(h)

        return h


def build_vision_encoder(
    backbone_name: str = "resnet34",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    weights_path: Optional[str] = None,
):
    """Selects the vision encoder both policy heads sit on.

    Exists so `cnn` and `qwen*` cannot drift apart on the one component they must share for
    their comparison to mean anything. A backbone difference between the two arms would make
    every head-vs-head number uninterpretable.

    `regnety_032` routes to the CARLA-pretrained TransFuser++ encoder and *requires*
    `weights_path` - there is deliberately no ImageNet fallback for it, because silently
    training on ImageNet features while the run config claims CARLA ones is precisely the class
    of undetectable mismatch that produced the worst bugs in this project.
    """
    if backbone_name.lower().startswith("regnet"):
        from src.models.world_on_rails.carla_encoder import CarlaPretrainedEncoder
        # Without weights_path this is the evaluation path, where the encoder weights come from
        # the run's own frozen_backbone.pth rather than the original TransFuser++ file (which
        # need not exist on an eval box). Never an ImageNet fallback either way - timm is always
        # constructed with pretrained=False.
        return CarlaPretrainedEncoder(
            weights_path=weights_path, arch=backbone_name, freeze_backbone=freeze_backbone,
            expect_external_weights=not weights_path)
    return PretrainedVisionEncoder(
        backbone_name=backbone_name, pretrained=pretrained,
        freeze_backbone=freeze_backbone, weights_path=weights_path)

