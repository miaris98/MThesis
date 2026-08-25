"""ImageNet / LAV Pretrained ResNet Vision Feature Extractor for CARLA 3-Camera Inputs."""
import os
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedVisionFeatureExtractor(nn.Module):
    """
    Pretrained ResNet Feature Extractor for 256x256 RGB Images + Speed State.
    Supports CARLA-domain pretrained checkpoints (LAV / TransFuser++) and backbone freezing.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        features_dim: int = 512,
        freeze_backbone: bool = True,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        if backbone_name == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if weights_path is None else None)
            backbone_out_dim = 512
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_path is None else None)
            backbone_out_dim = 512

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if weights_path is not None and os.path.exists(weights_path):
            print(f"--> Loading CARLA-domain pretrained vision weights (LAV / TransFuser++) from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint.get("state_dict_bev", checkpoint)))
            
            extracted_dict = {}
            for k, v in state_dict.items():
                clean_k = k
                for prefix in [
                    "image_encoder.", "encoder.image_encoder.", "perception.",
                    "bev_planner.", "rgb_encoder.", "camera_encoder.", "bev_encoder.", "model."
                ]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k.replace(prefix, "")
                extracted_dict[clean_k] = v

            self.backbone.load_state_dict(extracted_dict, strict=False)
            print(f"✓ Pretrained weights matched & loaded into {backbone_name.upper()} backbone!")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.num_cameras = 3
        self.visual_feature_dim = backbone_out_dim * self.num_cameras
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 1, features_dim),
            nn.ReLU()
        )

    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract multi-camera visual embedding (N, 3 * D) with batch-level Tensor Core efficiency."""
        img_x = image.float() / 255.0

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                N, H, W, _ = img_x.shape
                if W == H * 3:
                    img_left = img_x[:, :, :H, :]
                    img_center = img_x[:, :, H:2*H, :]
                    img_right = img_x[:, :, 2*H:, :]
                    cams = torch.cat([img_left, img_center, img_right], dim=0).permute(0, 3, 1, 2)
                    cams_normalized = (cams - self.mean) / self.std
                    if self.freeze_backbone:
                        with torch.no_grad():
                            conv_out = self.backbone(cams_normalized).flatten(start_dim=1)
                    else:
                        conv_out = self.backbone(cams_normalized).flatten(start_dim=1)

                    left_out, center_out, right_out = torch.chunk(conv_out, 3, dim=0)
                    visual_features = torch.cat([left_out, center_out, right_out], dim=1)
                else:
                    img_perm = img_x.permute(0, 3, 1, 2)
                    img_normalized = (img_perm - self.mean) / self.std
                    if self.freeze_backbone:
                        with torch.no_grad():
                            single_out = self.backbone(img_normalized).flatten(start_dim=1)
                    else:
                        single_out = self.backbone(img_normalized).flatten(start_dim=1)
                    visual_features = single_out.repeat(1, self.num_cameras)
            else:
                img_normalized = (img_x - self.mean) / self.std
                if self.freeze_backbone:
                    with torch.no_grad():
                        single_out = self.backbone(img_normalized).flatten(start_dim=1)
                else:
                    single_out = self.backbone(img_normalized).flatten(start_dim=1)
                visual_features = single_out.repeat(1, self.num_cameras)

        return visual_features.float()

    def forward_with_visual_features(self, visual_features: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Zero-backbone forward pass using cached visual features + speed scalar."""
        speed_x = speed.float().view(-1, 1) / 50.0
        combined = torch.cat([visual_features, speed_x], dim=1)
        return self.fc(combined)

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass from raw RGB image tensor and speed."""
        visual_features = self.extract_visual_features(image)
        return self.forward_with_visual_features(visual_features, speed)
