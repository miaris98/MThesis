"""NatureCNN-style visual feature extractor for multi-camera CARLA input."""
import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    NatureCNN-style architecture for extracting features from multi-camera RGB images + speed.
    """
    def __init__(self, in_channels: int = 3, features_dim: int = 512):
        super().__init__()
        self.num_cameras = 3
        self.freeze_backbone = False
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.visual_feature_dim = 12544 * self.num_cameras
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 1, features_dim),
            nn.ReLU()
        )

    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract multi-camera visual embeddings (N, 3 * D)."""
        img_x = image.float() / 255.0
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                N, H, W, _ = img_x.shape
                if W == H * 3:
                    img_left = img_x[:, :, :H, :]
                    img_center = img_x[:, :, H:2*H, :]
                    img_right = img_x[:, :, 2*H:, :]
                    cams = torch.cat([img_left, img_center, img_right], dim=0).permute(0, 3, 1, 2)
                    conv_out = self.conv(cams)
                    left_out, center_out, right_out = torch.chunk(conv_out, 3, dim=0)
                    visual_features = torch.cat([left_out, center_out, right_out], dim=1)
                else:
                    img_perm = img_x.permute(0, 3, 1, 2)
                    single_out = self.conv(img_perm)
                    visual_features = single_out.repeat(1, self.num_cameras)
            else:
                single_out = self.conv(img_x)
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
