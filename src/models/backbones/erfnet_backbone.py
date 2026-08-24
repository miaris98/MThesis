"""ERFNet Semantic Camera Feature Extractor for CARLA Perception."""
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    """Downsampling residual convolution block for ERFNet."""
    def __init__(self, ninput: int, noutput: int):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = torch.cat([self.conv(input_tensor), self.pool(input_tensor)], 1)
        output = self.bn(output)
        return F.relu(output)


class NonBottleneck1D(nn.Module):
    """1D factorized non-bottleneck residual block for efficient spatial feature extraction."""
    def __init__(self, chann: int, dropprob: float, dilated: int):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.conv3x1_1(input_tensor))
        output = self.bn1(F.relu(self.conv1x3_1(output)))
        output = F.relu(self.conv3x1_2(output))
        output = self.bn2(self.conv1x3_2(output))
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input_tensor)


class ERFNetFeatureExtractor(nn.Module):
    """ERFNet Camera Feature Extractor for CARLA Semantic Perception."""
    def __init__(
        self,
        features_dim: int = 512,
        freeze_backbone: bool = True,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for _ in range(5):
            self.layers.append(NonBottleneck1D(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for _ in range(2):
            self.layers.append(NonBottleneck1D(128, 0.3, 2))
            self.layers.append(NonBottleneck1D(128, 0.3, 4))
            self.layers.append(NonBottleneck1D(128, 0.3, 8))
            self.layers.append(NonBottleneck1D(128, 0.3, 16))

        if weights_path is not None and os.path.exists(weights_path):
            print(f"--> Loading ERFNet CARLA camera weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            self.load_state_dict(state_dict, strict=False)

        if self.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_cameras = 3
        self.visual_feature_dim = 128 * self.num_cameras
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 1, features_dim),
            nn.ReLU()
        )

    def _forward_erfnet(self, tensor_input: torch.Tensor) -> torch.Tensor:
        x = self.initial_block(tensor_input)
        for layer in self.layers:
            x = layer(x)
        return self.pool(x).flatten(start_dim=1)

    def extract_visual_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract multi-camera ERFNet feature vectors."""
        img_x = image.float() / 255.0
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                N, H, W, _ = img_x.shape
                if W == H * 3:
                    img_left = img_x[:, :, :H, :]
                    img_center = img_x[:, :, H:2*H, :]
                    img_right = img_x[:, :, 2*H:, :]
                    cams = torch.cat([img_left, img_center, img_right], dim=0).permute(0, 3, 1, 2)
                    if self.freeze_backbone:
                        with torch.no_grad():
                            conv_out = self._forward_erfnet(cams)
                    else:
                        conv_out = self._forward_erfnet(cams)
                    left_out, center_out, right_out = torch.chunk(conv_out, 3, dim=0)
                    visual_features = torch.cat([left_out, center_out, right_out], dim=1)
                else:
                    img_perm = img_x.permute(0, 3, 1, 2)
                    if self.freeze_backbone:
                        with torch.no_grad():
                            single_out = self._forward_erfnet(img_perm)
                    else:
                        single_out = self._forward_erfnet(img_perm)
                    visual_features = single_out.repeat(1, self.num_cameras)
            else:
                if self.freeze_backbone:
                    with torch.no_grad():
                        single_out = self._forward_erfnet(img_x)
                else:
                    single_out = self._forward_erfnet(img_x)
                visual_features = single_out.repeat(1, self.num_cameras)
        return visual_features.float()

    def forward_with_visual_features(self, visual_features: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        speed_x = speed.float().view(-1, 1) / 50.0
        combined = torch.cat([visual_features, speed_x], dim=1)
        return self.fc(combined)

    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        visual_features = self.extract_visual_features(image)
        return self.forward_with_visual_features(visual_features, speed)
