"""World on Rails (WoR) Neural Policy Architecture.

Implements the end-to-end sensorimotor driving model from:
"Learning to drive from a world on rails" (Chen et al., ICCV 2021)
and compatible with the PCLA (Pretrained CARLA Leaderboard Agents) framework.
"""
from typing import Dict, List, Optional, Tuple, Union
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            "bev_encoder.", "model.", "net.", "policy.encoder."
        ]

        filtered = {}
        for k, v in state_dict.items():
            clean_k = k
            for prefix in prefixes:
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
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


class SpatialQHead(nn.Module):
    """
    Predicts spatial Q-values / Value heatmap across discrete candidate rail waypoints.
    """
    def __init__(
        self,
        in_channels: int = 512,
        state_dim: int = 64,
        num_commands: int = 6,
        grid_size: Tuple[int, int] = (16, 16),
        num_rails: int = 9
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_commands = num_commands
        self.num_rails = num_rails

        # Fuse state (speed + command embedding) into feature maps
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + state_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Q-map decoder: outputs (B, num_commands, H_grid, W_grid)
        self.q_map_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_commands, kernel_size=1)
        )

        # Direct rail-path Q-value head: (B, num_commands, num_rails)
        self.rail_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_commands * num_rails)
        )

        # Waypoint trajectory offset head: (B, num_commands, 5, 2)
        self.waypoint_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_commands * 5 * 2)
        )

    def forward(
        self,
        features: torch.Tensor,
        state_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Spatial visual features (B, in_channels, H, W)
            state_emb: Embedded speed & command (B, state_dim)
        Returns:
            q_map: (B, num_commands, H_grid, W_grid)
            rail_q: (B, num_commands, num_rails)
            waypoints: (B, num_commands, 5, 2)
        """
        B, _, H, W = features.shape
        # Tile state embedding spatially
        state_spatial = state_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fused = torch.cat([features, state_spatial], dim=1)
        fused = self.fusion_conv(fused)

        # Q-map
        q_map = self.q_map_head(fused)
        if (H, W) != self.grid_size:
            q_map = F.interpolate(q_map, size=self.grid_size, mode="bilinear", align_corners=False)

        # Discrete Rail Q-values
        rail_q = self.rail_head(fused).view(B, self.num_commands, self.num_rails)

        # Continuous Waypoint offsets (x_forward, y_lateral)
        waypoints = self.waypoint_head(fused).view(B, self.num_commands, 5, 2)

        return q_map, rail_q, waypoints


class PIDController:
    """
    Translates selected waypoint trajectory / Q-value paths into smooth CARLA vehicle controls.
    """
    def __init__(
        self,
        kp_steer: float = 0.75,
        ki_steer: float = 0.05,
        kd_steer: float = 0.15,
        kp_speed: float = 1.0,
        ki_speed: float = 0.05,
        kd_speed: float = 0.10,
        target_speed: float = 20.0  # km/h
    ):
        self.kp_steer = kp_steer
        self.ki_steer = ki_steer
        self.kd_steer = kd_steer

        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.target_speed = target_speed

        self.steer_error_integral = 0.0
        self.steer_error_prev = 0.0

        self.speed_error_integral = 0.0
        self.speed_error_prev = 0.0

    def reset(self):
        self.steer_error_integral = 0.0
        self.steer_error_prev = 0.0
        self.speed_error_integral = 0.0
        self.speed_error_prev = 0.0

    def control_from_waypoints(
        self,
        waypoints: np.ndarray,
        current_speed_kmh: float,
        target_speed_kmh: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Computes (steer, throttle, brake) from predicted ego-frame waypoints.
        Waypoints shape: (N, 2) where (x_forward, y_lateral).
        """
        if target_speed_kmh is None:
            target_speed_kmh = self.target_speed

        # 1. Lateral Control (Pure Pursuit / PID on aim waypoint)
        aim_point = waypoints[min(2, len(waypoints) - 1)]
        dx = float(aim_point[0])
        dy = float(aim_point[1])

        # Heading angle error to target waypoint
        desired_angle = math.atan2(dy, max(dx, 0.5))
        steer_error = desired_angle

        self.steer_error_integral = np.clip(self.steer_error_integral + steer_error, -1.0, 1.0)
        steer_deriv = steer_error - self.steer_error_prev
        self.steer_error_prev = steer_error

        steer = (
            self.kp_steer * steer_error +
            self.ki_steer * self.steer_error_integral +
            self.kd_steer * steer_deriv
        )
        steer = float(np.clip(steer, -1.0, 1.0))

        # 2. Longitudinal Control (Speed PID)
        speed_error = (target_speed_kmh - current_speed_kmh) / 3.6  # convert to m/s
        self.speed_error_integral = np.clip(self.speed_error_integral + speed_error, -10.0, 10.0)
        speed_deriv = speed_error - self.speed_error_prev
        self.speed_error_prev = speed_error

        accel = (
            self.kp_speed * speed_error +
            self.ki_speed * self.speed_error_integral +
            self.kd_speed * speed_deriv
        )

        if accel >= 0:
            throttle = float(np.clip(accel, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-accel, 0.0, 1.0))

        return steer, throttle, brake


class WorldOnRailsPolicy(nn.Module):
    """
    Full World on Rails Sensorimotor Driving Policy.
    Maps front RGB cameras + Speed + Navigational Command -> Q-Values / Waypoints -> Vehicle Controls.
    """
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        weights_path: Optional[str] = None,
        num_commands: int = 6,
        state_dim: int = 64,
        grid_size: Tuple[int, int] = (16, 16),
        num_rails: int = 9,
        route_points: int = 4
    ):
        super().__init__()
        self.num_commands = num_commands
        self.num_rails = num_rails
        self.route_points = route_points

        # 1. Pretrained Multi-view Vision Encoder
        self.encoder = PretrainedVisionEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            weights_path=weights_path
        )

        # 2. Command, Speed & Route State Embedder.
        # The route encoder is what makes steering learnable at all: PDM-Lite leaves
        # the command enum at LANEFOLLOW on every frame, so cmd_embed is a constant
        # and image+speed alone cannot tell a left turn from a right one at a
        # junction - the model then correctly predicts straight everywhere. The
        # ego-frame route carries that intent (corr ~0.84 with the lateral target).
        self.cmd_embed = nn.Embedding(num_commands, 32)
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        self.route_mlp = nn.Sequential(
            nn.Linear(route_points * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(96, state_dim),
            nn.ReLU(inplace=True)
        )

        # 3. Spatial Q-Value and Waypoint Head
        self.q_head = SpatialQHead(
            in_channels=self.encoder.out_channels,
            state_dim=state_dim,
            num_commands=num_commands,
            grid_size=grid_size,
            num_rails=num_rails
        )

        # 4. Controller
        self.controller = PIDController()

    def embed_state(
        self,
        speed: torch.Tensor,
        command: torch.Tensor,
        route: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embeds speed (B, 1), discrete command (B, ) and the ego-frame route
        (B, route_points, 2) into a joint state embedding (B, state_dim).

        `route` is optional so callers that have no route planner still run, but they
        get a zero route, which leaves the policy without navigation intent - it will
        drive straight through junctions. Supply a real route wherever steering matters.
        """
        if command.ndim > 1:
            command = command.argmax(dim=-1)
        command = command.long().clamp(0, self.num_commands - 1)

        c_emb = self.cmd_embed(command)
        s_emb = self.speed_mlp(speed.view(-1, 1).float())

        B = c_emb.shape[0]
        if route is None:
            route = torch.zeros(B, self.route_points, 2, device=c_emb.device, dtype=c_emb.dtype)
        r_emb = self.route_mlp(route.reshape(B, -1).float())

        state = torch.cat([c_emb, s_emb, r_emb], dim=-1)
        return self.state_proj(state)

    def forward(
        self,
        rgb: torch.Tensor,
        speed: torch.Tensor,
        command: torch.Tensor,
        route: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            rgb: Image tensor (B, 3, H, W) or (B, H, W, 3)
            speed: Scalar vehicle speed tensor (B, 1) or (B, )
            command: High-level command indices (B, )
        Returns:
            Dict containing:
                'q_map': (B, num_commands, H_grid, W_grid)
                'rail_q': (B, num_commands, num_rails)
                'waypoints': (B, num_commands, 5, 2)
                'selected_waypoints': (B, 5, 2) based on active command
                'selected_rail_q': (B, num_rails) based on active command
        """
        # 1. Extract visual features
        feats = self.encoder(rgb)

        # 2. Embed speed, command and route
        state_emb = self.embed_state(speed, command, route)

        # 3. Predict Q-maps and waypoints
        q_map, rail_q, waypoints = self.q_head(feats, state_emb)

        # 4. Gather predictions corresponding to current command
        B = rgb.shape[0]
        if command.ndim > 1:
            cmd_idx = command.argmax(dim=-1).long()
        else:
            cmd_idx = command.long().clamp(0, self.num_commands - 1)

        batch_indices = torch.arange(B, device=rgb.device)
        selected_waypoints = waypoints[batch_indices, cmd_idx]      # (B, 5, 2)
        selected_rail_q = rail_q[batch_indices, cmd_idx]            # (B, num_rails)

        return {
            "q_map": q_map,
            "rail_q": rail_q,
            "waypoints": waypoints,
            "selected_waypoints": selected_waypoints,
            "selected_rail_q": selected_rail_q
        }

    @torch.no_grad()
    def act(
        self,
        rgb: Union[np.ndarray, torch.Tensor],
        speed: Union[float, torch.Tensor],
        command: int = 2,
        device: str = "cuda",
        route: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[float, float, float]:
        """
        Generates (steer, throttle, brake) controls for direct CARLA execution.

        `route` is the ego-frame planned route, (route_points, 2). Omitting it feeds
        a zero route, which leaves the policy with no navigation intent and makes it
        drive straight through junctions - so a caller that wants steering must
        provide one.
        """
        self.eval()
        if isinstance(rgb, torch.Tensor):
            rgb_tensor = rgb
            if rgb_tensor.ndim == 3:
                rgb_tensor = rgb_tensor.unsqueeze(0)
        else:
            try:
                rgb_tensor = torch.as_tensor(rgb, dtype=torch.float32)
            except Exception:
                rgb_tensor = torch.tensor(rgb.tolist(), dtype=torch.float32)
            if rgb_tensor.ndim == 3:
                rgb_tensor = rgb_tensor.unsqueeze(0)

        rgb_tensor = rgb_tensor.to(device)

        if isinstance(speed, (int, float)):
            speed_tensor = torch.tensor([[speed]], device=device, dtype=torch.float32)
            current_speed_kmh = float(speed)
        elif isinstance(speed, torch.Tensor):
            speed_tensor = speed.to(device).view(-1, 1).float()
            current_speed_kmh = float(speed_tensor.item())
        else:
            speed_tensor = torch.tensor([[float(speed)]], device=device, dtype=torch.float32)
            current_speed_kmh = float(speed)

        cmd_tensor = torch.tensor([command], device=device, dtype=torch.long)

        route_tensor = None
        if route is not None:
            route_tensor = torch.as_tensor(np.asarray(route, dtype=np.float32), device=device)
            if route_tensor.ndim == 2:
                route_tensor = route_tensor.unsqueeze(0)

        out = self.forward(rgb_tensor, speed_tensor, cmd_tensor, route_tensor)
        wps_tensor = out["selected_waypoints"][0].cpu()
        try:
            wps = wps_tensor.numpy()
        except Exception:
            wps = wps_tensor.tolist()

        steer, throttle, brake = self.controller.control_from_waypoints(
            waypoints=wps,
            current_speed_kmh=current_speed_kmh
        )
        return steer, throttle, brake
