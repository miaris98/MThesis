"""World on Rails (WoR) Neural Policy Architecture.

Implements the end-to-end sensorimotor driving model from:
"Learning to drive from a world on rails" (Chen et al., ICCV 2021)
and compatible with the PCLA (Pretrained CARLA Leaderboard Agents) framework.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-exported so `from ...wor_policy import PIDController` keeps working.
from src.models.world_on_rails.pid_controller import PIDController


# Re-exported so `from ...wor_policy import PretrainedVisionEncoder` (and build_vision_encoder)
# keeps working for every existing caller. The implementations moved to vision_encoder.py when
# this module outgrew the 500-line limit; see that module's docstring for the seam.
from src.models.world_on_rails.vision_encoder import (  # noqa: F401
    PretrainedVisionEncoder, build_vision_encoder)


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
        route_points: int = 4,
        pool_vision: bool = False,
        use_target_speed: bool = False
    ):
        super().__init__()
        self.num_commands = num_commands
        self.num_rails = num_rails
        self.route_points = route_points
        # Ablation only. Collapses the encoder's 8x8 feature map to 1x1 before
        # SpatialQHead sees it, giving the CNN exactly the spatially-blind input the
        # globally-pooled Qwen trunk was given. Without this, a CNN-beats-transformer
        # result cannot be attributed: it confounds architecture family with the fact
        # that only one of the two heads was ever shown where things are in the frame.
        self.pool_vision = pool_vision

        # 1. Pretrained Multi-view Vision Encoder
        self.encoder = build_vision_encoder(
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

        # 5. Optional target-speed classifier, reading the same fused state the waypoint head
        # does. Opt-in so it can be ablated against an otherwise identical run.
        if use_target_speed:
            from src.models.world_on_rails.aux_heads import TargetSpeedHead
            self.target_speed_head = TargetSpeedHead(state_dim)
        else:
            self.target_speed_head = None

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
        rgb: Optional[torch.Tensor],
        speed: torch.Tensor,
        command: torch.Tensor,
        route: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            rgb: Image tensor (B, 3, H, W) or (B, H, W, 3). May be None if
                `vision_features` is supplied instead (see below).
            speed: Scalar vehicle speed tensor (B, 1) or (B, )
            command: High-level command indices (B, )
            vision_features: Precomputed `self.encoder(rgb)` output, (B, C, H/32, W/32).
                The frozen backbone's output is a deterministic function of a frame that
                never changes across epochs (no RGB augmentation anywhere in this
                pipeline - see wor_dataset.py), so it can be computed once, cached to
                disk, and replayed here instead of re-run 50 times. Passing this skips
                `self.encoder(rgb)` entirely; `rgb` is then unused and may be None. Not
                used at all when None (the default), so every existing caller - and the
                already-running live training processes, which hold this function in
                memory and never re-read this file - is unaffected.
        Returns:
            Dict containing:
                'q_map': (B, num_commands, H_grid, W_grid)
                'rail_q': (B, num_commands, num_rails)
                'waypoints': (B, num_commands, 5, 2)
                'selected_waypoints': (B, 5, 2) based on active command
                'selected_rail_q': (B, num_rails) based on active command
        """
        # 1. Extract visual features (or reuse them, see vision_features above)
        feats = vision_features if vision_features is not None else self.encoder(rgb)
        if self.pool_vision:
            feats = F.adaptive_avg_pool2d(feats, (1, 1))

        # 2. Embed speed, command and route
        state_emb = self.embed_state(speed, command, route)

        # 3. Predict Q-maps and waypoints
        q_map, rail_q, waypoints = self.q_head(feats, state_emb)

        # 4. Gather predictions corresponding to current command
        # B and the device come from feats, not rgb: rgb is None on the cached-feature path.
        B = feats.shape[0]
        if command.ndim > 1:
            cmd_idx = command.argmax(dim=-1).long()
        else:
            cmd_idx = command.long().clamp(0, self.num_commands - 1)

        batch_indices = torch.arange(B, device=feats.device)
        selected_waypoints = waypoints[batch_indices, cmd_idx]      # (B, 5, 2)
        selected_rail_q = rail_q[batch_indices, cmd_idx]            # (B, num_rails)

        out = {
            "q_map": q_map,
            "rail_q": rail_q,
            "waypoints": waypoints,
            "selected_waypoints": selected_waypoints,
            "selected_rail_q": selected_rail_q
        }
        # Longitudinal intent as an explicit output rather than something the PID infers from
        # waypoint spacing. Spacing can express "slow" but only approaches "stopped" in the
        # limit, so a policy that should brake hard instead creeps - and "stationary while
        # steering", which is how PDM-Lite solves ParkingExit, is not expressible at all.
        if self.target_speed_head is not None:
            out["target_speed_logits"] = self.target_speed_head(state_emb)
        return out

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

        `speed` is in **metres per second**. That is the unit PDM-Lite's measurement
        files store and therefore the unit `speed_mlp` was trained on, so it is the
        network's input verbatim. The PID is the only consumer that wants km/h and the
        conversion happens here, at the one place that knows both.

        This used to hand the same scalar to both, which cannot be right for both: the
        two callers in the repo disagreed about which unit they were passing, so each
        was feeding one consumer correctly and the other a value 3.6x off. Passing m/s
        to the controller silently raises its effective target from 20 km/h to 72;
        passing km/h to the network puts its input 3.6x outside the training
        distribution. Neither raises an error.

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
            speed_mps = float(speed)
        elif isinstance(speed, torch.Tensor):
            speed_mps = float(speed.view(-1)[0].item())
        else:
            speed_mps = float(speed)

        speed_tensor = torch.tensor([[speed_mps]], device=device, dtype=torch.float32)
        current_speed_kmh = speed_mps * 3.6

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
