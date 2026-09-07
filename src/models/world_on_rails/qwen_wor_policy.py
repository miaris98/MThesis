"""Qwen-Transformer World on Rails Policy - offline imitation-learning variant.

The Qwen decision transformer in src/models/transformer/qwen_transformer.py was built
for online PPO: it outputs a 3-dim raw control (steer/throttle/brake) directly and has
no PID controller, because a PPO actor learns the control mapping itself through reward.

This module adapts the same Qwen transformer building blocks (QwenTransformerBlock,
RMSNorm) into a *WoR-style* policy instead: predict a per-command waypoint trajectory
from (vision, speed, command, ego-frame route) tokens, exactly like
WorldOnRailsPolicy.SpatialQHead, and hand the result to the same PIDController used by
the CNN-based WoR policy. This keeps the two policies directly comparable - same
frozen vision encoder, same route conditioning, same PID conversion to vehicle
controls, same WorldOnRailsTrainer/WorldOnRailsDataset training pipeline - with only
the CNN+MLP head swapped for a Qwen transformer trunk.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.world_on_rails.wor_policy import PretrainedVisionEncoder, PIDController
from src.models.transformer.layers import RMSNorm, QwenTransformerBlock

_MODEL_SIZES = {
    "100m": dict(depth=12, embed_dim=768, num_heads=12, ffn_dim=2816),
    "500m": dict(depth=28, embed_dim=1024, num_heads=16, ffn_dim=4096),
    "900m": dict(depth=24, embed_dim=1536, num_heads=24, ffn_dim=6144),
}


class QwenWaypointTransformer(nn.Module):
    """Qwen transformer trunk that turns (vision, speed, route, command) tokens into
    per-command waypoint trajectories and rail Q-values, mirroring SpatialQHead's
    outputs but via self-attention over a short token sequence instead of a conv head.

    `num_vision_tokens` is how many vision tokens the trunk is sized for. With the
    encoder's 8x8 feature map fed through as 64 tokens the sequence is 68 long; with
    the legacy globally-pooled single vector it is 5. See QwenWorldOnRailsPolicy's
    `vision_grid` for why that difference dominates everything else in this module.
    """

    def __init__(self, embed_dim: int, depth: int, num_heads: int, ffn_dim: int,
                 num_commands: int = 6, num_rails: int = 9, num_vision_tokens: int = 64):
        super().__init__()
        self.num_commands = num_commands
        self.num_rails = num_rails
        self.num_vision_tokens = num_vision_tokens

        self.policy_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Attention is permutation-invariant and this module has no causal mask, so
        # without these the trunk cannot tell the speed token from the route token
        # from the command token by position - it can only hope speed_proj/route_proj/
        # cmd_embed happen to land in separable subspaces. `vision_pos` gives each
        # cell of the feature grid a distinct identity (which is what makes spatial
        # reasoning possible at all), and `type_embed` marks the four non-grid roles.
        self.vision_pos = nn.Parameter(torch.zeros(1, num_vision_tokens, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 4, embed_dim))  # policy, speed, route, cmd

        self.blocks = nn.ModuleList([
            QwenTransformerBlock(dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(embed_dim)

        self.waypoint_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_commands * 5 * 2)
        )
        self.rail_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_commands * num_rails)
        )
        nn.init.trunc_normal_(self.policy_token, std=0.02)
        nn.init.trunc_normal_(self.vision_pos, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, vision_tok: torch.Tensor, speed_tok: torch.Tensor,
                route_tok: torch.Tensor, cmd_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`vision_tok` is (B, N, D) with N >= 1 - one token per retained feature-map
        cell, or a single token in the legacy globally-pooled configuration."""
        B, N, _ = vision_tok.shape
        if N > self.vision_pos.shape[1]:
            raise ValueError(
                f"Trunk was built for at most {self.vision_pos.shape[1]} vision tokens "
                f"but received {N}. Rebuild the policy with a matching vision_grid."
            )

        p_tok = self.policy_token.expand(B, -1, -1) + self.type_embed[:, 0:1]
        v_tok = vision_tok + self.vision_pos[:, :N]
        s_tok = speed_tok + self.type_embed[:, 1:2]
        r_tok = route_tok + self.type_embed[:, 2:3]
        c_tok = cmd_tok + self.type_embed[:, 3:4]

        tokens = torch.cat([p_tok, v_tok, s_tok, r_tok, c_tok], dim=1)

        # No autocast is opened here. This module used to force float16 whenever the
        # input was on CUDA, which overrode the caller's choice (WorldOnRailsTrainer's
        # use_amp=False still ran the trunk in half precision) and applied fp16 to
        # act() at evaluation time, where precision matters more than throughput.
        # The ambient autocast context now governs, as it does for every other module.
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        policy_repr = tokens[:, 0]

        waypoints = self.waypoint_head(policy_repr).float().view(B, self.num_commands, 5, 2)
        rail_q = self.rail_head(policy_repr).float().view(B, self.num_commands, self.num_rails)

        return waypoints, rail_q


class QwenWorldOnRailsPolicy(nn.Module):
    """Drop-in replacement for WorldOnRailsPolicy: same constructor shape, same
    forward(rgb, speed, command, route) -> dict contract expected by
    WorldOnRailsTrainer/WorldOnRailsDataset, and the same act() -> (steer, throttle,
    brake) contract expected by WorldOnRailsAgent/eval_wor.py - only the decision head
    is a Qwen transformer trunk instead of SpatialQHead's conv+MLP."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        weights_path: Optional[str] = None,
        num_commands: int = 6,
        num_rails: int = 9,
        route_points: int = 4,
        model_size: str = "100m",
        vision_grid: int = 8
    ):
        super().__init__()
        self.num_commands = num_commands
        self.num_rails = num_rails
        self.route_points = route_points
        # Side length of the vision token grid. The encoder emits a (B, C, 8, 8) map
        # for the 256x256 input; `vision_grid=8` forwards every cell as its own token,
        # 4 pools it to 4x4 first, and 0 restores the original single globally-averaged
        # token. That last setting is what the first Qwen runs used, and it is the
        # single largest handicap in them: AdaptiveAvgPool2d((1,1)) makes the gradient
        # of the output with respect to every one of the 64 cells *identical*, so the
        # trunk is exactly blind to where anything is in the frame, while the CNN's
        # SpatialQHead convolves state into all 64 cells before pooling. It is kept
        # only so the ablation can be run.
        self.vision_grid = int(vision_grid)

        # Same frozen pretrained vision encoder as WorldOnRailsPolicy - training a
        # vision model stays out of scope, and --weights_path (e.g. the CARLA-domain
        # PCLA WoR checkpoint) plugs in identically here.
        self.encoder = PretrainedVisionEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            weights_path=weights_path
        )

        cfg = _MODEL_SIZES.get(str(model_size).lower(), _MODEL_SIZES["100m"])
        self.embed_dim = cfg["embed_dim"]

        # vision_grid=0 keeps the legacy 1x1 global average; any other value pools the
        # encoder map to that side length (a no-op when it already matches).
        self.num_vision_tokens = 1 if self.vision_grid <= 0 else self.vision_grid ** 2
        self.vision_pool = nn.AdaptiveAvgPool2d(
            (1, 1) if self.vision_grid <= 0 else (self.vision_grid, self.vision_grid)
        )
        self.vision_proj = nn.Linear(self.encoder.out_channels, self.embed_dim)
        self.speed_proj = nn.Linear(1, self.embed_dim)
        self.cmd_embed = nn.Embedding(num_commands, self.embed_dim)
        self.route_proj = nn.Linear(route_points * 2, self.embed_dim)

        self.trunk = QwenWaypointTransformer(
            embed_dim=self.embed_dim, depth=cfg["depth"], num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"], num_commands=num_commands, num_rails=num_rails,
            num_vision_tokens=self.num_vision_tokens
        )

        self.controller = PIDController()

        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        seq_len = self.num_vision_tokens + 4
        print(f"✓ Qwen-{str(model_size).upper()} WoR Decision Transformer initialized! "
              f"Trainable trunk parameters: {trunk_params:,} ({trunk_params / 1e6:.1f}M) | "
              f"sequence: {self.num_vision_tokens} vision + 4 state = {seq_len} tokens"
              f"{' (GLOBALLY POOLED - spatially blind ablation)' if self.vision_grid <= 0 else ''}")

    def _tokenize_state(
        self, feats: torch.Tensor, speed: torch.Tensor, command: torch.Tensor, route: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = feats.shape[0]

        # (B, C, H, W) -> (B, N, C) -> (B, N, embed_dim), one token per retained cell.
        # N is 1 in the legacy globally-pooled configuration, so the rest of the
        # pipeline is shape-identical either way.
        vis = self.vision_pool(feats).flatten(2).transpose(1, 2)
        vision_tok = self.vision_proj(vis)

        speed_tok = self.speed_proj(speed.view(-1, 1).float()).unsqueeze(1)

        if command.ndim > 1:
            command = command.argmax(dim=-1)
        command = command.long().clamp(0, self.num_commands - 1)
        cmd_tok = self.cmd_embed(command).unsqueeze(1)

        if route is None:
            route = torch.zeros(B, self.route_points, 2, device=feats.device, dtype=vision_tok.dtype)
        route_tok = self.route_proj(route.reshape(B, -1).float()).unsqueeze(1)

        return vision_tok, speed_tok, route_tok, cmd_tok, command

    def forward(
        self,
        rgb: torch.Tensor,
        speed: torch.Tensor,
        command: torch.Tensor,
        route: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(rgb)
        vision_tok, speed_tok, route_tok, cmd_tok, cmd_idx = self._tokenize_state(feats, speed, command, route)

        waypoints, rail_q = self.trunk(vision_tok, speed_tok, route_tok, cmd_tok)

        B = rgb.shape[0]
        batch_indices = torch.arange(B, device=rgb.device)
        selected_waypoints = waypoints[batch_indices, cmd_idx]
        selected_rail_q = rail_q[batch_indices, cmd_idx]

        return {
            "waypoints": waypoints,
            "rail_q": rail_q,
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
        """Mirrors WorldOnRailsPolicy.act() exactly, including the "no route -> no
        navigation intent" caveat: same PID conversion, same input handling."""
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
