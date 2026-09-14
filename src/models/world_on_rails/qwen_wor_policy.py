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

from src.models.world_on_rails.wor_policy import (
    PretrainedVisionEncoder, PIDController, build_vision_encoder)
from src.models.transformer.layers import RMSNorm, QwenTransformerBlock
from src.models.world_on_rails.ray_geometry import (
    RAY_GEOMETRY_CHANNELS, append_ray_geometry)
from src.config.camera import DEFAULT_CROP_BOTTOM_FRAC

# Trunk sizes. The 10m/30m entries exist because the offline dataset is ~9,600 frames:
# at 100m the trunk carries roughly 11,000 trainable parameters per training sample and
# still *underfits* (its training loss is worse than a 1.8M conv head's), so the whole
# 100m/500m/900m range is on the wrong side of the size/data trade-off for this task.
# Sizing down is the cheapest experiment available and the one that recalibrates every
# other comparison, so it is a first-class option rather than a debug setting.
_MODEL_SIZES = {
    "10m": dict(depth=6, embed_dim=384, num_heads=6, ffn_dim=1024),
    "30m": dict(depth=8, embed_dim=512, num_heads=8, ffn_dim=1536),
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
                 num_commands: int = 6, num_rails: int = 9, num_vision_tokens: int = 64,
                 use_rail_q: bool = True):
        super().__init__()
        self.use_rail_q = use_rail_q
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
        # Built only when something trains it. With --q_loss_weight 0 (this project's
        # default) and PDM-Lite's all-zero target_q, this head receives no gradient at all
        # and act() never reads its output - dead weight feeding a `Q Loss` column that
        # nothing optimises.
        self.rail_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_commands * num_rails)
        ) if use_rail_q else None
        nn.init.trunc_normal_(self.policy_token, std=0.02)
        nn.init.trunc_normal_(self.vision_pos, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, vision_tok: torch.Tensor, speed_tok: torch.Tensor,
                route_tok: torch.Tensor, cmd_tok: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
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
        rail_q = (self.rail_head(policy_repr).float().view(B, self.num_commands, self.num_rails)
                  if self.rail_head is not None else None)

        # policy_repr is returned rather than discarded: it is the only tensor here that has
        # attended over the vision tokens, and the target-speed head needs exactly that.
        # See QwenWorldOnRailsPolicy.__init__'s target_speed_input.
        return waypoints, rail_q, policy_repr


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
        vision_grid: int = 8,
        use_target_speed: bool = False,
        target_speed_input: str = "policy",
        use_rail_q: bool = True,
        use_ray_geometry: bool = False,
        crop_bottom_frac: float = DEFAULT_CROP_BOTTOM_FRAC
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
        self.vision_grid = int(vision_grid[0]) if isinstance(vision_grid, (tuple, list)) \
            else int(vision_grid)

        # Same frozen pretrained vision encoder as WorldOnRailsPolicy - training a
        # vision model stays out of scope, and --weights_path (e.g. the CARLA-domain
        # PCLA WoR checkpoint) plugs in identically here.
        self.encoder = build_vision_encoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            weights_path=weights_path
        )

        cfg = _MODEL_SIZES.get(str(model_size).lower(), _MODEL_SIZES["100m"])
        self.embed_dim = cfg["embed_dim"]

        # vision_grid=0 keeps the legacy 1x1 global average; any other value pools the
        # encoder map to that side length (a no-op when it already matches).
        #
        # A square grid is only correct for a square input. Once the image is the source's own
        # 8:3 aspect (a 192x512 crop gives a 6x16 feature map), pooling to NxN squeezes 16
        # columns into 8 and stretches 6 rows into 8 - re-imposing in feature space exactly the
        # horizontal squash that cropping to the true aspect ratio was meant to remove. So the
        # grid is carried as (H, W): pass a single int for the legacy square behaviour, or an
        # (H, W) pair to keep the encoder's native rectangular layout.
        if isinstance(vision_grid, (tuple, list)):
            gh, gw = int(vision_grid[0]), int(vision_grid[1])
        else:
            gh = gw = self.vision_grid
        self.vision_grid_hw = (gh, gw)
        self.num_vision_tokens = 1 if min(gh, gw) <= 0 else gh * gw
        self.vision_pool = nn.AdaptiveAvgPool2d(
            (1, 1) if min(gh, gw) <= 0 else (gh, gw)
        )
        # Ray geometry rides in as extra channels on each vision token rather than as extra
        # tokens, so the sequence length - and the attention cost - is unchanged. Both arms
        # take it the same way, which is what keeps the head-to-head about the trunk.
        self.use_rail_q = bool(use_rail_q)
        self.use_ray_geometry = bool(use_ray_geometry)
        self.crop_bottom_frac = float(crop_bottom_frac)
        geo_ch = RAY_GEOMETRY_CHANNELS if self.use_ray_geometry else 0
        self.vision_proj = nn.Linear(self.encoder.out_channels + geo_ch, self.embed_dim)
        self.speed_proj = nn.Linear(1, self.embed_dim)
        self.cmd_embed = nn.Embedding(num_commands, self.embed_dim)
        self.route_proj = nn.Linear(route_points * 2, self.embed_dim)

        self.trunk = QwenWaypointTransformer(
            embed_dim=self.embed_dim, depth=cfg["depth"], num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"], num_commands=num_commands, num_rails=num_rails,
            num_vision_tokens=self.num_vision_tokens, use_rail_q=bool(use_rail_q)
        )

        self.controller = PIDController()

        # "policy" feeds the trunk's post-attention policy token, which has attended over
        # the vision tokens. "state" is the original speed+route+command concatenation, kept
        # so earlier runs stay reproducible - but under "state" this head is structurally
        # unable to see a red light or a lead vehicle, so it can only regress target speed
        # from route curvature, while its gradient still pulls on the shared projections.
        self.target_speed_input = str(target_speed_input).lower()
        if use_target_speed:
            from src.models.world_on_rails.aux_heads import TargetSpeedHead
            ts_dim = (self.embed_dim if self.target_speed_input == "policy"
                      else self.embed_dim * 3)
            self.target_speed_head = TargetSpeedHead(ts_dim)
        else:
            self.target_speed_head = None

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
        pooled = self.vision_pool(feats)
        if self.use_ray_geometry:
            # After pooling, not before: the geometry has to describe the grid the trunk
            # actually receives, which is vision_grid_hw and not necessarily the encoder's.
            pooled = append_ray_geometry(pooled, self.crop_bottom_frac)
        vis = pooled.flatten(2).transpose(1, 2)
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
        rgb: Optional[torch.Tensor],
        speed: torch.Tensor,
        command: torch.Tensor,
        route: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """`vision_features`, when given, replaces `self.encoder(rgb)` (rgb may then be
        None). See WorldOnRailsPolicy.forward's docstring - same cached-feature contract,
        same reasoning: the frozen backbone's output for an unaugmented frame never
        changes across epochs, so it need not be recomputed 50 times."""
        feats = vision_features if vision_features is not None else self.encoder(rgb)
        vision_tok, speed_tok, route_tok, cmd_tok, cmd_idx = self._tokenize_state(feats, speed, command, route)

        waypoints, rail_q, policy_repr = self.trunk(vision_tok, speed_tok, route_tok, cmd_tok)

        # B and the device come from feats, not rgb: rgb is None on the cached-feature path.
        B = feats.shape[0]
        batch_indices = torch.arange(B, device=feats.device)
        selected_waypoints = waypoints[batch_indices, cmd_idx]

        out = {
            "waypoints": waypoints,
            "selected_waypoints": selected_waypoints,
        }
        # Omitted entirely rather than emitted as zeros when the rail head is off: a
        # missing key is caught by the one consumer (waypoint_losses), whereas a zero
        # tensor would silently report a meaningless q_loss of 0.0 forever.
        if rail_q is not None:
            out["rail_q"] = rail_q
            out["selected_rail_q"] = rail_q[batch_indices, cmd_idx]
        # Same head, same reasoning as the CNN arm (wor_policy.py). It is fed the concatenated
        # state tokens rather than a fused vector because that is what this trunk has - keeping
        # the two arms' target-speed inputs as close as their architectures allow, so the
        # comparison stays about the trunk and not about what the speed head could see.
        if self.target_speed_head is not None:
            if self.target_speed_input == "policy":
                ts_in = policy_repr
            else:
                ts_in = torch.cat(
                    [speed_tok.squeeze(1), route_tok.squeeze(1), cmd_tok.squeeze(1)], dim=-1)
            out["target_speed_logits"] = self.target_speed_head(ts_in)
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
        """Mirrors WorldOnRailsPolicy.act() exactly, including the "no route -> no
        navigation intent" caveat: same PID conversion, same input handling.

        `speed` is in metres per second (the unit the dataset stores and the network
        trained on); the km/h conversion for the PID happens here. See the note in
        WorldOnRailsPolicy.act - this copy mirrored that method's units bug as
        faithfully as everything else, which is the recurring cost of a method whose
        contract is "identical to the other one" rather than a shared implementation."""
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

        # Longitudinal intent, when the policy has a head for it. Without this the
        # controller falls back to PIDController.target_speed - a CONSTANT 20 km/h that
        # nothing in this repo ever writes to - so the waypoints steered and the car
        # cruised at a fixed speed regardless of red lights, lead vehicles or turns.
        # The head was trained but its output was discarded here.
        target_speed_kmh = None
        if self.target_speed_head is not None and "target_speed_logits" in out:
            # expected_speed returns m/s (TARGET_SPEEDS is in m/s); the PID wants km/h,
            # and this is the one place that knows both - same convention as speed above.
            target_speed_kmh = 3.6 * float(
                self.target_speed_head.expected_speed(
                    out["target_speed_logits"])[0].item())
            # Published so the eval HUD and telemetry report the live decision rather
            # than the unchanging default (see eval_wor.py's target_speed_kmh read).
            self.controller.target_speed = target_speed_kmh

        steer, throttle, brake = self.controller.control_from_waypoints(
            waypoints=wps,
            current_speed_kmh=current_speed_kmh,
            target_speed_kmh=target_speed_kmh
        )
        return steer, throttle, brake
