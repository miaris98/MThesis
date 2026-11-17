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
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.world_on_rails.wor_policy import (
    PretrainedVisionEncoder, PIDController, build_vision_encoder, tfpp_brake_override)
from src.models.transformer.layers import RMSNorm, QwenTransformerBlock
from src.models.world_on_rails.ray_geometry import (
    RAY_GEOMETRY_CHANNELS, append_ray_geometry)
from src.config.camera import DEFAULT_CROP_BOTTOM_FRAC
from src.models.world_on_rails.q_critic import QCriticHead, ValueHead, q_decode_speed, q_shift_lateral
from src.models.world_on_rails.world_model_decode import world_model_q
from src.models.world_on_rails.maneuver_rl import rl_maneuver

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


class MoEWaypointDelta(nn.Module):
    """2026-11-07, mixture-of-experts test: `num_experts` small two-layer experts on the policy token, top-k gated, whose output is
    ADDED to the trunk's own waypoint head. Every expert's last layer starts at zero, so a run resumed from E15 starts exactly at
    E15 and the experts learn corrections where the gate sends them. `last_aux` holds the Switch-Transformer load-balance term
    (E * sum_e f_e * P_e, 1.0 when balanced) for the trainer to add. Small expert counts are computed densely; large ones
    (e.g. 10,000 tiny experts) gather only the k selected experts per sample."""

    def __init__(self, dim: int, out_dim: int, num_experts: int, hidden: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts, self.hidden, self.top_k = int(num_experts), int(hidden), int(top_k)
        if not 1 <= self.top_k <= self.num_experts:   # (review finding) otherwise topk fails on the first forward
            raise ValueError(f"MoE top_k must be between 1 and the number of experts ({self.num_experts}), got {self.top_k}")
        self.gate = nn.Linear(dim, self.num_experts)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        self.w1 = nn.Parameter(torch.randn(self.num_experts, dim, self.hidden) * dim ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(self.num_experts, self.hidden))
        self.w2 = nn.Parameter(torch.zeros(self.num_experts, self.hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_experts, out_dim))
        self.last_aux: Optional[torch.Tensor] = None
        self.last_load: Optional[torch.Tensor] = None   # fraction of routed slots per expert (for logging expert use)
        self.last_top_i: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        logits = self.gate(x).float()
        self.last_logits = logits                                     # (B, E): raw router scores (the scenario-label loss reads them)
        probs = logits.softmax(-1)                                    # (B, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)                 # (B, k)
        weights = top_p / top_p.sum(-1, keepdim=True)
        if self.num_experts <= 256:                                   # dense: every expert on every sample, then pick k
            h = F.gelu(torch.einsum("bd,edh->beh", x, self.w1) + self.b1)
            y_all = torch.einsum("beh,eho->beo", h, self.w2) + self.b2  # (B, E, out)
            y = torch.gather(y_all, 1, top_i.unsqueeze(-1).expand(-1, -1, y_all.shape[-1]))
        else:                                                         # sparse: gather the k selected experts' weights
            h = F.gelu(torch.einsum("bd,bkdh->bkh", x, self.w1[top_i]) + self.b1[top_i])
            y = torch.einsum("bkh,bkho->bko", h, self.w2[top_i]) + self.b2[top_i]
        out = (weights.unsqueeze(-1) * y).sum(1)
        load = torch.zeros(self.num_experts, device=x.device).scatter_add_(
            0, top_i.reshape(-1), torch.ones(top_i.numel(), device=x.device)) / top_i.numel()
        self.last_load = load.detach()
        self.last_top_i = top_i.detach()   # (B, k): which experts each sample used (speed-head experts: routing split)
        self.last_aux = self.num_experts * (load * probs.mean(0)).sum()
        return out


# 2026-11-12, speed-head experts (TODO A81): the experts above, placed on the target-speed head instead of the waypoints, so they can change
# the stop / go decision that the obstacle and stall failures come from. The gate and the experts read the policy token plus two explicit
# ego-speed inputs (speed / 10 m/s and a standing flag), so "standing" and "moving" can route to different experts.
STANDING_MPS = 0.5


def speed_moe_inputs(policy_repr: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    """(B, E) policy token and (B,) ego speed in m/s -> (B, E + 2): the token, speed / 10 and a standing flag (speed < STANDING_MPS)."""
    s = speed.reshape(-1, 1).float()
    return torch.cat([policy_repr.float(), s / 10.0, (s < STANDING_MPS).float()], dim=-1)


def routing_split(top_i: torch.Tensor, standing: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Total-variation distance between the expert load of standing and of moving samples in one batch: 0 = both groups use the
    experts in the same proportions, 1 = they use disjoint experts. NaN when either group is empty (nothing to compare)."""
    standing = standing.reshape(-1).bool()
    if bool(standing.all()) or not bool(standing.any()):
        return torch.tensor(float("nan"), device=top_i.device)

    def load(rows: torch.Tensor) -> torch.Tensor:
        idx = rows.reshape(-1)
        return torch.zeros(num_experts, device=top_i.device).scatter_add_(0, idx, torch.ones(idx.numel(), device=top_i.device)) / idx.numel()

    return 0.5 * (load(top_i[standing]) - load(top_i[~standing])).abs().sum()


# 2026-11-13, TODO A83 (scenario-labelled router, after DriveMoE's skill-supervised action MoE): expert k is the one the router should pick for
# scenario class k, so the router's scores get a cross-entropy against the class index. Samples without a label (-1) are ignored.
# Returns None when no sample in the batch has a label, so the caller adds nothing.
def route_label_loss(gate_logits: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
    labels = labels.reshape(-1).long().to(gate_logits.device)
    keep = labels >= 0
    if not bool(keep.any()):
        return None
    return F.cross_entropy(gate_logits[keep].float(), labels[keep])


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
                 use_rail_q: bool = True, qk_norm: bool = False,
                 moe_experts: int = 0, moe_hidden: int = 8, moe_top_k: int = 2):
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
            QwenTransformerBlock(dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim, qk_norm=qk_norm)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(embed_dim)

        self.waypoint_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_commands * 5 * 2)
        )
        # 2026-11-07: optional mixture-of-experts correction added to waypoint_head's output (zero at init; see MoEWaypointDelta)
        self.moe = MoEWaypointDelta(embed_dim, num_commands * 5 * 2, moe_experts, moe_hidden, moe_top_k) if moe_experts > 0 else None
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
        self._last_vision_out = tokens[:, 1:1 + N]   # 2026-11-07: the vision tokens after the trunk, read by the auxiliary per-cell head

        wp = self.waypoint_head(policy_repr).float()
        if self.moe is not None:
            wp = wp + self.moe(policy_repr)
        waypoints = wp.view(B, self.num_commands, 5, 2)
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
        prev_frame: bool = False,
        route_windows: int = 0,
        moe_experts: int = 0,
        moe_hidden: int = 8,
        moe_top_k: int = 2,
        moe_aux_weight: float = 0.01,
        speed_moe_experts: int = 0,
        speed_moe_hidden: int = 32,
        speed_moe_top_k: int = 2,
        vision_moe_experts: int = 0,
        vision_moe_hidden: int = 16,
        vision_moe_top_k: int = 2,
        aux_cells: bool = False,
        aux_cell_weight: float = 0.1,
        pca_input: int = 0,
        attn_pool: int = 0,
        query_decoder: int = 0,
        qd_layers: int = 2,
        fuse_pca: int = 0,
        fuse_hw=None,
        extra_vision_channels: int = 0,
        win_aux: str = "",
        win_aux_weight: float = 0.5,
        privileged_boxes: int = 0,
        full_adapter: bool = False,
        full_aux_weight: float = 0.0,
        crop_bottom_frac: float = DEFAULT_CROP_BOTTOM_FRAC,
        use_path_head: bool = False,
        path_cascade: bool = False,
        path_points: int = 10,
        token_mode: str = "pool",
        s2d_patch: int = 3,
        token_mask: float = 0.0,
        qk_norm: bool = False,
        use_q_critic: bool = False,
        use_v_head: bool = False,
        use_reason_heads: bool = False,   # TODO A84: expert-reason / danger heads on the policy token (reason_heads.py)
        reason_detach: bool = False,      # TODO A84: the heads read a detached token (a classifier beside an unchanged policy)
        reason_cascade: bool = False      # TODO A84: the speed head also reads the heads' predictions (zero-init coupling)
    ):
        super().__init__()
        if float(token_mask) > 0 and (route_windows or attn_pool or query_decoder):   # (review finding) refuse at construction, before any run
            raise ValueError("token_mask applies to the plain vision grid only: it cannot be combined with route windows, attn_pool or query_decoder")
        self.num_commands = num_commands
        # TODO B67: Qwen3-style QK-norm in every trunk block (see QwenAttentionWithSkip). It adds tensors, so it is stamped in the
        # checkpoint config and rebuilt by load_wor_model; off by default, which keeps every existing checkpoint (E15 included) exact.
        self.qk_norm = bool(qk_norm)
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
        # 2026-11-08 idea 8: the privileged teacher reads M logged objects (one token each) instead of the camera
        self.privileged_boxes = int(privileged_boxes)
        if self.privileged_boxes:
            self.num_vision_tokens = self.privileged_boxes
        # 2026-11-07, route-window test: K extra vision tokens, each the 2x2-cell mean of the stride-32 map around a point on the route
        # (from the pooled cache's <prefix>.win<K>.f16, built by build_pooled_cache.py --route_windows K). The dataset packs them after the
        # 4x4 grid as extra columns: vision_features (B, C, gh, gw + K/gh). The trunk gets gh*gw + K tokens; slot k = k-th point along the route.
        self.route_windows = int(route_windows)
        if self.route_windows:
            assert min(gh, gw) > 0 and self.route_windows % gh == 0, "route_windows must be a multiple of the grid height"
            assert not use_ray_geometry, "route windows and ray geometry are separate arms"
            self.num_vision_tokens += self.route_windows
        self.moe_aux_weight = float(moe_aux_weight)
        self.moe_experts = int(moe_experts)
        # 2026-11-07, spatial tests:
        #   aux_cells: a per-token head on the first 16 vision tokens AFTER the trunk predicts each 4x4 cell's box occupancy and nearest distance
        #     (labels from the logged boxes; dropped at test time) - teaches the trunk to keep where-information in each token.
        #   pca_input K: the vision input is the whole stride-32 map compressed to K PCA channels (cached), one token per cell (vision_grid 9x24).
        #   attn_pool Q: the full-map cells are pooled to Q tokens by learned queries conditioned on the route token (route-attention pooling).
        self.aux_cells, self.aux_cell_weight = bool(aux_cells), float(aux_cell_weight)
        self.pca_input, self.attn_pool_q = int(pca_input), int(attn_pool)
        if self.pca_input:
            assert not (prev_frame or self.route_windows or use_ray_geometry), "the full-map input is its own arm"
        if self.attn_pool_q:
            assert self.pca_input, "attention pooling reads the full map (pca_input)"
            self.num_vision_cells = self.num_vision_tokens
            self.num_vision_tokens = self.attn_pool_q
        # 2026-11-09, TF++-style readout (Jaeger et al. 2023, transformer decoder instead of pooling): Q learned queries (NOT conditioned on the
        # route: the route is the shortcut being removed) run `qd_layers` decoder layers (self-attention, cross-attention into every cell of the
        # PCA full map with 2D sine-cosine positions, FFN). Their Q tokens are ADDED to the pooled vision_grid tokens of the same map; the output
        # projection starts at zero, so an arm starts exactly at the pooled model and the decoder has to earn its place.
        self.query_decoder, self.qd_layers = int(query_decoder), int(qd_layers)
        # 2026-11-09, two-encoder arm: the vision input is [this encoder's pooled vision_grid map | a SECOND encoder's PCA full map (fuse_pca
        # channels, fuse_hw cells)] flattened (WOR_FUSE_CACHE); the pooled part takes the usual path (vision_proj, so a resumed arm keeps its
        # trained input layer) and the query decoder reads the second encoder's map.
        self.fuse_pca = int(fuse_pca or 0)
        self.fuse_hw = tuple(int(v) for v in fuse_hw) if fuse_hw else None
        if self.fuse_pca:
            assert self.query_decoder and self.fuse_hw and not self.pca_input, "fuse_pca needs query_decoder, fuse_hw and a pooled primary input"
        if self.query_decoder:
            assert (self.pca_input or self.fuse_pca) and not (self.attn_pool_q or self.route_windows or full_adapter),                 "query_decoder reads a PCA full map (pca_input or fuse_pca) and adds its tokens to the pooled grid (no attn_pool / route windows / full_adapter)"
            self.num_vision_tokens += self.query_decoder
        self.vision_pool = nn.AdaptiveAvgPool2d(
            (1, 1) if min(gh, gw) <= 0 else (gh, gw)
        )
        # Ray geometry rides in as extra channels on each vision token rather than as extra
        # tokens, so the sequence length - and the attention cost - is unchanged. Both arms
        # take it the same way, which is what keeps the head-to-head about the trunk.
        self.use_rail_q = bool(use_rail_q)
        self.use_ray_geometry = bool(use_ray_geometry)
        # 2026-11-07 (test of the previous-frame idea): the map of frame t-1 is stacked on the map of frame t along the channel axis,
        # so the token count, the trunk and its vision_pos are unchanged; only the input width of vision_proj doubles. The dataset
        # builds the stacked tensor (WOR_PREV_FRAME=1); a frame with no earlier frame in the cache repeats its own map.
        self.prev_frame = bool(prev_frame)
        self.crop_bottom_frac = float(crop_bottom_frac)
        geo_ch = RAY_GEOMETRY_CHANNELS if self.use_ray_geometry else 0
        # TODO A54 (BevAD): token_mode "s2d" feeds the stride-16 pyramid level through a pixel-unshuffle (space-to-depth, patch p) instead of average-pooling the stride-32 map
        # to a 4x4 grid: nothing is averaged away (a lead car at brake onset is 0.4-0.55 of one stride-32 cell), each token carries p*p x 576 channels, and `token_mask` zeroes
        # that fraction of the lateral token columns in training (BevAD: the unmasked model attends to distant / irrelevant cells). Pass vision_grid = (H/16/p, W/16/p),
        # e.g. 6x16 for 288x768 at p = 3. Live encoder only: the pooled cache holds the stride-32 map.
        self.token_mode, self.s2d_patch, self.token_mask = str(token_mode), int(s2d_patch), float(token_mask)
        vis_ch = self.encoder.out_channels
        if self.token_mode == "s2d":
            assert hasattr(self.encoder, "forward_pyramid") and self.encoder.feature_reductions[-2] == 16, "token_mode s2d needs the CARLA timm encoder (stride-16 level)"
            vis_ch = self.encoder.feature_channels[-2] * self.s2d_patch ** 2
        if self.prev_frame:
            assert self.token_mode == "pool", "prev_frame is built for the pooled 4x4 tokens only"
            vis_ch = 2 * vis_ch
        if self.pca_input:
            vis_ch = self.pca_input
        # 2026-11-08: channels the pooled cache appends to every token after the encoder's (TF++ decoder statistics, window ground
        # coordinates); a resumed arm starts with zero weights on them (widen-on-resume).
        self.extra_vision_channels = int(extra_vision_channels)
        vis_ch += self.extra_vision_channels
        if self.privileged_boxes:
            from scripts.training.build_box_targets import BOX_FEATS
            vis_ch, geo_ch = BOX_FEATS, 0
        self.vision_proj = nn.Linear(vis_ch + geo_ch, self.embed_dim)
        if self.query_decoder:
            E = self.embed_dim
            self.qd_cell_proj = nn.Linear(self.fuse_pca or self.pca_input, E)
            self.qd_queries = nn.Parameter(torch.randn(1, self.query_decoder, E) * 0.02)
            self.qd = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(E, nhead=cfg["num_heads"], dim_feedforward=4 * E, dropout=0.0, batch_first=True, norm_first=True),
                num_layers=self.qd_layers, norm=nn.LayerNorm(E))
            self.qd_out = nn.Linear(E, E)
            nn.init.zeros_(self.qd_out.weight)
            nn.init.zeros_(self.qd_out.bias)
            self._qd_pos = {}
        # 2026-11-08 idea 12: a trainable per-cell adapter on the full stride-32 map (PCA input, pooled to the token grid INSIDE the model), with
        # an auxiliary per-cell obstacle head at full resolution; the adapter's last layer starts at zero, so the arm starts at the plain pooled map.
        self.full_adapter, self.full_aux_weight = bool(full_adapter), float(full_aux_weight)
        if self.full_adapter:
            assert self.pca_input and not self.attn_pool_q, "full_adapter reads the PCA full map (pca_input) and pools it to vision_grid"
            k = self.pca_input
            self.adapter = nn.Sequential(nn.Conv2d(k, k, 1), nn.GELU(), nn.Conv2d(k, k, 1))
            nn.init.zeros_(self.adapter[2].weight)
            nn.init.zeros_(self.adapter[2].bias)
            self.full_head = nn.Conv2d(k, 2, 1)
        self._full_logits = None
        self.return_policy_repr = False   # set by the trainer for feature distillation (idea 8 student)
        # 2026-11-08, idea 2 (path-relative obstacle targets on the window tokens; labels from the logged boxes, training only):
        #   "post": a head on the window tokens AFTER the trunk (like aux_cells);
        #   "pre":  an MLP on the window tokens BEFORE the trunk, whose detached prediction is embedded (zero-init) and added back to
        #           those tokens, so the trunk reads an explicit "obstacle here / how far" signal.
        self.win_aux, self.win_aux_weight = str(win_aux or ""), float(win_aux_weight)
        if self.win_aux:
            assert self.win_aux in ("post", "pre") and self.route_windows, "win_aux needs route windows ('post' or 'pre')"
            if self.win_aux == "post":
                self.win_head = nn.Linear(self.embed_dim, 2)
            else:
                self.win_pre_head = nn.Sequential(nn.Linear(self.embed_dim, 128), nn.GELU(), nn.Linear(128, 2))
                self.win_feed = nn.Linear(2, self.embed_dim)
                nn.init.zeros_(self.win_feed.weight)
                nn.init.zeros_(self.win_feed.bias)
        self._win_logits = None
        if self.attn_pool_q:
            self.cell_pos = nn.Parameter(torch.zeros(1, self.num_vision_cells, self.embed_dim))
            nn.init.trunc_normal_(self.cell_pos, std=0.02)
            self.pool_queries = nn.Parameter(torch.zeros(1, self.attn_pool_q, self.embed_dim))
            nn.init.trunc_normal_(self.pool_queries, std=0.02)
            self.attn_pool = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True)
        if self.aux_cells:
            self.cell_head = nn.Linear(self.embed_dim, 2)
        self.speed_proj = nn.Linear(1, self.embed_dim)
        self.cmd_embed = nn.Embedding(num_commands, self.embed_dim)
        self.route_proj = nn.Linear(route_points * 2, self.embed_dim)

        self.trunk = QwenWaypointTransformer(
            embed_dim=self.embed_dim, depth=cfg["depth"], num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"], num_commands=num_commands, num_rails=num_rails,
            num_vision_tokens=self.num_vision_tokens, use_rail_q=bool(use_rail_q), qk_norm=self.qk_norm,
            moe_experts=self.moe_experts, moe_hidden=int(moe_hidden), moe_top_k=int(moe_top_k)
        )
        self.moe_hidden, self.moe_top_k = int(moe_hidden), int(moe_top_k)
        # 2026-11-13, TODO A82: gated experts on every vision token, before the trunk. Each token is routed on its own content (per-cell
        # routing), and the experts' output is added residually. The last layer starts at zero, so a run resumed from E15 starts at its base.
        self.vision_moe_experts, self.vision_moe_hidden, self.vision_moe_top_k = int(vision_moe_experts), int(vision_moe_hidden), int(vision_moe_top_k)
        # Built from a forked RNG: the experts draw random init, and drawing it from the global stream would shift every later module's
        # init (target_speed_head, path head, ...) so that a model with experts and one without would not share a seed. (review finding)
        if self.vision_moe_experts > 0:
            with torch.random.fork_rng(devices=[]):
                self.vis_moe = MoEWaypointDelta(self.embed_dim, self.embed_dim, self.vision_moe_experts, self.vision_moe_hidden, self.vision_moe_top_k)
        else:
            self.vis_moe = None

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
        # 2026-11-12 (TODO A81): speed-head experts, a zero-init correction to the target-speed logits (a resumed arm starts exactly at its base).
        self.speed_moe_experts, self.speed_moe_hidden, self.speed_moe_top_k = int(speed_moe_experts), int(speed_moe_hidden), int(speed_moe_top_k)
        if self.speed_moe_experts > 0:
            assert self.target_speed_head is not None, "speed_moe_experts needs the target-speed head (--target_speed_loss_weight > 0)"
            self.speed_moe = MoEWaypointDelta(self.embed_dim + 2, self.target_speed_head.num_bins,
                                              self.speed_moe_experts, self.speed_moe_hidden, self.speed_moe_top_k)
        else:
            self.speed_moe = None

        # TODO A15: a space-indexed path head next to the time-indexed waypoints (TF++ path checkpoints, CarLLaVA / SimLingo / BevAD): `path_points` points of the expert's
        # shifted route in the ego frame, read from the policy token. With `path_cascade` the speed head is conditioned on the predicted path (AlignDrive): path_embed
        # is zero-initialised, so a cascade arm starts exactly as the plain head and learns the coupling.
        self.use_path_head, self.path_cascade, self.path_points = bool(use_path_head), bool(path_cascade and use_path_head), int(path_points)
        if self.use_path_head:
            self.path_head = nn.Sequential(nn.Linear(self.embed_dim, 256), nn.GELU(), nn.Linear(256, self.path_points * 2))
            nn.init.zeros_(self.path_head[-1].weight)
            with torch.no_grad():  # zero weights + the nominal straight path as bias: the head starts at 'drive straight' (x = 3.5, 5.5, .. 21.5 m, y = 0), not at the ego origin
                self.path_head[-1].bias.copy_(torch.tensor([[3.5 + 2.0 * k, 0.0] for k in range(self.path_points)]).reshape(-1))
            if self.path_cascade:
                self.path_embed = nn.Linear(self.path_points * 2, self.embed_dim)
                nn.init.zeros_(self.path_embed.weight)
                nn.init.zeros_(self.path_embed.bias)
        # TODO A71: Q(s, speed bin, lateral offset) on the policy token, trained on the counterfactual labels and read by act() under WOR_QCRITIC
        self.use_q_critic = bool(use_q_critic)
        self.q_head = QCriticHead(self.embed_dim) if self.use_q_critic else None
        # TODO A72: V(s) on the same token for the IQL / CQL bootstrap (offline_rl.py); forward then also returns the token for the target Q head
        self.use_v_head = bool(use_v_head)
        self.v_head = ValueHead(self.embed_dim) if self.use_v_head else None
        # TODO A84: built last and from a forked RNG, so every module above keeps the init it has without them
        self.use_reason_heads, self.reason_detach = bool(use_reason_heads), bool(reason_detach)
        self.reason_cascade = bool(reason_cascade)
        if self.reason_cascade and (not self.use_reason_heads or self.target_speed_head is None or self.target_speed_input != "policy"):
            raise ValueError("reason_cascade needs the reason heads and the target-speed head reading the policy token")
        self.reason_head, self.reason_embed = None, None
        if self.use_reason_heads:
            from src.models.world_on_rails.reason_heads import CASCADE_DIM, ReasonHeads, zero_linear
            with torch.random.fork_rng(devices=[]):
                self.reason_head = ReasonHeads(self.embed_dim, detach=self.reason_detach)
                self.reason_embed = zero_linear(CASCADE_DIM, self.embed_dim) if self.reason_cascade else None

        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        seq_len = self.num_vision_tokens + 4
        print(f"✓ Qwen-{str(model_size).upper()} WoR Decision Transformer initialized! "
              f"Trainable trunk parameters: {trunk_params:,} ({trunk_params / 1e6:.1f}M) | "
              f"sequence: {self.num_vision_tokens} vision + 4 state = {seq_len} tokens"
              f"{' (GLOBALLY POOLED - spatially blind ablation)' if self.vision_grid <= 0 else ''}"
              f"{' | QK-norm: on (TODO B67)' if self.qk_norm else ''}")

    def _qd_sincos(self, h: int, w: int, device) -> torch.Tensor:
        """(1, h*w, E) fixed 2D sine-cosine positions (half the channels encode the row, half the column), cached per map size."""
        key = (h, w, str(device))
        if key not in self._qd_pos:
            E = self.embed_dim
            q = E // 4
            freq = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32, device=device) / q))
            ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device), torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
            ay, ax = ys.reshape(-1, 1) * freq, xs.reshape(-1, 1) * freq
            pos = torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], dim=1)
            if pos.shape[1] < E:
                pos = torch.cat([pos, torch.zeros(pos.shape[0], E - pos.shape[1], device=device)], dim=1)
            self._qd_pos[key] = pos.unsqueeze(0)
        return self._qd_pos[key]

    def _tokenize_state(
        self, feats: torch.Tensor, speed: torch.Tensor, command: torch.Tensor, route: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = feats.shape[0]
        if self.privileged_boxes:   # (B, M, BOX_FEATS): one token per logged object; padding rows (valid flag 0) give zero tokens
            from scripts.training.build_box_targets import VALID_SLOT
            vision_tok = self.vision_proj(feats.float()) * feats[..., VALID_SLOT:VALID_SLOT + 1].float()
            speed_tok = self.speed_proj(speed.view(-1, 1).float()).unsqueeze(1)
            if command.ndim > 1:
                command = command.argmax(dim=-1)
            command = command.long().clamp(0, self.num_commands - 1)
            cmd_tok = self.cmd_embed(command).unsqueeze(1)
            if route is None:
                route = torch.zeros(B, self.route_points, 2, device=feats.device, dtype=vision_tok.dtype)
            route_tok = self.route_proj(route.reshape(B, -1).float()).unsqueeze(1)
            return vision_tok, speed_tok, route_tok, cmd_tok, command
        if self.full_adapter:   # (B, K, H, W) PCA full map -> + adapter -> per-cell aux logits -> pooled below
            feats = feats + self.adapter(feats.float()).to(feats.dtype)
            self._full_logits = self.full_head(feats.float())

        qd_tok = None
        qd_map = feats
        if self.fuse_pca:   # flat [pooled primary | second encoder's PCA map] -> the two maps
            gh, gw = self.vision_grid_hw
            n1 = self.vision_proj.in_features * gh * gw
            qd_map = feats[:, n1:].reshape(B, self.fuse_pca, self.fuse_hw[0], self.fuse_hw[1])
            feats = feats[:, :n1].reshape(B, -1, gh, gw)
        if self.query_decoder:   # (B, K, H, W) PCA full map -> Q decoder tokens (added after the pooled grid tokens below)
            Bq, Kq, Hq, Wq = qd_map.shape
            mem = self.qd_cell_proj(qd_map.flatten(2).transpose(1, 2).float()) + self._qd_sincos(Hq, Wq, qd_map.device)
            qd_tok = self.qd_out(self.qd(self.qd_queries.expand(Bq, -1, -1), mem))

        win = None
        if self.route_windows:  # cached (B, C, gh, gw + K/gh): the last K/gh columns are the route windows, row-major in route order
            gw = self.vision_grid_hw[1]
            feats, win = feats[..., :gw], feats[..., gw:]
            assert win.shape[-1] * win.shape[-2] == self.route_windows, f"expected {self.route_windows} route windows, got {tuple(win.shape)}"

        # (B, C, H, W) -> (B, N, C) -> (B, N, embed_dim), one token per retained cell.
        # N is 1 in the legacy globally-pooled configuration, so the rest of the
        # pipeline is shape-identical either way.
        pooled = self.vision_pool(feats)
        if self.use_ray_geometry:
            # After pooling, not before: the geometry has to describe the grid the trunk
            # actually receives, which is vision_grid_hw and not necessarily the encoder's.
            pooled = append_ray_geometry(pooled, self.crop_bottom_frac)
        vis = pooled.flatten(2)
        if win is not None:
            vis = torch.cat([vis, win.flatten(2)], dim=2)
        vis = vis.transpose(1, 2)
        vision_tok = self.vision_proj(vis)
        if qd_tok is not None:
            vision_tok = torch.cat([vision_tok, qd_tok.to(vision_tok.dtype)], dim=1)
        if self.win_aux == "pre":   # 2026-11-08: predict each window's obstacle state from its own token, feed the prediction back in
            n_cells = vision_tok.shape[1] - self.route_windows
            wtok = vision_tok[:, n_cells:]
            logits = self.win_pre_head(wtok)
            self._win_logits = logits
            vision_tok = torch.cat([vision_tok[:, :n_cells], wtok + self.win_feed(torch.sigmoid(logits).detach())], dim=1)

        speed_tok = self.speed_proj(speed.view(-1, 1).float()).unsqueeze(1)

        if command.ndim > 1:
            command = command.argmax(dim=-1)
        command = command.long().clamp(0, self.num_commands - 1)
        cmd_tok = self.cmd_embed(command).unsqueeze(1)

        if route is None:
            route = torch.zeros(B, self.route_points, 2, device=feats.device, dtype=vision_tok.dtype)
        route_tok = self.route_proj(route.reshape(B, -1).float()).unsqueeze(1)

        if self.attn_pool_q:  # 2026-11-07: Q learned queries, shifted by the route token, read the full map's cell tokens
            kv = vision_tok + self.cell_pos[:, :vision_tok.shape[1]]
            q = self.pool_queries.expand(B, -1, -1) + route_tok
            vision_tok, _ = self.attn_pool(q, kv, kv, need_weights=False)

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
        if self.token_mode == "s2d":
            if vision_features is not None:
                raise ValueError("token_mode s2d reads the stride-16 level: it cannot use the pooled stride-32 cache (--pooled_cache); train with the live encoder")
            # 2026-11-04: stop the encoder at the stride-16 level it reads (exact, tests/test_encoder_fast_path.py)
            lvl = self.encoder.forward_level(rgb, -2) if hasattr(self.encoder, "forward_level") else self.encoder.forward_pyramid(rgb)[-2]
            feats = F.pixel_unshuffle(lvl, self.s2d_patch)
        else:
            if self.route_windows and vision_features is None:
                raise NotImplementedError("route windows are read from the pooled cache; the live-encoder path (closed-loop act) is not built yet")
            feats = vision_features if vision_features is not None else self.encoder(rgb)
        if not self.training:
            self._last_feats = feats.detach()                   # TODO A73: the frozen map the world-model decode reads (world_model_decode.py)
        vision_tok, speed_tok, route_tok, cmd_tok, cmd_idx = self._tokenize_state(feats, speed, command, route)
        vis_aux = None
        if self.vis_moe is not None:   # TODO A82: each vision token gets its own routing; residual, so zero at init (before the token mask)
            Bv, Nv, Dv = vision_tok.shape
            delta = self.vis_moe(vision_tok.reshape(Bv * Nv, Dv)).view(Bv, Nv, Dv)
            vision_tok = vision_tok + delta.to(vision_tok.dtype)
            vis_aux = self.vis_moe.last_aux
        if self.training and self.token_mask > 0 and self.vision_grid_hw[0] > 0:
            # (review findings) the mask multiplies the plain grid's gh*gw tokens. Route windows, attn_pool and the query decoder change the
            # token count, so the mask cannot apply to them: refuse, rather than train with a mask that silently does not apply.
            if self.route_windows or self.attn_pool_q or self.query_decoder:
                raise ValueError("token_mask applies to the plain vision grid only: it cannot be combined with route windows, attn_pool or query_decoder")
            gh, gw = self.vision_grid_hw
            cols = (torch.rand(feats.shape[0], 1, gw, device=feats.device) >= self.token_mask).to(vision_tok.dtype)       # per sample: keep a lateral column or not
            vision_tok = vision_tok * cols.expand(-1, gh, -1).reshape(feats.shape[0], gh * gw, 1)

        waypoints, rail_q, policy_repr = self.trunk(vision_tok, speed_tok, route_tok, cmd_tok)
        if not self.training:
            self._last_policy_repr = policy_repr.detach()       # TODO A74: the token the maneuver head reads (maneuver_rl.py)

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
        if self.trunk.moe is not None and self.trunk.moe.last_aux is not None:
            out["moe_aux"] = self.trunk.moe.last_aux   # the trainer adds moe_aux_weight * this (load balance)
        if self.trunk.moe is not None and self.trunk.moe.last_logits is not None:
            out["moe_gate_logits"] = self.trunk.moe.last_logits   # TODO A83: the waypoint router's scores, for the scenario-label loss
        if vis_aux is not None:   # TODO A82: the vision experts' load balance joins the same term
            out["moe_aux"] = out["moe_aux"] + vis_aux if "moe_aux" in out else vis_aux
        if self.aux_cells:   # (B, 16, 2): occupancy logit and distance logit of each 4x4 cell, from the first 16 vision tokens after the trunk
            out["cell_logits"] = self.cell_head(self.trunk._last_vision_out[:, :16])
        if self.full_adapter:
            out["full_logits"] = self._full_logits
        if self.return_policy_repr:
            out["policy_repr"] = policy_repr
        if self.win_aux == "post":   # (B, M, 2): occupancy / distance logits of each window, from the window tokens after the trunk
            out["win_logits"] = self.win_head(self.trunk._last_vision_out[:, -self.route_windows:])
        elif self.win_aux == "pre":
            out["win_logits"] = self._win_logits
        # Same head, same reasoning as the CNN arm (wor_policy.py). It is fed the concatenated
        # state tokens rather than a fused vector because that is what this trunk has - keeping
        # the two arms' target-speed inputs as close as their architectures allow, so the
        # comparison stays about the trunk and not about what the speed head could see.
        if self.use_path_head:
            out["path"] = self.path_head(policy_repr).view(B, self.path_points, 2)
        if self.q_head is not None:
            out["q_sa"] = self.q_head(policy_repr)
        if self.v_head is not None:
            out["v_s"] = self.v_head(policy_repr)
            out["policy_repr"] = policy_repr
        if self.reason_head is not None:   # TODO A84
            out.update(self.reason_head(policy_repr))
        if self.target_speed_head is not None:
            if self.target_speed_input == "policy":
                ts_in = policy_repr
                if self.path_cascade:
                    ts_in = ts_in + self.path_embed(out["path"].reshape(B, -1))
                if self.reason_embed is not None:   # TODO A84 cascade: the speed decision reads the predicted cause / distance / danger
                    from src.models.world_on_rails.reason_heads import cascade_features
                    ts_in = ts_in + self.reason_embed(cascade_features(out)).to(ts_in.dtype)
            else:
                ts_in = torch.cat(
                    [speed_tok.squeeze(1), route_tok.squeeze(1), cmd_tok.squeeze(1)], dim=-1)
            out["target_speed_logits"] = self.target_speed_head(ts_in)
            if self.speed_moe is not None:   # TODO A81: experts on the stop / go decision; aux joins the trunk experts' load-balance term
                logits = out["target_speed_logits"]
                out["target_speed_logits"] = logits + self.speed_moe(speed_moe_inputs(policy_repr, speed)).to(logits.dtype)
                aux = self.speed_moe.last_aux
                out["moe_aux"] = out["moe_aux"] + aux if "moe_aux" in out else aux
                out["speed_moe_split"] = routing_split(self.speed_moe.last_top_i, speed.reshape(-1) < STANDING_MPS, self.speed_moe_experts)
        return out

    @torch.no_grad()
    def act(
        self,
        rgb: Union[np.ndarray, torch.Tensor],
        speed: Union[float, torch.Tensor],
        command: int = 2,
        device: str = "cuda",
        route: Optional[Union[np.ndarray, torch.Tensor]] = None,
        vision_features: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, float]:
        """`vision_features` (2026-11-09, two-encoder arms): the flat [pooled primary | second encoder PCA map] vector built by
        src/models/world_on_rails/fuse_encoder.py; it replaces self.encoder(rgb) in forward, exactly like the training cache.

        Mirrors WorldOnRailsPolicy.act() exactly, including the "no route -> no
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

        if self.fuse_pca and vision_features is None:
            raise RuntimeError("this checkpoint was trained on two encoders (fuse_pca): act() needs vision_features from fuse_encoder.py")
        if vision_features is not None:
            vision_features = vision_features.to(device)
        out = self.forward(rgb_tensor, speed_tensor, cmd_tensor, route_tensor, vision_features=vision_features)
        out = world_model_q(self, out)                                                           # TODO A73: no-op unless WOR_WORLDMODEL + WOR_QCRITIC
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
            # TODO A50 / A57 / A52: optional eval-time decision layer over the speed posterior (src/agents/safety_layer.py, env-switched;
            # None = this exact plain decode). last_debug feeds the A44 clip recorder.
            if not getattr(self, "_safety_ready", False):
                from src.agents.safety_layer import SafetyLayer
                self._safety, self._safety_ready, self._last_brake = SafetyLayer.from_env(), True, 0.0
            probs = out["target_speed_logits"][0].float().softmax(-1).cpu().numpy()
            target_speed_kmh, probs = q_decode_speed(self, out, probs, target_speed_kmh)        # TODO A71: no-op unless WOR_QCRITIC is set
            if self._safety is not None:
                tgt_mps, force_brake = self._safety.update(speed_mps, probs, self.target_speed_head.bins.cpu().numpy(), target_speed_kmh / 3.6,
                                                         self._last_brake, getattr(self, "_last_throttle", 0.0))
                target_speed_kmh = 3.6 * tgt_mps
            from src.models.world_on_rails.reason_heads import danger_gate_kmh
            target_speed_kmh = danger_gate_kmh(self, out, speed_mps, target_speed_kmh)                 # TODO A84: no-op unless WOR_DANGER_GATE is set
            self.last_debug = {"probs": probs, "waypoints": wps, "target_kmh": target_speed_kmh, "force_brake": bool(self._safety is not None and force_brake),
                               "danger_p": getattr(self, "_last_danger_p", None)}
            # Published so the eval HUD and telemetry report the live decision rather
            # than the unchanging default (see eval_wor.py's target_speed_kmh read).
            self.controller.target_speed = target_speed_kmh

        if getattr(self, "use_path_head", False) and "path" in out and os.environ.get("WOR_PATH_STEER") == "1":
            # TODO A15 eval switch: the lateral coordinate of every waypoint comes from the predicted path (np.interp at the waypoint's forward position, clamped at the
            # ends); the longitudinal spacing (speed) stays with the waypoint / speed heads. Logged once so a lane's log proves it ran.
            p_ = out["path"][0].float().cpu().numpy()
            o_ = np.argsort(p_[:, 0])
            wps = np.array(wps, dtype=np.float32, copy=True)
            wps[:, 1] = np.interp(wps[:, 0], p_[o_, 0], p_[o_, 1])
            if not getattr(self, "_path_steer_logged", False):
                print("[A15] steering from the predicted path (WOR_PATH_STEER=1)", flush=True)
                self._path_steer_logged = True

        if getattr(self, "_rl", None) is not None:                                               # TODO A74: the maneuver head (agent wrapper attaches it per route)
            wps, target_speed_kmh = rl_maneuver(self, wps, route, target_speed_kmh, speed_mps)
            self.controller.target_speed = target_speed_kmh
            if getattr(self, "last_debug", None) is not None:
                self.last_debug.update(target_kmh=target_speed_kmh, waypoints=wps, maneuver=self._rl.decisions[-1]["action"] if self._rl.decisions else None)
        wps = q_shift_lateral(self, wps, route)                                                  # TODO A71: no-op unless a lateral offset was chosen
        steer, throttle, brake = self.controller.control_from_waypoints(
            waypoints=wps,
            current_speed_kmh=current_speed_kmh,
            target_speed_kmh=target_speed_kmh
        )
        throttle, brake = tfpp_brake_override(throttle, brake, speed_mps, target_speed_kmh)
        if getattr(self, "last_debug", None) is not None and self.last_debug["force_brake"]:
            throttle, brake = 0.0, 1.0                                   # A50 contact reflex hold
        self._last_brake, self._last_throttle = float(brake), float(throttle)
        return steer, throttle, brake
