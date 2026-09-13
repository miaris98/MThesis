"""Auxiliary prediction heads trained alongside the waypoint head.

WHY THESE EXIST
---------------
Roughly 60% of this project's CARLA Leaderboard 2.0 infractions are collisions with vehicles,
and another ~11% are collisions with layout. The waypoint head is not the problem - validation
ADE is 0.42 m, so the trajectories are fine. The policy plans a reasonable path and then drives
into things, which is a perception failure.

TransFuser++, trained on the same PDM-Lite dataset, optimises sixteen loss terms. This project
optimised one (waypoints) plus two geometry regularisers. The dataset ships eleven modalities
per frame - depth, front-view semantics, BEV semantics, 3D boxes with per-object position,
extent, yaw, speed and brake state - and the loader read two of them.

These heads consume the modalities that were being discarded. All of them predict *from the RGB
the policy already sees*; none adds an input sensor. That distinction is deliberate: depth and
semantics as supervision targets teach the frozen features' consumer where things are, whereas
feeding them in as inputs would hand the policy privileged information it will not have at
evaluation.

A caveat worth stating plainly: with the vision backbone frozen, these heads can only surface
structure the encoder already encodes. They cannot create it. The gain is real but bounded -
which is why the backbone was also moved to CARLA-pretrained weights (carla_encoder.py), whose
features were themselves trained under exactly this supervision.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# TransFuser++'s target-speed bins (team_code/config.py: self.target_speeds), in m/s. The first
# bin is exactly 0.0 and that is the point: it lets the policy command a full stop as a class
# rather than having to express "stopped" as waypoints that happen to coincide.
TARGET_SPEEDS: Sequence[float] = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)


class TargetSpeedHead(nn.Module):
    """Classifies the expert's target speed into `TARGET_SPEEDS` bins.

    WHY CLASSIFICATION RATHER THAN REGRESSION, AND WHY AT ALL
    ---------------------------------------------------------
    Longitudinal control here is currently implicit: the PID derives speed from the spacing of
    predicted waypoints. That representation degrades gracefully in the wrong direction - it can
    express "slow" but reaches "stopped" only in the limit, so a policy that should brake hard
    instead creeps. ~10% of this project's infractions are scenario timeouts and a further share
    are collisions that a decisive stop would have avoided.

    It is also the missing half of the ParkingExit failure (11.15). PDM-Lite solves that scenario
    by holding `steer = -1.0` at a standstill for ~28 frames and then accelerating out. Waypoint
    targets derived from pose deltas record that hold as (0, 0) - 63.4% of the scenario's frames
    carry a degenerate zero-motion target - so the manoeuvre is unrepresentable. A separate
    target-speed output decouples the longitudinal decision from the geometric one: "stationary"
    becomes class 0 while the steering is carried by the waypoints, which is expressible.

    Classification over regression follows TF++: the distribution of expert target speeds is
    multi-modal (stop / creep / cruise), and an L2 regressor on a multi-modal target predicts the
    mean of the modes - the one speed the expert never drives.
    """

    def __init__(self, in_dim: int, num_bins: int = len(TARGET_SPEEDS), hidden: int = 256):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bins", torch.tensor(TARGET_SPEEDS, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_bins),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, num_bins) logits."""
        return self.net(features)

    def expected_speed(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax-weighted bin average, in m/s.

        Used at inference. The argmax bin would be the harder commitment, but a soft average is
        what makes the head usable as a PID target without chattering between adjacent bins;
        because bin 0 is exactly 0.0, a confident stop still yields a near-zero speed.
        """
        return (logits.softmax(dim=-1) * self.bins).sum(dim=-1)


def two_hot_target_speed(speeds: torch.Tensor,
                         bins: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Two-hot encode continuous speeds over the bin centres (TF++ `use_twohot_target_speeds`).

    A one-hot label throws away where the value sits between two bins: 9.9 m/s and 8.1 m/s both
    collapse to the 8.0 bin, so the head cannot learn the difference and its expected value is
    biased toward bin centres. Two-hot splits the mass linearly between the two neighbouring
    bins, which keeps `expected_speed` unbiased for values between them.

    Returns (B, num_bins) summing to 1 along the last dim.
    """
    if bins is None:
        bins = torch.tensor(TARGET_SPEEDS, dtype=torch.float32, device=speeds.device)
    bins = bins.to(speeds.device)
    speeds = speeds.reshape(-1).clamp(min=float(bins[0]), max=float(bins[-1]))

    out = torch.zeros(speeds.shape[0], bins.shape[0], device=speeds.device, dtype=torch.float32)
    # Index of the upper neighbouring bin for each speed.
    upper = torch.searchsorted(bins, speeds.contiguous(), right=True).clamp(1, bins.shape[0] - 1)
    lower = upper - 1
    lo, hi = bins[lower], bins[upper]
    # Guard identical neighbours (only possible with a degenerate bin table) before dividing.
    span = (hi - lo).clamp(min=1e-6)
    w_hi = ((speeds - lo) / span).clamp(0.0, 1.0)
    out.scatter_(1, upper.unsqueeze(1), w_hi.unsqueeze(1))
    out.scatter_add_(1, lower.unsqueeze(1), (1.0 - w_hi).unsqueeze(1))
    return out


def target_speed_loss(logits: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy against the two-hot encoding of the expert's target speed."""
    target = two_hot_target_speed(speeds, bins=None)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class DenseDecoder(nn.Module):
    """Small upsampling decoder from the encoder's final feature map to a dense map.

    Shared by the depth and front-view semantic heads, which differ only in output channels and
    loss. Deliberately shallow: it sits on a *frozen* encoder, so its job is to read out
    structure the features already carry, not to build a depth estimator of its own. Making it
    deep would mostly add parameters that memorise the training towns.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 256, num_up: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        c = in_channels
        for _ in range(num_up):
            layers += [
                nn.Conv2d(c, hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
            c = hidden
            hidden = max(hidden // 2, 32)
        layers += [nn.Conv2d(c, out_channels, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class DepthHead(nn.Module):
    """Predicts a single-channel depth map from the frozen features.

    Depth is the most directly relevant discarded modality: "how far away is that car" is
    precisely the quantity a collision-avoidance policy needs and an ImageNet-style feature map
    is not organised around. Trained with an L1 loss on normalised inverse depth, which weights
    near surfaces - the ones that can be collided with - more heavily than the sky.
    """

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.decoder = DenseDecoder(in_channels, out_channels=1, hidden=hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat).sigmoid()

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 on [0, 1] depth. `target` is resized to the prediction rather than the reverse:
        upsampling a prediction to compare against a full-resolution label would let the loss be
        dominated by interpolation error that the network cannot act on."""
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return F.l1_loss(pred, target)


class SemanticHead(nn.Module):
    """Predicts front-view semantic classes from the frozen features.

    Cheaper and better-posed than BEV segmentation for a camera-only model: the labels live in
    the same image plane as the features, so no view transform is needed and the head cannot
    quietly become a monocular-BEV research project. Teaching the consumer of the features to
    separate road / vehicle / pedestrian / structure is the point.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.decoder = DenseDecoder(in_channels, out_channels=num_classes, hidden=hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat)

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy. `target` is (B, H, W) int64 and is nearest-resized to the logits -
        never bilinear, which would invent class indices that do not exist."""
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target.unsqueeze(1).float(), size=logits.shape[-2:],
                                   mode="nearest").squeeze(1)
        return F.cross_entropy(logits, target.long())


class AuxiliaryHeads(nn.Module):
    """Bundles the enabled auxiliary heads and returns their weighted loss.

    Each head is optional so a run can enable them one at a time: turning on four new losses
    simultaneously and observing a changed driving score would confound four hypotheses, which
    is the mistake 13.5 documents for optimiser defaults.
    """

    def __init__(
        self,
        feature_channels: int,
        state_dim: int,
        use_target_speed: bool = False,
        use_depth: bool = False,
        use_semantic: bool = False,
        num_semantic_classes: int = 7,
    ):
        super().__init__()
        self.use_target_speed = use_target_speed
        self.use_depth = use_depth
        self.use_semantic = use_semantic

        self.target_speed_head = TargetSpeedHead(state_dim) if use_target_speed else None
        self.depth_head = DepthHead(feature_channels) if use_depth else None
        self.semantic_head = SemanticHead(feature_channels, num_semantic_classes) \
            if use_semantic else None

    def forward(self, feat: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self.target_speed_head is not None:
            out["target_speed_logits"] = self.target_speed_head(state)
        if self.depth_head is not None:
            out["depth"] = self.depth_head(feat)
        if self.semantic_head is not None:
            out["semantic"] = self.semantic_head(feat)
        return out

    def losses(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Per-head losses, already weighted. Missing targets are skipped rather than zero-filled
        so a partially-loaded batch cannot silently train a head against zeros."""
        weights = weights or {}
        out: Dict[str, torch.Tensor] = {}
        if "target_speed_logits" in preds and targets.get("target_speed") is not None:
            out["target_speed"] = weights.get("target_speed", 1.0) * target_speed_loss(
                preds["target_speed_logits"], targets["target_speed"])
        if "depth" in preds and targets.get("depth") is not None:
            out["depth"] = weights.get("depth", 1.0) * DepthHead.loss(
                preds["depth"], targets["depth"])
        if "semantic" in preds and targets.get("semantic") is not None:
            out["semantic"] = weights.get("semantic", 1.0) * SemanticHead.loss(
                preds["semantic"], targets["semantic"])
        return out
