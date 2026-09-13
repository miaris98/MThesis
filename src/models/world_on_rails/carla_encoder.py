"""CARLA-pretrained vision encoder, extracted from the TransFuser++ release.

WHY THIS EXISTS
---------------
`PretrainedVisionEncoder` in wor_policy.py is a torchvision ResNet carrying ImageNet
classification weights. Its own `_load_custom_weights` docstring already concedes the problem
this module solves: RegNet-backboned sources like TransFuser++ "can only ever partially match
regardless of prefix stripping since the layers themselves differ". So every attempt to give
that class CARLA weights silently fell back to mostly-ImageNet features.

That mattered more than it looked. ~60% of this project's Leaderboard 2.0 infractions are
collisions with vehicles - a perception failure, not a planning one. ImageNet features are
trained to answer "what object is this", not "where is that car and how far away". The policy
plans a fine trajectory (val ADE 0.42 m) and then drives into things.

WHAT THIS LOADS
---------------
`backbone.image_encoder.*` from carla_garage's released TransFuser++ checkpoint: a timm
`regnety_032` (18.0 M parameters) trained end-to-end on PDM-Lite / CARLA 0.9.15 - the same
dataset, simulator and 1024x512 camera geometry this project trains on. It was trained jointly
with depth, BEV-semantic, semantic and bounding-box decoders, so the features it produces are
shaped by exactly the spatial supervision our policy lacks.

The weights load with `missing=0, unexpected=0` into a freshly constructed timm model, which is
the check that this is the real architecture rather than a lucky partial match.

HONEST CAVEAT
-------------
TransFuser++ *initialises* `image_encoder` from ImageNet (`timm.create_model(..., pretrained=
True)`) and then trains end-to-end on CARLA. The released weights are therefore CARLA-trained
features that began from an ImageNet init. No purely-CARLA-from-scratch driving encoder is
publicly available. `pretrained=False` below is deliberate and load-bearing: it guarantees that
nothing here downloads ImageNet weights, and that every value in this module comes from the
CARLA checkpoint.

NORMALISATION
-------------
The ImageNet mean/std below is not an ImageNet *weight* - it is the input convention these CARLA
weights were trained under (`transfuser_utils.normalize_imagenet`, enabled by
`config.normalize_imagenet = True`). Feeding them anything else would silently degrade the
features, so it is matched exactly.
"""
import os
from typing import List, Optional

import torch
import torch.nn as nn

# The key TransFuser++ stores its image branch under.
_TFPP_IMAGE_PREFIX = "backbone.image_encoder."


class CarlaPretrainedEncoder(nn.Module):
    """A frozen, CARLA-trained timm feature extractor.

    Mirrors `PretrainedVisionEncoder`'s interface - `forward(x) -> (B, out_channels, H/32, W/32)`
    and an `out_channels` attribute - so it is a drop-in replacement for the heads that consume
    it. Unlike that class it also keeps the intermediate pyramid levels available via
    `forward_pyramid()`, which the auxiliary depth / semantic / BEV decoders need.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        arch: str = "regnety_032",
        freeze_backbone: bool = True,
        strict: bool = True,
        expect_external_weights: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment problem, not logic
            raise ImportError(
                "timm is required for the CARLA-pretrained encoder (pip install timm)") from exc

        self.arch = arch
        # Same attribute name PretrainedVisionEncoder exposes: the trainer records it in the
        # run config and telemetry, so both encoders have to answer to it or a run directory
        # stops being self-describing.
        self.backbone_name = arch
        self.freeze_backbone = freeze_backbone

        # pretrained=False is the whole point: no ImageNet weights are ever fetched. Every
        # parameter in this module comes from the CARLA checkpoint loaded below.
        self.encoder = timm.create_model(arch, pretrained=False, features_only=True)
        self.feature_channels: List[int] = list(self.encoder.feature_info.channels())
        self.feature_reductions: List[int] = list(self.encoder.feature_info.reduction())
        self.out_channels = self.feature_channels[-1]

        # Input convention of the CARLA weights - see module docstring.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if weights_path:
            self._load_carla_weights(weights_path, strict=strict)
        elif expect_external_weights:
            # Evaluation path: the encoder weights live in the checkpoint's own
            # frozen_backbone.pth (written once per run by the trainer) rather than in the
            # original TransFuser++ file, which need not be present on an eval box at all. The
            # architecture is built here and the loader fills it immediately afterwards.
            #
            # This stays safe because `pretrained=False` above means the fallback is *random*
            # vision, never ImageNet vision - and random vision is loud (the car will not drive)
            # whereas ImageNet vision would be quietly plausible. run_leaderboard_official.sh
            # additionally refuses to start when frozen_backbone.pth is missing (11.10).
            print("--> CARLA encoder built uninitialised; weights must be supplied by the "
                  "checkpoint loader (frozen_backbone.pth).")
        else:
            raise ValueError(
                "CarlaPretrainedEncoder needs either weights_path (training: the TransFuser++ "
                "checkpoint) or expect_external_weights=True (evaluation: weights arrive from "
                "the run's own frozen_backbone.pth). Refusing to silently build random vision.")

        if self.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def _load_carla_weights(self, weights_path: str, strict: bool):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"CARLA-pretrained weights not found at {weights_path}. This encoder has no "
                f"ImageNet fallback by design - refusing to run on randomly initialised vision.")

        print(f"--> Loading CARLA-pretrained vision weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        enc = {k[len(_TFPP_IMAGE_PREFIX):]: v
               for k, v in state.items() if k.startswith(_TFPP_IMAGE_PREFIX)}
        if not enc:
            # Also accept an already-extracted encoder-only file.
            enc = {k: v for k, v in state.items() if not k.startswith(("head.", "join.", "loss_"))}

        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        n_loaded = len(self.encoder.state_dict()) - len(missing)
        print(f"  matched {n_loaded}/{len(self.encoder.state_dict())} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

        # A partial match is the exact failure mode this module was written to end (see the
        # docstring's note about RegNet weights silently half-loading into a ResNet). Treat it
        # as fatal rather than training on a half-CARLA, half-random encoder and wondering why
        # the numbers moved.
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"CARLA encoder weights did not load cleanly: {len(missing)} missing, "
                f"{len(unexpected)} unexpected. Refusing to proceed - a partial load means the "
                f"features are not the ones that were validated. Pass strict=False only if you "
                f"know why the mismatch is benign.")

    def train(self, mode: bool = True):
        """Keeps a frozen backbone in eval mode.

        Same reasoning as PretrainedVisionEncoder.train(): requires_grad=False stops the weights
        moving but leaves BatchNorm updating its running statistics on every forward pass, which
        would drift a supposedly fixed perception stack toward our training data and make its
        features non-deterministic across epochs. RegNet is BatchNorm-heavy, so this matters here
        at least as much as it did there.
        """
        return super().train(False) if self.freeze_backbone else super().train(mode)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        return (x - self.mean) / self.std

    def forward_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """All pyramid levels, coarse-to-fine order as timm returns them.

        The auxiliary decoders (depth, semantic, BEV) upsample from these, which is why they are
        exposed rather than discarded like the base encoder does.
        """
        x = self._prepare(x)
        if self.freeze_backbone:
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Final feature map, (B, out_channels, H/32, W/32) - the interface the heads expect."""
        return self.forward_pyramid(x)[-1]


def default_tfpp_weights(root: str = "/workspace/tfpp_pretrained/pretrained_models",
                         variant: str = "all_towns", seed: int = 0) -> Optional[str]:
    """Conventional location of the released TransFuser++ checkpoints on the training box.

    `variant` is 'all_towns' or 'town13_withheld'. Our own WoR training uses all eight towns
    including Town12/13, so 'all_towns' matches the distribution this project already trains on;
    'town13_withheld' exists for the stricter held-out-town protocol.
    """
    path = os.path.join(root, variant, f"model_0030_{seed}.pth")
    return path if os.path.exists(path) else None
