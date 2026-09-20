"""Precomputes the frozen vision backbone's output for every frame in a dataset, once.

WHY THIS EXISTS
---------------
The backbone is frozen (no gradients, ever) and nothing upstream of it is randomized - no
crop/flip/color-jitter on RGB anywhere in this pipeline (see wor_dataset.py). So its output
for a given frame is identical on epoch 1, epoch 25 and epoch 50 of a run. Measured on this
project's regnety_032 CARLA-pretrained encoder, the backbone forward is 90.9% of a CNN head's
per-step compute and 50.5% of a Qwen head's - recomputing it 50 times is pure waste, and with
two arms training concurrently on the same frames under the same weights, it was being computed
redundantly twice per step as well.

This script runs that forward pass exactly once per frame and writes the result to a sidecar
file next to the source JPEG, using the same "sibling file, atomic rename" convention as
WorldOnRailsDataset's existing decoded-pixel cache. Training then reads the cached tensor
instead of running the encoder at all (see WorldOnRailsDataset.feature_cache_tag and
WorldOnRailsPolicy.forward's vision_features argument).

This is NOT an approximation. The encoder is deterministic in eval mode (CarlaPretrainedEncoder
forces this even when unfrozen - see its train() override), so the cached value is exactly what
`self.encoder(rgb)` would have computed live, modulo the fp16 storage roundtrip (see PRECISION
below). It is memoization, not distillation.

USAGE
-----
Must be run with the *exact* img_size/crop_bottom_frac/route_overlay/route_points/backbone/
weights_path a training run will use - these select the cache key (see
src.training.wor_dataset.feature_cache_path) and are read from the same CLI surface as
train_wor.py precisely so the two can't drift apart:

    python scripts/training/build_feature_cache.py --data_dir /workspace/dataset/wor_trajectories \\
        --backbone regnety_032 --weights_path /workspace/tfpp_pretrained/.../model_0030_0.pth \\
        --img_size 192x512 --crop_bottom_frac 0.25 --route_overlay 1 --batch_size 256

Resumable: frames whose cache file already exists are skipped (checked before any decode), so
an interrupted run just needs to be re-launched.

PRECISION
---------
Features are computed inside the same bfloat16 autocast context live training uses
(--use_amp's default), then stored as float16. float16 has a 10-bit mantissa versus bfloat16's
7, so the storage roundtrip is *more* precise than the arithmetic training already performs
inside autocast - this does not introduce a new source of error, and the dataset loader upcasts
back to float32 on read (see WorldOnRailsDataset.__getitem__). float16's ~65504 dynamic range
comfortably covers post-BatchNorm activations, which is why float16 rather than bfloat16 is the
on-disk format (numpy has no native bfloat16 dtype anyway).
"""
import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Repository root, so `src.*` and sibling `scripts.*` modules resolve whether this file is
# run directly (`python scripts/training/build_feature_cache.py`) or imported as `scripts.training.build_feature_cache`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.world_on_rails.vision_encoder import build_vision_encoder
from src.training.wor_dataset import WorldOnRailsDataset, feature_cache_path


def _parse_img_size(spec: str) -> Tuple[int, int]:
    if "x" in spec.lower():
        h, w = spec.lower().split("x")
        return int(h), int(w)
    n = int(spec)
    return n, n


class _PendingFramesDataset(Dataset):
    """Yields (preprocessed_rgb, feature_cache_path) for every frame not yet cached.

    Reuses the base dataset's own `_load_rgb`, not a reimplementation of it, so the pixels
    handed to the encoder here are bit-for-bit what training would have decoded - the same
    guarantee `_load_pdm_lite_sample` gives the live training path.
    """

    def __init__(self, base: WorldOnRailsDataset, feature_tag: str):
        self.base = base
        self.feature_tag = feature_tag
        pixel_tag = base.pixel_cache_tag()
        self.pending: List[Tuple[str, Optional[list]]] = []
        skipped_no_path = 0
        already_cached = 0
        for item in base.samples:
            rgb_path = item.get("rgb_path")
            if not rgb_path:
                skipped_no_path += 1
                continue
            cpath = feature_cache_path(rgb_path, pixel_tag, feature_tag)
            if os.path.exists(cpath):
                already_cached += 1
                continue
            self.pending.append((rgb_path, item.get("route")))
        print(f"--> {len(self.pending)} frames to encode, {already_cached} already cached"
              + (f", {skipped_no_path} skipped (no rgb_path)" if skipped_no_path else ""))

    def __len__(self) -> int:
        return len(self.pending)

    def __getitem__(self, idx: int):
        rgb_path, route = self.pending[idx]
        # Mirrors _load_pdm_lite_sample's call exactly: route_xy is a no-op inside _load_rgb
        # whenever route_overlay is off, so this is correct for both configurations.
        rgb = self.base._load_rgb(rgb_path, route_xy=route)
        pixel_tag = self.base.pixel_cache_tag()
        cpath = feature_cache_path(rgb_path, pixel_tag, self.feature_tag)
        return torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.uint8), cpath


def build_cache_for_dir(
    data_dir: str,
    backbone: str,
    weights_path: str,
    img_size: Tuple[int, int],
    crop_bottom_frac: float,
    route_overlay: bool,
    route_points: int,
    batch_size: int,
    num_workers: int,
    device: str,
    use_amp: bool,
    limit_frames: int = 0,
    use_augmented_camera: bool = False
) -> int:
    ds = WorldOnRailsDataset(
        data_dir=data_dir, is_train=True, cache_decoded=True, route_points=route_points,
        img_size=img_size, crop_bottom_frac=crop_bottom_frac, route_overlay=route_overlay,
        use_augmented_camera=use_augmented_camera
    )
    if ds.is_synthetic:
        print(f"[Warning] {data_dir} indexed as synthetic/empty - nothing to cache.")
        return 0

    encoder = build_vision_encoder(
        backbone_name=backbone, pretrained=True, freeze_backbone=True, weights_path=weights_path
    ).to(device)
    encoder.eval()

    pending = _PendingFramesDataset(ds, feature_tag=backbone)
    if limit_frames and limit_frames > 0:
        pending.pending = pending.pending[:limit_frames]
        print(f"--> --limit_frames set: capping this pass to {len(pending)} frames")
    if len(pending) == 0:
        return 0

    loader = DataLoader(
        pending, batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device == "cuda"), shuffle=False
    )

    written = 0
    t0 = time.time()
    last_report = t0
    with torch.no_grad():
        for rgb_batch, paths in loader:
            rgb_batch = rgb_batch.to(device, non_blocking=True)
            rgb_batch = rgb_batch.permute(0, 3, 1, 2).float().div_(255.0)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=use_amp and device == "cuda"):
                feats = encoder(rgb_batch)
            feats = feats.float().cpu().numpy().astype(np.float16)

            for i, cpath in enumerate(paths):
                # Same temp-file + atomic-rename pattern as _load_rgb's pixel cache: a worker
                # killed mid-write can never leave a corrupt file for training to load.
                tmp_path = f"{cpath}.tmp{os.getpid()}.npy"
                np.save(tmp_path, feats[i])
                os.replace(tmp_path, cpath)
            written += len(paths)

            if time.time() - last_report > 15:
                elapsed = time.time() - t0
                rate = written / elapsed
                remaining = (len(pending) - written) / max(rate, 1e-6)
                print(f"--> {written}/{len(pending)} cached ({rate:.1f}/s, "
                      f"~{remaining/60:.1f} min remaining)")
                last_report = time.time()

    elapsed = time.time() - t0
    print(f"--> Done: {written} features written in {elapsed:.1f}s "
          f"({written/max(elapsed,1e-6):.1f}/s)")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--val_data_dir", type=str, default=None,
                   help="Optional separate held-out directory - cached in the same pass if given.")
    p.add_argument("--backbone", type=str, default="regnety_032")
    p.add_argument("--weights_path", type=str, required=True)
    p.add_argument("--img_size", type=str, default="192x512", help="HxW, must match the training run")
    p.add_argument("--crop_bottom_frac", type=float, default=0.25)
    p.add_argument("--route_overlay", type=int, default=1)
    p.add_argument("--route_points", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Pure-inference batch size for the cache build - independent of the "
                        "--batch_size=32 pinned for training comparability, since this pass "
                        "never touches an optimizer.")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true",
                   help="Build in fp32 instead of the bf16 autocast training itself uses. Off by "
                        "default so the cached values match what live training actually computes.")
    p.add_argument("--limit_frames", type=int, default=0,
                   help="Cap how many pending frames to encode per directory (0 = no cap). For "
                        "a quick validation pass before committing to the full dataset.")
    p.add_argument("--use_augmented_camera", type=int, default=0,
                   help="Also encode the rgb_augmented/ recovery frames (1=yes). Must match the "
                        "training run's --use_augmented_camera: a run with the flag set indexes "
                        "those frames as samples, and feature_cache_tag makes a missing entry a "
                        "hard error, so a cache built without this flag crashes such a run on its "
                        "first augmented frame. Each augmented frame caches under its own "
                        "rgb_augmented/ path, and its route overlay is drawn from the *reprojected* "
                        "route the dataset stores for it, so the two never collide or cross-serve.")
    args = p.parse_args()

    img_size = _parse_img_size(args.img_size)
    dirs = [args.data_dir] + ([args.val_data_dir] if args.val_data_dir else [])
    total = 0
    for d in dirs:
        print(f"=== Building feature cache for {d} (backbone={args.backbone}) ===")
        total += build_cache_for_dir(
            data_dir=d, backbone=args.backbone, weights_path=args.weights_path,
            img_size=img_size, crop_bottom_frac=args.crop_bottom_frac,
            route_overlay=bool(args.route_overlay), route_points=args.route_points,
            batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
            use_amp=not args.no_amp, limit_frames=args.limit_frames,
            use_augmented_camera=bool(args.use_augmented_camera)
        )
    print(f"=== All done: {total} features written across {len(dirs)} director{'y' if len(dirs)==1 else 'ies'} ===")


if __name__ == "__main__":
    main()
