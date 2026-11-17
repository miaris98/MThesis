"""TODO A59 (S-125): build the pooled-token feature cache (src/training/pooled_feature_cache.py) for head-only CARLA training.

One pass of the frozen encoder over every frame the training run will index (same img_size / crop / overlay / route_key / recovery views), pooled to the head's token
grid and stored as one fp16 memmap: 48 KB per frame at 1512 x 4 x 4, 20 GB for 427k frames, instead of the 653 KB full map of build_feature_cache.py. Exact for the J-family
recipe (the head pools the map before any learnable layer; nothing random upstream of the frozen encoder). Resumable and shardable: run one process per GPU with
`--shard k --num_shards n --device cuda:k`; the first to arrive creates the files, frames already flagged in `<prefix>.done` are skipped.

    python scripts/training/build_pooled_cache.py --data_dir /workspace/dataset/wor_trajectories --out_dir /workspace/pooled \\
        --backbone regnety_032 --weights_path <tfpp model_0030_0.pth> --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 1 \\
        --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 --device cuda:0 --shard 0 --num_shards 2

Measured on 128 real Town01 frames of arm J (S-125): head output of the cached run vs live fp32 max |dwaypoint| 0.008 m (mean 0.00009), equal to the noise floor of two live runs at another batch split;
a bf16-built cache differs by 0.125 m max, equal to bf16's own batch-shape noise. Step time (A40 shared with 6 CARLA lanes): live encoder 2.08 s, cache 0.087 s.

then train with `train_wor.py ... --pooled_cache <the printed prefix>` (same flags, `--vision_grid 4`, frozen backbone, no colour augmentation).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.training.pooled_feature_cache import (DSTAT_CHANNELS, cache_files, cell_labels_from_boxes, grid_dstat_file, init_cache,  # noqa: E402
                                               label_files, pca_file, rel_key, route_window_fracs, route_window_points, side_window_points,
                                               window_coord_channels, window_file, window_label_valid_file, window_labels_from_boxes,
                                               window_side_files)

STRAIGHT_ROUTE = [[2.5, 0.0], [8.5, 0.0], [15.5, 0.0], [21.5, 0.0]]   # the fixed-window control places its K windows on this line


def pool_and_store(feats: torch.Tensor, rows, mm, done, grid) -> None:
    """(B, C, H, W) encoder output -> (B, C, gh, gw) fp16 rows of the memmap, flagged done. The pooling is the head's own `AdaptiveAvgPool2d(grid)` on the float32 map."""
    pooled = F.adaptive_avg_pool2d(feats.float(), tuple(grid)).half().cpu().numpy()
    rows = np.asarray(rows)
    mm[rows] = pooled
    done[rows] = 1


def store_route_windows(feats: torch.Tensor, rows, fracs: torch.Tensor, wm) -> None:
    """2026-11-07, route-window arm: (B, C, H, W) encoder map + (B, K, 2) image fractions (v/H, u/W) of K route points ->
    (B, C, K) fp16 rows of the window memmap. Window k = the mean of the 2x2 cells nearest the k-th route point."""
    f = feats.float()
    B, C, H, W = f.shape
    pair = F.avg_pool2d(f, kernel_size=2, stride=1)                    # (B, C, H-1, W-1): entry (r, c) = mean of cells r..r+1, c..c+1
    fr = fracs.to(f.device).float()
    r0 = torch.floor(fr[..., 0] * H - 0.5).long().clamp(0, H - 2)
    c0 = torch.floor(fr[..., 1] * W - 0.5).long().clamp(0, W - 2)
    idx = (r0 * (W - 1) + c0).unsqueeze(1).expand(-1, C, -1)          # (B, C, K)
    win = torch.gather(pair.flatten(2), 2, idx)
    wm[np.asarray(rows)] = win.half().cpu().numpy()


def frame_boxes(rgb_path: str):
    """(boxes, offset_inv) of one frame, or (None, None) when unreadable. A recovery-camera frame (rgb_augmented/) gets the inverse of its
    camera offset, which re-expresses the boxes in that camera's frame, as its route (wor_dataset._camera_offset_matrix)."""
    import gzip
    import json
    from src.training.wor_dataset import _camera_offset_matrix
    route_dir, stem = os.path.dirname(os.path.dirname(rgb_path)), os.path.splitext(os.path.basename(rgb_path))[0]
    try:
        with gzip.open(os.path.join(route_dir, "boxes", f"{stem}.json.gz"), "rt") as f:
            boxes = json.load(f)
        offset_inv = None
        if os.path.basename(os.path.dirname(rgb_path)) == "rgb_augmented":
            with gzip.open(os.path.join(route_dir, "measurements", f"{stem}.json.gz"), "rt") as f:
                m = json.load(f)
            offset_inv = np.linalg.inv(_camera_offset_matrix(m.get("augmentation_translation", 0.0), m.get("augmentation_rotation", 0.0)))
        return boxes, offset_inv
    except Exception:
        return None, None


def frame_box_labels(rgb_path: str, mirror: bool):
    """Per-cell box labels of one frame (2, 4, 4) and a validity flag (see frame_boxes for the recovery-camera frames)."""
    boxes, offset_inv = frame_boxes(rgb_path)
    if boxes is None:
        return np.zeros((2, 4, 4), dtype=np.float32), 0
    return cell_labels_from_boxes(boxes, offset_inv, mirror), 1


class _Pending(Dataset):
    """One frame for the encoder pass: pixels exactly as training loads them (optionally mirrored left-right, after the overlay is drawn),
    the image fractions of its K route windows (on its own route, mirrored with the image), and its per-cell box labels.
    2026-11-08: with side_offset > 0 also the 2K side windows (one lane left / right); the window ground points (route then side) and,
    with window_labels, their path-relative obstacle targets; fractions are of the cropped frame when crop_bottom_frac > 0."""

    def __init__(self, ds, rows, route_windows: int = 0, flip: bool = False, labels: bool = False, side_offset: float = 0.0,
                 window_labels: bool = False, crop_bottom_frac: float = 0.0):
        self.ds, self.rows, self.route_windows, self.flip, self.labels = ds, rows, int(route_windows), bool(flip), bool(labels)
        self.side_offset, self.window_labels, self.crop = float(side_offset), bool(window_labels), float(crop_bottom_frac)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        it = self.ds.samples[r]
        rgb = self.ds._load_rgb(it["rgb_path"], route_xy=it.get("route"))  # the very function training calls: identical pixels
        route = it.get("route") or []
        if self.flip:  # mirror image and route together (the dataset mirrors waypoints and swaps left/right commands at train time)
            rgb = rgb[:, ::-1]
            route = [[float(p[0]), -float(p[1])] for p in route]
        rgb = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.uint8)
        K = self.route_windows
        if K:
            pts = route_window_points(route, K)
            if self.side_offset > 0:
                pts = np.concatenate([pts, side_window_points(route, K, self.side_offset)], axis=0)
            fr = route_window_fracs(route, K, self.crop, points=pts)
        else:
            pts, fr = np.zeros((1, 2)), np.zeros((1, 2), dtype=np.float32)
        lab, valid = frame_box_labels(it["rgb_path"], self.flip) if self.labels else (np.zeros((2, 4, 4), dtype=np.float32), 0)
        wlab, wvalid = np.zeros((2, len(pts)), dtype=np.float32), 0
        if self.window_labels and K:
            assert not self.flip, "window labels are built on unmirrored frames"
            boxes, offset_inv = frame_boxes(it["rgb_path"])
            if boxes is not None:
                wlab, wvalid = window_labels_from_boxes(pts, boxes, offset_inv), 1
        return (r, rgb, torch.as_tensor(fr), torch.as_tensor(lab), valid, torch.as_tensor(window_coord_channels(pts)),
                torch.as_tensor(wlab), wvalid)


def fit_pca(encoder, ds, rows, k: int, device: str, batch_size: int, num_workers: int, use_amp: bool):
    """Mean (C,) and the top-k principal directions (k, C) of the frozen stride-32 map's cell vectors, from the given frames."""
    loader = DataLoader(_Pending(ds, rows), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    s1 = s2 = None
    n = 0
    with torch.no_grad():
        for item in loader:
            x = item[1].to(device).permute(0, 3, 1, 2).float().div_(255.0)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                f = encoder(x)
            v = f.float().permute(0, 2, 3, 1).reshape(-1, f.shape[1]).double()
            s1 = v.sum(0) if s1 is None else s1 + v.sum(0)
            s2 = v.T @ v if s2 is None else s2 + v.T @ v
            n += v.shape[0]
    mean = s1 / n
    cov = s2 / n - torch.outer(mean, mean)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)[:k]
    explained = float(evals[order].sum() / evals.clamp(min=0).sum())
    return mean.float().cpu(), evecs[:, order].T.float().cpu(), explained, n


def _size(spec):
    h, w = spec.lower().split("x")
    return int(h), int(w)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--backbone", default="regnety_032")
    p.add_argument("--weights_path", default=None, help="the TF++ release checkpoint (model_0030_0.pth) the encoder was extracted from")
    p.add_argument("--frozen_backbone", default=None, help="alternative to --weights_path: the training run's own frozen_backbone.pth (the exact encoder weights of that run)")
    p.add_argument("--random_init", action="store_true", help="image-use control: randomly initialised encoder instead of pretrained weights (2026-10-15)")
    p.add_argument("--img_size", default="288x768")
    p.add_argument("--crop_bottom_frac", type=float, default=0.0)
    p.add_argument("--route_overlay", type=int, default=1)
    p.add_argument("--route_points", type=int, default=4)
    p.add_argument("--route_key", default="route", choices=["route", "route_original"])
    p.add_argument("--use_augmented_camera", type=int, default=0)
    p.add_argument("--grid", default="4", help="token grid of the head, 'N' or 'HxW' (= train_wor.py --vision_grid)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit_frames", type=int, default=0)
    p.add_argument("--route_windows", type=int, default=0,
                   help="2026-11-07: also write <prefix>.win<K>.f16, K route windows per frame (2x2-cell means of the stride-32 map around K points "
                        "along the frame's route), for train_wor.py --route_windows K. Needs --crop_bottom_frac 0 (the projection maps the full frame).")
    p.add_argument("--fixed_windows", type=int, default=0,
                   help="2026-11-07, placement control: with --route_windows K also write <prefix>.fwin<K>.f16, the same K windows placed on a fixed "
                        "straight-ahead line instead of the frame's route")
    p.add_argument("--flip", type=int, default=0,
                   help="2026-11-07, flip arm: encode every frame mirrored left-right (after the overlay); windows and labels mirrored with it. "
                        "Use its own --out_dir; train with --flip_cache <that prefix>")
    p.add_argument("--box_labels", type=int, default=0,
                   help="2026-11-07, auxiliary-target arm: also write per-cell labels from the logged boxes (<prefix>.lab.f16, .labvalid.u8)")
    p.add_argument("--pca_full", type=int, default=0,
                   help="2026-11-07, full-map arms: also write the whole stride-32 map compressed to K PCA channels (<prefix>.pca<K>.f16); "
                        "the basis is fitted first on --pca_fit_frames frames and saved as <file>.basis.pt")
    p.add_argument("--pca_fit_frames", type=int, default=3000)
    p.add_argument("--encoder_mode", default="", choices=["", "stages", "fusion_zero", "fusion_pseudo"],
                   help="2026-11-08: encode with TF++'s camera branch from --tfpp_dir (src/models/world_on_rails/tfpp_camera.py): 'stages' = image "
                        "stages only (= the default encoder), 'fusion_zero' = TF++'s fusion blocks kept with an empty LiDAR, 'fusion_pseudo' = with a "
                        "pseudo-LiDAR from TF++'s own predicted depth. The cache prefix gets the mode as a suffix.")
    p.add_argument("--tfpp_dir", default="", help="the TF++ release folder holding config.json and model_0030_0.pth (pretrained_models/all_towns)")
    p.add_argument("--side_windows", type=float, default=0.0,
                   help="2026-11-08: with --route_windows K also write <prefix>.swin<K>.f16, 2K windows this many metres left / right of the route windows")
    p.add_argument("--window_labels", type=int, default=0,
                   help="2026-11-08: with --route_windows also write the path-relative obstacle targets of every window (.lab.f16) and <prefix>.wlabvalid.u8")
    p.add_argument("--decoder_stats", type=int, default=0,
                   help="2026-11-08 (needs --encoder_mode): also write TF++ depth / semantic statistics per token cell (<prefix>.dstat.f16) and per window")
    p.add_argument("--val_only", type=int, default=0,
                   help="1: encode only the held-out frames of the validation split that vision_ablation_on_val.py reads (route_group_split, "
                        "val_split 0.15, seed 0, no augmented renders). The cache is then partial (meta partial=1): the reader checks each frame's "
                        "done flag and fails on any unencoded frame. 2026-11-06: the disk was too small for the full 20 GB cache.")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (default fp32: the cache then matches an fp32 forward to numerical noise, 0.008 m max on real frames; bf16 matches only to its own batch-shape noise, 0.125 m max)")
    a = p.parse_args()

    os.environ["WOR_ROUTE_KEY"] = a.route_key
    os.environ.pop("WOR_POOLED_CACHE", None)  # the builder must read pixels, never a cache
    from src.models.world_on_rails.vision_encoder import build_vision_encoder
    from src.training.wor_dataset import WorldOnRailsDataset

    gh, gw = _size(a.grid) if "x" in a.grid.lower() else (int(a.grid), int(a.grid))
    ds = WorldOnRailsDataset(data_dir=a.data_dir, is_train=True, cache_decoded=False, route_points=a.route_points, img_size=_size(a.img_size),
                             crop_bottom_frac=a.crop_bottom_frac, route_overlay=bool(a.route_overlay), use_augmented_camera=bool(a.use_augmented_camera))
    if ds.is_synthetic:
        raise SystemExit(f"{a.data_dir} indexed as synthetic/empty: nothing to cache")
    assert all(s.get("format") == "pdm_lite" and s.get("rgb_path") for s in ds.samples), "only PDM-Lite frames with an rgb_path are supported"
    keys = [rel_key(s["rgb_path"], a.data_dir) for s in ds.samples]
    assert len(set(keys)) == len(keys), "duplicate frame keys"
    tag = ds.pixel_cache_tag()
    prefix = os.path.join(a.out_dir, f"{os.path.basename(os.path.normpath(a.data_dir))}_{tag}_{a.backbone}_g{gh}x{gw}"
                                     + (f"_{a.encoder_mode}" if a.encoder_mode else ""))

    val_keys = None
    if a.val_only:  # the held-out frames exactly as vision_ablation_on_val.py builds them (same loader arguments)
        from src.training.wor_dataloaders import create_wor_train_val_dataloaders
        _, vl = create_wor_train_val_dataloaders(
            data_dir=a.data_dir, batch_size=a.batch_size, num_workers=0, val_split=0.15, split_seed=0, seed=0,
            img_size=_size(a.img_size), crop_bottom_frac=0.0, route_overlay=bool(a.route_overlay), route_points=a.route_points,
            use_augmented_camera=False)
        val_keys = {rel_key(vl.dataset.dataset.samples[i]["rgb_path"], a.data_dir) for i in vl.dataset.indices}
        missing = val_keys - set(keys)
        assert not missing, f"{len(missing)} held-out frames are not in the cache index (first: {sorted(missing)[:3]})"
        print(f"--> val_only: {len(val_keys)} held-out frames of {len(keys)} indexed", flush=True)

    # --random_init: the image-use control (2026-10-15). A randomly initialised encoder replaces the pretrained one, so the head
    # can only learn from features that carry no driving-specific content. Recorded in the cache meta; never used for a claim on its own.
    assert a.random_init or a.weights_path or a.frozen_backbone or a.encoder_mode, "give --weights_path or --frozen_backbone (or --random_init for the image-use control)"
    assert not a.decoder_stats or a.encoder_mode, "--decoder_stats reads TF++'s decoders: give --encoder_mode and --tfpp_dir"
    if a.encoder_mode:
        assert a.tfpp_dir and not (a.frozen_backbone or a.random_init or a.pca_full), "--encoder_mode needs --tfpp_dir (and no --frozen_backbone / --random_init / --pca_full)"
        from src.models.world_on_rails.tfpp_camera import TfppCamera
        encoder = TfppCamera(a.tfpp_dir, a.encoder_mode)
    else:
        encoder = build_vision_encoder(backbone_name=a.backbone, pretrained=(a.frozen_backbone is None and not a.random_init), freeze_backbone=True,
                                       weights_path=None if (a.frozen_backbone or a.random_init) else a.weights_path)
    if a.frozen_backbone:  # keys of the policy state dict: 'encoder.<name>'
        sd = torch.load(a.frozen_backbone, map_location="cpu")
        sd = sd.get("model", sd)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        assert enc_sd and not missing, f"frozen_backbone.pth does not hold this encoder: missing {list(missing)[:5]}, {len(enc_sd)} tensors found"
        print(f"--> encoder weights from {a.frozen_backbone}: {len(enc_sd)} tensors, unexpected {len(unexpected)}", flush=True)
    encoder = encoder.to(a.device).eval()
    meta = {"pixel_tag": tag, "data_dir_name": os.path.basename(os.path.normpath(a.data_dir)), "backbone": a.backbone, "weights_path": a.weights_path or a.frozen_backbone,
            "img_size": a.img_size, "crop_bottom_frac": a.crop_bottom_frac, "route_overlay": a.route_overlay, "route_points": a.route_points,
            "route_key": a.route_key, "use_augmented_camera": a.use_augmented_camera, "created": time.strftime("%F %T"),
            "partial": bool(a.val_only), "val_only": bool(a.val_only), "flip": bool(a.flip),
            "encoder_mode": a.encoder_mode, "tfpp_dir": a.tfpp_dir, "side_windows": a.side_windows, "window_labels": bool(a.window_labels),
            "decoder_stats": bool(a.decoder_stats)}
    jp, fp, dp = cache_files(prefix)
    os.makedirs(a.out_dir, exist_ok=True)
    lock = prefix + ".lock"
    try:  # the first shard creates the layout; the others wait for the json, which is written last
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        init_cache(prefix, keys, int(encoder.out_channels), (gh, gw), meta)
    except FileExistsError:
        while not os.path.exists(jp):
            time.sleep(2)
    init_cache(prefix, keys, int(encoder.out_channels), (gh, gw), meta)  # idempotent: verifies the layout is the one this shard indexed
    mm = np.memmap(fp, dtype=np.float16, mode="r+", shape=(len(keys), int(encoder.out_channels), gh, gw))
    done = np.memmap(dp, dtype=np.uint8, mode="r+", shape=(len(keys),))
    C = int(encoder.out_channels)
    N = len(keys)

    def _mm(path, shape, dtype=np.float16):
        if not os.path.exists(path):
            np.memmap(path, dtype=dtype, mode="w+", shape=shape).flush()
        return np.memmap(path, dtype=dtype, mode="r+", shape=shape)

    wm = fwm = None
    fixed_fr = None
    K = a.route_windows
    swm = None
    side = {}                                    # 2026-11-08 window side files: {(kind, what): memmap}
    if a.route_windows:
        assert a.route_windows % gh == 0, "--route_windows must be a multiple of the grid height (the windows are packed as extra columns)"
        wm = _mm(window_file(prefix, K, "route"), (N, C, K))
        print(f"--> route windows: K={K} -> {window_file(prefix, K, 'route')}", flush=True)
        if a.fixed_windows:
            fwm = _mm(window_file(prefix, K, "fixed"), (N, C, K))
            fixed_fr = torch.as_tensor(route_window_fracs(STRAIGHT_ROUTE, K, a.crop_bottom_frac))   # symmetric about the centre: the same when mirrored
            print(f"--> fixed windows (placement control) -> {window_file(prefix, K, 'fixed')}", flush=True)
        kinds = [("route", K)] + ([("side", 2 * K)] if a.side_windows > 0 else [])
        if a.side_windows > 0:
            swm = _mm(window_file(prefix, K, "side"), (N, C, 2 * K))
            print(f"--> side windows: 2K={2 * K}, {a.side_windows} m off the route -> {window_file(prefix, K, 'side')}", flush=True)
        for kind, M in kinds:
            f = window_side_files(prefix, K, kind)
            side[(kind, "xy")] = _mm(f["xy"], (N, 2, M))
            if a.window_labels:
                side[(kind, "lab")] = _mm(f["lab"], (N, 2, M))
            if a.decoder_stats:
                side[(kind, "dstat")] = _mm(f["dstat"], (N, DSTAT_CHANNELS, M))
        if a.window_labels:
            side["wlv"] = _mm(window_label_valid_file(prefix), (N,), np.uint8)
    dsm = _mm(grid_dstat_file(prefix), (N, DSTAT_CHANNELS, gh, gw)) if a.decoder_stats else None
    lm = lv = None
    if a.box_labels:
        lp, vp = label_files(prefix)
        lm, lv = _mm(lp, (N, 2, 4, 4)), _mm(vp, (N,), np.uint8)
        print(f"--> box labels -> {lp}", flush=True)
    use_amp = a.amp and a.device.startswith("cuda") and torch.cuda.is_bf16_supported(including_emulation=False)
    todo = [i for i in range(len(keys)) if i % a.num_shards == a.shard and not done[i] and (val_keys is None or keys[i] in val_keys)]
    if a.limit_frames:
        todo = todo[:a.limit_frames]
    pm = pca_mean = pca_basis = None
    if a.pca_full:
        pf = pca_file(prefix, a.pca_full)
        bp = pf + ".basis.pt"
        if not os.path.exists(bp):
            rng = np.random.RandomState(0)
            fit_rows = sorted(rng.choice(len(keys), size=min(a.pca_fit_frames, len(keys)), replace=False).tolist())
            mean, basis, explained, nvec = fit_pca(encoder, ds, fit_rows, a.pca_full, a.device, a.batch_size, a.num_workers, use_amp)
            torch.save({"mean": mean, "basis": basis, "explained": explained, "fit_frames": len(fit_rows), "fit_vectors": nvec}, bp)
            print(f"--> PCA basis: {a.pca_full} of {C} channels explain {explained:.3f} of the variance ({len(fit_rows)} frames, {nvec} cell vectors) -> {bp}", flush=True)
        st = torch.load(bp)
        pca_mean, pca_basis = st["mean"].to(a.device), st["basis"].to(a.device)
        with torch.no_grad():
            fh, fw = encoder(torch.zeros(1, 3, *_size(a.img_size), device=a.device)).shape[2:]
        json.dump({"shape": [N, a.pca_full, int(fh), int(fw)], "basis": bp, "explained": st["explained"]}, open(pf + ".json", "w"))
        pm = _mm(pf, (N, a.pca_full, int(fh), int(fw)))
        print(f"--> PCA full map: ({a.pca_full}, {fh}, {fw}) per frame -> {pf}", flush=True)
    print(f"--> cache {prefix}: {len(keys)} frames, shard {a.shard}/{a.num_shards}: {len(todo)} to encode, grid {gh}x{gw}, channels {C}"
          f"{' (mirrored frames)' if a.flip else ''}", flush=True)
    if not todo:
        print(f"CACHE_PREFIX {prefix}")
        return
    loader = DataLoader(_Pending(ds, todo, a.route_windows, flip=bool(a.flip), labels=bool(a.box_labels), side_offset=a.side_windows,
                                 window_labels=bool(a.window_labels), crop_bottom_frac=a.crop_bottom_frac), batch_size=a.batch_size,
                        num_workers=a.num_workers, pin_memory=a.device.startswith("cuda"), shuffle=False)
    t0, n, last = time.time(), 0, time.time()
    with torch.no_grad():
        for b, item in enumerate(loader):
            rows, rgb = item[0], item[1]
            rn = rows.numpy()
            x = rgb.to(a.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            maps = None
            with torch.autocast(device_type="cuda" if a.device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                if a.encoder_mode:
                    out = encoder(x, crop_bottom_frac=a.crop_bottom_frac, return_maps=bool(a.decoder_stats))
                    feats, maps = out if a.decoder_stats else (out, None)
                else:
                    feats = encoder(x)
            # every side file is written before pool_and_store flags the rows done
            if wm is not None:
                fr = item[2]
                store_route_windows(feats, rn, fr[:, :K], wm)
                if swm is not None:
                    store_route_windows(feats, rn, fr[:, K:], swm)
                side[("route", "xy")][rn] = item[5][:, :, :K].numpy().astype(np.float16)
                if swm is not None:
                    side[("side", "xy")][rn] = item[5][:, :, K:].numpy().astype(np.float16)
                if a.window_labels:
                    side[("route", "lab")][rn] = item[6][:, :, :K].numpy().astype(np.float16)
                    if swm is not None:
                        side[("side", "lab")][rn] = item[6][:, :, K:].numpy().astype(np.float16)
                    side["wlv"][rn] = item[7].numpy().astype(np.uint8)
            if maps is not None:
                from src.models.world_on_rails.tfpp_camera import decoder_stat_maps
                st = decoder_stat_maps(maps[0], maps[1], a.crop_bottom_frac)                 # (B, 10, H/32, W/32)
                dsm[rn] = F.adaptive_avg_pool2d(st, (gh, gw)).half().cpu().numpy()
                if wm is not None:
                    store_route_windows(st, rn, item[2][:, :K], side[("route", "dstat")])
                    if swm is not None:
                        store_route_windows(st, rn, item[2][:, K:], side[("side", "dstat")])
            if fwm is not None:
                store_route_windows(feats, rn, fixed_fr.unsqueeze(0).expand(len(rn), -1, -1), fwm)
            if lm is not None:
                lm[rn] = item[3].numpy().astype(np.float16)
                lv[rn] = item[4].numpy().astype(np.uint8)
            if pm is not None:
                pm[rn] = torch.einsum("bchw,kc->bkhw", feats.float() - pca_mean.view(1, -1, 1, 1), pca_basis).half().cpu().numpy()
            pool_and_store(feats, rn, mm, done, (gh, gw))
            n += len(rows)
            if b % 20 == 19:
                for m_ in (mm, done, wm, fwm, lm, lv, pm, swm, dsm, *side.values()):
                    if m_ is not None:
                        m_.flush()
            if time.time() - last > 30:
                last = time.time()
                print(f"--> {n}/{len(todo)} ({n / (last - t0):.0f} frames/s, ~{(len(todo) - n) / max(n / (last - t0), 1e-6) / 60:.1f} min left)", flush=True)
    for m_ in (mm, done, wm, fwm, lm, lv, pm, swm, dsm, *side.values()):
        if m_ is not None:
            m_.flush()
    if a.window_labels and K:
        wl = np.asarray(side[("route", "lab")][todo[:20000]], dtype=np.float32)
        msg = f"--> window labels valid for {int(np.asarray(side['wlv'][todo]).sum())} of {len(todo)} frames; occupied route windows {float(wl[:, 0].mean()):.3f}"
        if swm is not None:
            msg += f", side windows {float(np.asarray(side[('side', 'lab')][todo[:20000]], dtype=np.float32)[:, 0].mean()):.3f}"
        print(msg, flush=True)
    if lv is not None:
        print(f"--> box labels valid for {int(np.asarray(lv[todo]).sum())} of {len(todo)} frames; mean occupied cells "
              f"{float(np.asarray(lm[todo[:20000]])[:, 0].astype(np.float32).sum((1, 2)).mean()):.2f} of 16", flush=True)
    fps = n / max(time.time() - t0, 1e-6)
    print(f"--> shard done: {n} frames in {time.time() - t0:.0f} s ({fps:.0f} frames/s)", flush=True)
    try:  # MLflow (project rule): throughput of the cache build
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + os.path.join(a.out_dir, "mlruns")))
        mlflow.set_experiment("A59_pooled_cache")
        with mlflow.start_run(run_name=os.path.basename(prefix) + f"_s{a.shard}"):
            mlflow.log_params({k: v for k, v in meta.items() if k != "created"} | {"shard": a.shard, "num_shards": a.num_shards, "grid": f"{gh}x{gw}"})
            mlflow.log_metrics({"frames": n, "frames_per_s": fps, "total_frames": len(keys)})
    except Exception as e:  # tracking must never fail the build
        print("mlflow skipped:", e)
    print(f"CACHE_PREFIX {prefix}")


if __name__ == "__main__":
    main()
