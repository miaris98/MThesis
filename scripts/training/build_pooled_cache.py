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

from src.training.pooled_feature_cache import cache_files, init_cache, rel_key  # noqa: E402


def pool_and_store(feats: torch.Tensor, rows, mm, done, grid) -> None:
    """(B, C, H, W) encoder output -> (B, C, gh, gw) fp16 rows of the memmap, flagged done. The pooling is the head's own `AdaptiveAvgPool2d(grid)` on the float32 map."""
    pooled = F.adaptive_avg_pool2d(feats.float(), tuple(grid)).half().cpu().numpy()
    rows = np.asarray(rows)
    mm[rows] = pooled
    done[rows] = 1


class _Pending(Dataset):
    def __init__(self, ds, rows):
        self.ds, self.rows = ds, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        it = self.ds.samples[r]
        rgb = self.ds._load_rgb(it["rgb_path"], route_xy=it.get("route"))  # the very function training calls: identical pixels
        return r, torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.uint8)


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
    prefix = os.path.join(a.out_dir, f"{os.path.basename(os.path.normpath(a.data_dir))}_{tag}_{a.backbone}_g{gh}x{gw}")

    assert a.weights_path or a.frozen_backbone, "give --weights_path or --frozen_backbone"
    encoder = build_vision_encoder(backbone_name=a.backbone, pretrained=a.frozen_backbone is None, freeze_backbone=True,
                                   weights_path=None if a.frozen_backbone else a.weights_path)
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
            "route_key": a.route_key, "use_augmented_camera": a.use_augmented_camera, "created": time.strftime("%F %T")}
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
    todo = [i for i in range(len(keys)) if i % a.num_shards == a.shard and not done[i]]
    if a.limit_frames:
        todo = todo[:a.limit_frames]
    print(f"--> cache {prefix}: {len(keys)} frames, shard {a.shard}/{a.num_shards}: {len(todo)} to encode, grid {gh}x{gw}, channels {encoder.out_channels}", flush=True)
    if not todo:
        print(f"CACHE_PREFIX {prefix}")
        return
    loader = DataLoader(_Pending(ds, todo), batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=a.device.startswith("cuda"), shuffle=False)
    use_amp = a.amp and a.device.startswith("cuda") and torch.cuda.is_bf16_supported(including_emulation=False)
    t0, n, last = time.time(), 0, time.time()
    with torch.no_grad():
        for b, (rows, rgb) in enumerate(loader):
            x = rgb.to(a.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            with torch.autocast(device_type="cuda" if a.device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                feats = encoder(x)
            pool_and_store(feats, rows.numpy(), mm, done, (gh, gw))
            n += len(rows)
            if b % 20 == 19:
                mm.flush(); done.flush()
            if time.time() - last > 30:
                last = time.time()
                print(f"--> {n}/{len(todo)} ({n / (last - t0):.0f} frames/s, ~{(len(todo) - n) / max(n / (last - t0), 1e-6) / 60:.1f} min left)", flush=True)
    mm.flush(); done.flush()
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
