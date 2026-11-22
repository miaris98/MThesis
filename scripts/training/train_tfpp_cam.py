"""TODO A86 (2026-11-21): train TF++ camera-only (its 'aim' backbone) on our sensor, resolution, frozen encoder and frame list.
What is ours and what is TF++'s: src/baselines/tfpp_cam.py. Frames, route split and recovery views are this project's
(WorldOnRailsDataset + route_group_split, split seed 0, 15% held-out routes: the same frames our policy trains and validates on);
for each training frame TF++'s rule picks the shifted camera with p = 0.5 and moves the labels with it; validation never shifts.
The frozen encoder's full 9x24 map comes from a pooled cache built with build_pooled_cache.py --grid 9x24 --use_augmented_camera 1.

Output: <save_dir>/tfpp_cam/{config.json, model_0031.pth} (carla_garage's agent format) + telemetry CSV + MLflow, resumable snapshots
every --save_freq epochs (model, optimizer, scheduler).
usage: train_tfpp_cam.py --data_dir D --pooled_cache P1,P2 --frozen_backbone E15/frozen_backbone.pth --save_dir OUT
"""
import argparse
import csv
import gzip
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.baselines import tfpp_cam as tc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pooled_cache", required=True, help="comma-separated full-map (9x24) cache prefixes")
    p.add_argument("--frozen_backbone", required=True, help="E15's frozen_backbone.pth (TF++'s image encoder, the weights our policy uses)")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--img_size", default="288x768")
    p.add_argument("--epochs", type=int, default=31, help="TF++'s: the cosine warm-restart cycles (1, 2, 4, 8, 16) end at 31")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--save_freq", type=int, default=3)
    p.add_argument("--max_batches", type=int, default=0, help="smoke tests: batches per epoch (0 = all)")
    p.add_argument("--max_frames", type=int, default=0, help="smoke tests: keep only this many training frames (0 = all)")
    p.add_argument("--resume", default="", help="a snapshot written by --save_freq (continues at its epoch + 1)")
    p.add_argument("--use_mlflow", type=int, default=1)
    p.add_argument("--experiment_name", default="tfpp_cam")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class TfppCamFrames(Dataset):
    """One main-camera frame per item: the cached map (its shifted-camera twin with TF++'s probability when training) + TF++'s labels."""

    def __init__(self, rgb_paths, data_dir, cache, labels, train: bool, augment_percentage: float):
        from src.training.pooled_feature_cache import rel_key
        self.paths, self.data_dir, self.cache, self.labels = list(rgb_paths), data_dir, cache, labels
        self.train, self.p_aug, self.rel_key = bool(train), float(augment_percentage), rel_key

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb = self.paths[i]
        meas_path = rgb.replace(os.sep + "rgb" + os.sep, os.sep + "measurements" + os.sep).replace("/rgb/", "/measurements/")
        meas_path = os.path.splitext(meas_path)[0] + ".json.gz"
        with gzip.open(meas_path, "rt", encoding="utf-8") as f:
            meas = json.load(f)
        key = self.rel_key(rgb, self.data_dir)
        aug_key = key.replace("/rgb/", "/rgb_augmented/")
        aug = self.train and random.random() <= self.p_aug and self.cache.has(aug_key) and "augmentation_rotation" in meas
        fmap = self.cache.get(aug_key if aug else key)
        lab = self.labels(meas, augmented=aug)
        return {"map": torch.from_numpy(np.asarray(fmap, dtype=np.float32)), "augmented": aug,
                **{k: torch.as_tensor(v) for k, v in lab.items()}}


class _CacheView:
    """PooledFeatureCache plus a membership test (the shifted-camera twin exists only where PDM-Lite rendered one)."""

    def __init__(self, cache):
        self.c = cache

    def has(self, key):
        return key in self.c.index

    def get(self, key):
        return self.c.get(key)

    def __getstate__(self):
        return {"c": self.c}

    def __setstate__(self, s):
        self.c = s["c"]


def frame_lists(args):
    """Our frame list and route split: the main-camera training frames and the held-out frames (validation is main camera only)."""
    from src.training.wor_dataset import WorldOnRailsDataset
    from src.training.wor_dataloaders import route_group_split
    h, w = (int(x) for x in args.img_size.split("x"))
    ds = WorldOnRailsDataset(data_dir=args.data_dir, is_train=True, route_points=4, img_size=(h, w), crop_bottom_frac=0.0,
                             route_overlay=False, use_augmented_camera=True)
    tr, va, _, keys = route_group_split(ds, args.val_split, args.split_seed)
    main = lambda i: isinstance(ds.samples[i], dict) and not ds.samples[i].get("is_recovery_augmented")
    train = [ds.samples[i]["rgb_path"] for i in tr if main(i)]
    val = [ds.samples[i]["rgb_path"] for i in va]
    print(f"--> our split: {len(tr)} train samples ({len(train)} main-camera frames) / {len(val)} held-out frames, {len(keys)} route groups, "
          f"split seed {args.split_seed}", flush=True)
    return train, val, ds.pixel_cache_tag()


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def forward(net, b):
    return net(rgb=b["map"], lidar_bev=None, target_point=b["target_point"], ego_vel=b["speed"].float().unsqueeze(1), command=b["command"])


@torch.no_grad()
def evaluate(net, loader, dev, max_batches=0):
    net.eval()
    acc, n = {"loss_checkpoint": 0.0, "loss_target_speed": 0.0, "checkpoint_l1_m": 0.0, "speed_class_acc": 0.0, "final_point_err_m": 0.0}, 0
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = to_dev(b, dev)
        out = forward(net, b)
        loss = tc.planning_loss(net, out, b)
        k = b["map"].shape[0]
        acc["loss_checkpoint"] += float(loss["loss_checkpoint"]) * k
        acc["loss_target_speed"] += float(loss["loss_target_speed"]) * k
        acc["checkpoint_l1_m"] += float((out[2] - b["checkpoint"]).abs().sum(-1).mean(-1).sum())
        acc["final_point_err_m"] += float((out[2][:, -1] - b["checkpoint"][:, -1]).norm(dim=-1).sum())
        acc["speed_class_acc"] += float((out[1].argmax(-1) == b["target_speed_twohot"].argmax(-1)).float().sum())
        n += k
    net.train()
    return {f"val_{k}": v / max(1, n) for k, v in acc.items()}


def main():
    args = parse_args()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    h, w = (int(x) for x in args.img_size.split("x"))
    config = tc.make_config((h, w))
    config.epochs, config.lr, config.batch_size = args.epochs, args.lr, args.batch_size
    labels = tc.labels_for(config)
    net = tc.build_model(config, torch.load(args.frozen_backbone, map_location="cpu", weights_only=False))
    tc.use_cached_maps(net)
    net.to(dev).train()
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"--> TF++ camera-only: {n_train:,} trainable parameters (decoder + heads); image encoder frozen (E15's weights)", flush=True)

    train_paths, val_paths, tag = frame_lists(args)
    if args.max_frames:
        train_paths = train_paths[:args.max_frames]
        val_paths = val_paths[:max(1, args.max_frames // 4)]
    from src.training.pooled_feature_cache import PooledFeatureCache, prefixes_from_env
    cache = _CacheView(PooledFeatureCache(prefixes_from_env(args.pooled_cache), tag, args.data_dir))
    if not cache.c.index:
        raise SystemExit(f"the pooled cache {args.pooled_cache} indexes no frames")
    probe = cache.get(next(iter(cache.c.index)))
    if tuple(probe.shape[-2:]) != (config.img_vert_anchors, config.img_horz_anchors):
        raise SystemExit(f"the cache holds {tuple(probe.shape)} maps; TF++'s decoder here needs (C, {config.img_vert_anchors}, "
                         f"{config.img_horz_anchors}): build it with --grid {config.img_vert_anchors}x{config.img_horz_anchors}")
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    if args.num_workers:
        kw["prefetch_factor"] = 4
    tl = DataLoader(TfppCamFrames(train_paths, args.data_dir, cache, labels, True, config.augment_percentage), shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(args.seed), **kw)
    vl = DataLoader(TfppCamFrames(val_paths, args.data_dir, cache, labels, False, 0.0), shuffle=False, **kw)

    # TF++'s optimizer and schedule (train.py): AdamW(lr, amsgrad=True) with torch's default weight decay 0.01, no parameter groups,
    # CosineAnnealingWarmRestarts(T_0=1, T_mult=2) stepped every batch at the fractional epoch; fp32, no gradient clipping
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, amsgrad=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config.cosine_t0, T_mult=config.cosine_t_mult)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"]), opt.load_state_dict(ck["optimizer"]), sched.load_state_dict(ck["scheduler"])
        start = int(ck["epoch"])
        print(f"--> resumed {args.resume} at epoch {start + 1}", flush=True)

    from src.logging.experiment_logger import ExperimentLogger
    logger = ExperimentLogger(args.save_dir, checkpoint_dir=args.save_dir, experiment_name=args.experiment_name, use_mlflow=bool(args.use_mlflow))
    logger.log_params({**{k: v for k, v in vars(args).items()}, "backbone": config.backbone, "img_anchors": f"{config.img_vert_anchors}x"
                       f"{config.img_horz_anchors}", "augment_percentage": config.augment_percentage, "use_color_aug": config.use_color_aug,
                       "use_twohot_target_speeds": config.use_twohot_target_speeds, "trainable_params": n_train})
    json.dump({**vars(args), "trainable_params": n_train}, open(os.path.join(args.save_dir, "run_config.json"), "w"), indent=2)
    tel_path = os.path.join(args.save_dir, "tfpp_cam_telemetry.csv")
    fields = ["epoch", "lr", "train_loss", "train_loss_checkpoint", "train_loss_target_speed", "aug_share", "val_loss_checkpoint",
              "val_loss_target_speed", "val_checkpoint_l1_m", "val_final_point_err_m", "val_speed_class_acc", "epoch_sec"]
    new = not os.path.exists(tel_path)
    tel = open(tel_path, "a", newline="")
    wr = csv.DictWriter(tel, fieldnames=fields)
    if new:
        wr.writeheader()
    iters = len(tl) if not args.max_batches else min(len(tl), args.max_batches)
    for epoch in range(start, args.epochs):
        t0 = time.time()
        s, n, aug = {"total": 0.0, "loss_checkpoint": 0.0, "loss_target_speed": 0.0}, 0, 0
        for i, b in enumerate(tl):
            if args.max_batches and i >= args.max_batches:
                break
            aug += int(b["augmented"].sum())
            b = to_dev(b, dev)
            loss = tc.planning_loss(net, forward(net, b), b)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            opt.step()
            sched.step(epoch + i / iters)
            for k in s:
                s[k] += float(loss[k])
            n += 1
        m = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_loss": s["total"] / max(1, n),
             "train_loss_checkpoint": s["loss_checkpoint"] / max(1, n), "train_loss_target_speed": s["loss_target_speed"] / max(1, n),
             "aug_share": aug / max(1, n * args.batch_size)}
        m.update(evaluate(net, vl, dev, args.max_batches))
        m["epoch_sec"] = time.time() - t0
        wr.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items() if k in fields}), tel.flush()
        for k, v in m.items():
            if k != "epoch":
                logger.add_scalar(f"tfpp_cam/{k}", v, epoch + 1)
        print(f"[tfpp_cam] epoch {epoch + 1:02d}/{args.epochs} train {m['train_loss']:.4f} (path {m['train_loss_checkpoint']:.4f}, speed "
              f"{m['train_loss_target_speed']:.4f}, shifted {m['aug_share']:.2f}) | held-out path L1 {m['val_checkpoint_l1_m']:.4f} m, last "
              f"point {m['val_final_point_err_m']:.3f} m, speed class acc {m['val_speed_class_acc']:.3f} | lr {m['lr']:.2e} | "
              f"{m['epoch_sec']:.0f} s", flush=True)
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            torch.save({"epoch": epoch + 1, "model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
                       os.path.join(args.save_dir, f"snapshot_{epoch + 1:03d}.pth"))
    tel.close()
    out = tc.save_tfpp_dir(net, config, os.path.join(args.save_dir, "tfpp_cam"), args.epochs)
    logger.log_artifact(tel_path)
    logger.close()
    print(f"--> TF++ camera-only saved for carla_garage's agent: {out}", flush=True)


if __name__ == "__main__":
    main()
