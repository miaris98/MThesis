#!/usr/bin/env python3
"""How much does the policy use its camera? Held-out loss with the visual tokens replaced (2026-11-04).

WHY THIS EXISTS
---------------
The vision-fix arms (fix 3 / fix 2 / fix 4) are judged on one question: does the head use the visual
information that is there? A closed-loop run answers it only after an expensive evaluation. This script
answers the cheap version on the held-out routes: take a trained checkpoint, run its held-out frames
through the head with the frozen encoder's feature map replaced by an uninformative one, and measure
how much the waypoint loss rises. A small rise means the head is ignoring the camera; a large rise means
it reads the map.

Five versions of the same batch, all through the same head and the same weights:
    real           the cached feature map of the frame (the number the training log reports)
    zero           all-zero map (the vision tokens carry no information)
    batch_mean     every frame gets the mean map of the batch (the average scene, same for all)
    scene_shuffle  each frame gets the map of another frame of the batch (a plausible but wrong scene)
    cell_shuffle   the 16 cells of each frame's map are permuted (the scene's content, in the wrong places)
The gap to `real` is the loss the head pays for each kind of wrong vision. zero and batch_mean measure
how much it needs any vision; scene_shuffle how much it needs the right scene; cell_shuffle how much it
needs the content where it is (the obstacle must be at the right cell to change the waypoints).

Only pooled-cache arms can run here (the cache holds the frozen stride-32 map, pooled to --vision_grid).
The head's checkpoint config (path head, vision grid, qk-norm) is read by load_wor_model, as in
score_checkpoints_on_val.py. The validation split is the trainer's (route-level, --val_split 0.15,
--split_seed 0) and augmented recovery renders are excluded, so the `real` row must reproduce the run's
logged val_loss (the check that the environment is wired the same as the trainer).

    WOR_POOLED_CACHE=<prefix> python scripts/analysis/vision_ablation_on_val.py \\
        --pooled_cache <prefix> --route_overlay 1 --checkpoints <model_epoch_030.pth> [...] --out_csv <file.csv>

Logs to MLflow (file store) when available, as CLAUDE.md requires.
"""
import argparse
import csv
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.world_on_rails.wor_loader import load_wor_model
import src.training.wor_dataset  # noqa: F401  (import order: wor_dataset must load before wor_dataloaders, which it imports back)
from src.training.wor_dataloaders import create_wor_train_val_dataloaders
from src.training.wor_eval import to_device_batch_with_features, waypoint_losses
from src.training.wor_trainer import autocast

VARIANTS = ("real", "zero", "batch_mean", "scene_shuffle", "cell_shuffle")


def _parse_img_size(spec):
    if "x" in spec.lower():
        h, w = spec.lower().split("x")
        return int(h), int(w)
    n = int(spec)
    return n, n


def transform(vf, mode, gen):
    """The feature map of a batch (B, C, ...) replaced according to `mode` (see the module docstring)."""
    if mode == "real":
        return vf
    if mode == "zero":
        return torch.zeros_like(vf)
    B = vf.shape[0]
    flat = vf.reshape(B, vf.shape[1], -1)
    if mode == "batch_mean":
        return flat.mean(0, keepdim=True).expand_as(flat).reshape_as(vf)
    if mode == "scene_shuffle":
        perm = torch.randperm(B, generator=gen).to(vf.device)
        return vf[perm]
    if mode == "cell_shuffle":
        perm = torch.randperm(flat.shape[2], generator=gen).to(vf.device)
        return flat[:, :, perm].reshape_as(vf)
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True)
    p.add_argument("--pooled_cache", type=str, required=True,
                   help="prefix of the pooled cache built with this arm's overlay setting (pooled_ov1 or pooled_ov0)")
    p.add_argument("--route_overlay", type=int, required=True, help="1 for pooled_ov1 arms, 0 for pooled_ov0")
    p.add_argument("--data_dir", type=str, default="/workspace/dataset/wor_trajectories")
    p.add_argument("--backbone", type=str, default="regnety_032")
    p.add_argument("--policy_arch", type=str, default="qwen30m")
    p.add_argument("--vision_grid", type=str, default="4",
                   help="'4', 'HxW', or 'auto' (2026-11-07: read from the checkpoint's own config, for the full-map / route-window arms)")
    p.add_argument("--img_size", type=str, default="288x768")
    p.add_argument("--route_points", type=int, default=4)
    p.add_argument("--route_key", type=str, default="route_original")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--wp_loss_weight", type=float, default=1.0)
    p.add_argument("--q_loss_weight", type=float, default=0.0)
    p.add_argument("--lateral_loss_weight", type=float, default=3.0)
    p.add_argument("--max_batches", type=int, default=0, help="0 = the whole held-out split")
    p.add_argument("--slow_dist_m", type=float, default=4.0,
                   help="the 'slow' subset: the expert's last waypoint is within this many metres of the ego car")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--use_mlflow", type=int, default=1)
    p.add_argument("--mlflow_port", type=int, default=10100)
    p.add_argument("--experiment_name", type=str, default="vision_ablation")
    args = p.parse_args()

    # The pooled-cache dataset reads these from the environment (set the same way train_wor.py sets them).
    os.environ["WOR_POOLED_CACHE"] = args.pooled_cache
    os.environ["WOR_ROUTE_KEY"] = args.route_key

    _, val_loader = create_wor_train_val_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        val_split=args.val_split, split_seed=args.split_seed, seed=args.split_seed,
        img_size=_parse_img_size(args.img_size), crop_bottom_frac=0.0,
        route_overlay=bool(args.route_overlay), route_points=args.route_points,
        use_augmented_camera=False,
    )
    if val_loader is None:
        print("[Error] no validation loader was built.")
        return 1
    print(f"--> {len(val_loader.dataset)} held-out frames, arm overlay={args.route_overlay}, cache={args.pooled_cache}", flush=True)

    logger = None
    try:
        from src.logging.experiment_logger import ExperimentLogger
        out_dir = os.path.dirname(os.path.abspath(args.out_csv))
        logger = ExperimentLogger(out_dir, checkpoint_dir=out_dir, experiment_name=args.experiment_name,
                                  use_mlflow=bool(args.use_mlflow), mlflow_port=args.mlflow_port)
        logger.log_params({"pooled_cache": args.pooled_cache, "route_overlay": args.route_overlay,
                           "val_frames": len(val_loader.dataset), "split_seed": args.split_seed,
                           "vision_grid": args.vision_grid, "n_checkpoints": len(args.checkpoints)})
    except Exception as exc:
        print(f"[Warning] experiment logging unavailable ({exc}) - continuing without it.", flush=True)
        logger = None

    rows = []
    for ci, path in enumerate(args.checkpoints):
        t0 = time.time()
        model = load_wor_model(
            checkpoint_path=path, backbone_name=args.backbone, pretrained_backbone=True, freeze_backbone=True,
            device=args.device, policy_arch=args.policy_arch, route_points=args.route_points,
            vision_grid=(None if args.vision_grid == "auto" else
                         tuple(int(v) for v in args.vision_grid.lower().split("x")) if "x" in args.vision_grid.lower() else int(args.vision_grid)),
        ).to(args.device).eval()
        acc = {v: {"loss": 0.0, "ade": 0.0, "n": 0} for v in VARIANTS}
        acc_slow = {v: {"loss": 0.0, "ade": 0.0, "n": 0} for v in VARIANTS}
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if args.max_batches and bi >= args.max_batches:
                    break
                rgb, vf, speed, command, route, target_q, target_wp = to_device_batch_with_features(batch, args.device)
                if vf is None:
                    raise SystemExit("the batch has no vision_features: this script needs a pooled-cache arm")
                # The expert nearly stopped over the horizon (its last waypoint within --slow_dist_m of the ego car):
                # the frames where a stop or a crawl is needed, which is where an obstacle would change the answer.
                B = vf.shape[0]
                slow = target_wp[:, -1].norm(dim=-1) < args.slow_dist_m
                idx = slow.nonzero(as_tuple=True)[0]
                for v in VARIANTS:
                    x = transform(vf, v, gen)
                    with autocast(enabled=not args.no_amp):
                        out = model(rgb, speed, command, route, vision_features=x)
                        losses = waypoint_losses(out, target_wp, target_q, args.wp_loss_weight,
                                                 args.q_loss_weight, args.lateral_loss_weight)
                        acc[v]["loss"] += losses["total"].item()
                        acc[v]["ade"] += losses["ade"].item()
                        acc[v]["n"] += 1
                        if idx.numel() > 0:
                            sub = {k: (t[idx] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t)
                                   for k, t in out.items()}
                            ls = waypoint_losses(sub, target_wp[idx], target_q[idx], args.wp_loss_weight,
                                                 args.q_loss_weight, args.lateral_loss_weight)
                            # batch means, as run_validation averages them (the slow subset is one batch's worth per batch)
                            acc_slow[v]["loss"] += ls["total"].item()
                            acc_slow[v]["ade"] += ls["ade"].item()
                            acc_slow[v]["n"] += 1
        name = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        real = acc["real"]["loss"] / max(1, acc["real"]["n"])
        real_slow = acc_slow["real"]["loss"] / max(1, acc_slow["real"]["n"])
        for subset, a, base in (("all", acc, real), ("slow", acc_slow, real_slow)):
            for v in VARIANTS:
                n = max(1, a[v]["n"])
                loss, ade = a[v]["loss"] / n, a[v]["ade"] / n
                rows.append({"checkpoint": name, "subset": subset, "variant": v, "val_loss": loss,
                             "val_ade_m": ade, "delta_loss_vs_real": loss - base, "batches": a[v]["n"]})
                print(f"  {name:36s} {subset:5s} {v:14s} val_loss={loss:.4f} ADE={ade:.3f}m  delta={loss - base:+.4f}",
                      flush=True)
                if logger is not None and subset == "all":
                    logger.add_scalar(f"ablation/{v}/val_loss", float(loss), ci)
                    logger.add_scalar(f"ablation/{v}/delta_loss", float(loss - base), ci)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}", flush=True)
    if logger is not None:
        logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
