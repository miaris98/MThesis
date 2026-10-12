#!/usr/bin/env python3
"""Re-scores already-trained checkpoints on a freshly-reconstructed validation split.

WHY THIS EXISTS
---------------
A run's reported val_loss is only meaningful against the split that run held out. When the
split itself changes - because it was wrong, or because two runs were configured differently -
every number recorded under the old split becomes incomparable, and the only honest way to
compare the models is to score them again on one split.

That is exactly what happened on 2026-09-20: `--use_augmented_camera` leaked PDM-Lite's
rgb_augmented/ recovery renders into the `--val_split` held-out set (fixed in 8734c58), so the
regnety_032 run (flag off, 20,477 val frames) and the resnet34 run (flag on, 40,954) reported
val losses measured on different distributions. Retraining both to compare them would cost
hours; re-scoring their saved checkpoints on the corrected split costs minutes and answers the
same question, because the corrected val set is a subset of the contaminated one drawn from
the same held-out routes - neither model ever trained on those routes.

This does NOT re-select `best_model.pth`. A checkpoint chosen on a contaminated metric may not
be the epoch that was actually best, which is why `--checkpoints` accepts the whole
`model_epoch_*.pth` series: score them all and let the corrected split pick the winner.

USAGE
-----
Every dataset-shaping flag must match the run being scored - they select the split and the
pixel/feature cache keys, exactly as in train_wor.py:

    python scripts/analysis/score_checkpoints_on_val.py \\
        --checkpoints '/workspace/checkpoints/wor_qwen30m_geom_resnet34/model_epoch_*.pth' \\
        --data_dir /workspace/dataset/wor_trajectories \\
        --backbone resnet34 --policy_arch qwen30m --vision_grid 8 \\
        --img_size 192x512 --crop_bottom_frac 0.25 --route_overlay 1 --route_points 4

The validation set is always built with use_augmented_camera=False regardless of how the run
was trained: recovery renders are a training-only device, and the held-out metric has to stay
one quantity across every run in the series.
"""
import argparse
import glob
import os
import sys
from typing import List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_dataloaders import create_wor_train_val_dataloaders
from src.training.wor_eval import run_validation
from src.training.wor_trainer import autocast


def _parse_img_size(spec: str) -> Tuple[int, int]:
    if "x" in spec.lower():
        h, w = spec.lower().split("x")
        return int(h), int(w)
    n = int(spec)
    return n, n


def _expand(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(c in p for c in "*?[") else ([p] if os.path.exists(p) else [])
        if not hits:
            print(f"[Warning] no checkpoint matched: {p}")
        out.extend(hits)
    # Dedupe while keeping order - a caller passing both a glob and an explicit best_model.pth
    # should not score the same file twice.
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True,
                   help="Paths or globs. Quote globs so the shell does not expand them.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--backbone", type=str, default="regnety_032")
    p.add_argument("--policy_arch", type=str, default="qwen30m")
    p.add_argument("--vision_grid", type=int, default=None,
                   help="Left unset, each checkpoint's own stored config decides - see load_wor_model.")
    p.add_argument("--img_size", type=str, default="192x512")
    p.add_argument("--crop_bottom_frac", type=float, default=0.25)
    p.add_argument("--route_overlay", type=int, default=1)
    p.add_argument("--route_points", type=int, default=4)
    p.add_argument("--feature_cache_tag", type=str, default=None,
                   help="Only for a backbone whose cache was actually built; a missing entry is a "
                        "hard error by design, not a silent fallback to live encoding.")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--wp_loss_weight", type=float, default=1.0)
    p.add_argument("--q_loss_weight", type=float, default=0.0)
    p.add_argument("--lateral_loss_weight", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--use_mlflow", type=int, default=1)
    p.add_argument("--mlflow_port", type=int, default=10100)
    p.add_argument("--experiment_name", type=str, default="WoR_Val_Rescore")
    p.add_argument("--run_label", type=str, default="rescore")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where val_rescore.csv and the TensorBoard/MLflow logs land. "
                        "Defaults to the first checkpoint's directory.")
    args = p.parse_args()

    ckpts = _expand(args.checkpoints)
    if not ckpts:
        print("[Error] no checkpoints to score.")
        return 1

    img_size = _parse_img_size(args.img_size)
    _, val_loader = create_wor_train_val_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        val_split=args.val_split, split_seed=args.split_seed, seed=args.split_seed,
        img_size=img_size, crop_bottom_frac=args.crop_bottom_frac,
        route_overlay=bool(args.route_overlay), route_points=args.route_points,
        feature_cache_tag=args.feature_cache_tag,
        # Never augmented: the held-out metric must mean the same thing for every run scored.
        use_augmented_camera=False
    )
    if val_loader is None:
        print("[Error] no validation loader was built - check --val_split and --data_dir.")
        return 1
    print(f"--> Scoring {len(ckpts)} checkpoint(s) on {len(val_loader.dataset)} held-out frames "
          f"(split_seed={args.split_seed}, augmented frames excluded).\n")

    logger = None
    try:
        from src.logging.experiment_logger import ExperimentLogger
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(ckpts[0]))
        logger = ExperimentLogger(
            out_dir, checkpoint_dir=out_dir, experiment_name=args.experiment_name,
            use_mlflow=bool(args.use_mlflow), mlflow_port=args.mlflow_port
        )
        logger.log_params({
            "backbone": args.backbone, "policy_arch": args.policy_arch,
            "val_split": args.val_split, "split_seed": args.split_seed,
            "val_frames": len(val_loader.dataset), "run_label": args.run_label,
            "use_augmented_camera": False, "n_checkpoints": len(ckpts),
            "feature_cache_tag": args.feature_cache_tag or "",
            "img_size": args.img_size, "crop_bottom_frac": args.crop_bottom_frac,
            "route_overlay": args.route_overlay, "route_points": args.route_points
        })
    except Exception as exc:
        print(f"[Warning] experiment logging unavailable ({exc}) - continuing without it.")
        logger = None

    rows = []
    for i, path in enumerate(ckpts):
        model = load_wor_model(
            checkpoint_path=path, backbone_name=args.backbone,
            pretrained_backbone=True, freeze_backbone=True, device=args.device,
            policy_arch=args.policy_arch, route_points=args.route_points,
            vision_grid=args.vision_grid
        ).to(args.device)
        m = run_validation(model, val_loader, args.device, autocast, not args.no_amp,
                           args.wp_loss_weight, args.q_loss_weight, args.lateral_loss_weight)
        rows.append((os.path.basename(path), m))
        print(f"  [{i+1}/{len(ckpts)}] {os.path.basename(path):24s} "
              f"val_loss={m['val_loss']:.4f}  ADE={m['val_wp_ade_m']:.3f}m  "
              f"lat={m['val_wp_lateral_error_m']:.3f}m", flush=True)
        if logger is not None:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    logger.add_scalar(f"rescore/{k}", float(v), i)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: r[1]["val_loss"])
    best_name, best = rows[0]
    print(f"\n=== BEST on the corrected split: {best_name} ===")
    print(f"    val_loss={best['val_loss']:.4f}  ADE={best['val_wp_ade_m']:.3f}m  "
          f"FDE={best['val_wp_fde_m']:.3f}m  lateral={best['val_wp_lateral_error_m']:.3f}m")
    if logger is not None:
        logger.log_params({"best_checkpoint": best_name})
        for k, v in best.items():
            if isinstance(v, (int, float)):
                logger.add_scalar(f"rescore_best/{k}", float(v), 0)
        logger.close()

    # A CSV beside the checkpoints, so the corrected numbers live with the run they belong to
    # rather than only in a terminal scrollback or an MLflow server that may not be reachable.
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(ckpts[0]))
    csv_path = os.path.join(out_dir, "val_rescore.csv")
    try:
        import csv as _csv
        keys = sorted({k for _, m in rows for k in m})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["checkpoint"] + keys)
            w.writeheader()
            for name, m in rows:
                w.writerow({"checkpoint": name, **m})
        print(f"    wrote {csv_path}")
    except Exception as exc:
        print(f"[Warning] could not write {csv_path}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
