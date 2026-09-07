"""check_val_noise.py - is the across-seed spread training noise, or measurement noise?

Challenge Group 13.13 measured a 4.3% across-seed spread on `qwen30m`'s held-out loss
and a 5.4% margin over the conv head. Because the margin only just clears the spread,
that spread is the detection threshold for every architectural change proposed after it
(13.14) - so it matters a great deal *where it comes from*.

There are two candidates and they call for opposite fixes:

  training noise      - the same data, optimised from a different initialisation and
                        batch order, lands in a different place. Fixed by weight
                        averaging, longer schedules, or more seeds.
  measurement noise   - the model is stable but the *held-out set* is not. 15% of 9,632
                        frames is ~1,445 frames, but the split falls on route boundaries
                        and consecutive frames within a route are near-duplicates, so the
                        effective sample is some tens of independent routes. Fixed by a
                        larger or differently-constructed evaluation set, and by nothing
                        else - no amount of architectural work becomes visible until it is.

This script separates them without training anything. It takes ONE fixed checkpoint -
one set of weights, so training noise is held constant at zero by construction - and
scores it against several different route-level splits. Whatever spread survives is
purely a property of the evaluation protocol.

Read the result against the training spread from `compare_wor_runs.py`:

  split spread ~= across-seed spread   -> the 4.3% is mostly measurement. Fix the val set
                                          before running any more architecture arms.
  split spread << across-seed spread   -> the val set is fine and the 4.3% is genuine
                                          training variance. EMA / more seeds are the
                                          lever, and Tier A changes can be measured as is.

Usage:
  python check_val_noise.py --checkpoint /workspace/checkpoints/wor_sweep/qwen30m_s0/best_model.pth \
      --data_dir /workspace/dataset/wor_trajectories --policy_arch qwen30m --splits 0 1 2 3 4 5 6 7
"""
import argparse
import statistics
import sys

import torch

from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_dataset import create_wor_train_val_dataloaders
from src.training.wor_eval import run_validation

try:  # torch >= 2.0 moved autocast off the cuda namespace
    from torch.cuda.amp import autocast
except ImportError:  # pragma: no cover - depends on installed torch
    from torch.amp import autocast


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="One trained checkpoint. Its weights are held fixed across every split.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--policy_arch", type=str, default="cnn",
                   help="Must match how the checkpoint was trained (cnn, qwen10m, qwen30m, ...).")
    p.add_argument("--backbone", type=str, default="resnet34")
    p.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="split_seed values to score against. Each is a different route-level partition.")
    p.add_argument("--val_split", type=float, default=0.15,
                   help="Held-out fraction. Must match the training runs being explained.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--route_points", type=int, default=4)
    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reference_spread", type=float, default=None,
                   help="Across-seed training spread from compare_wor_runs.py, e.g. 0.0308. "
                        "When given, the verdict at the end is stated against it.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 74)
    print("  Held-out split sensitivity - one fixed checkpoint, several route splits")
    print("=" * 74)
    print("  checkpoint : %s" % args.checkpoint)
    print("  arch       : %s | backbone %s" % (args.policy_arch, args.backbone))
    print("  val_split  : %.2f | splits %s" % (args.val_split, args.splits))
    print()

    # vision_grid / pool_vision are recovered from the checkpoint's own config block
    # (13.10), so the geometry cannot silently disagree with how it was trained.
    model = load_wor_model(
        checkpoint_path=args.checkpoint,
        backbone_name=args.backbone,
        policy_arch=args.policy_arch,
        device=args.device
    )
    model.eval()

    losses, ades = [], []
    for split_seed in args.splits:
        # Only the val half is used. The train loader is built and discarded because the
        # split is defined by the pair - asking for the val half alone would not produce
        # the same partition.
        _, val_loader = create_wor_train_val_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            route_points=args.route_points,
            val_split=args.val_split,
            split_seed=split_seed
        )
        if val_loader is None:
            sys.exit("[FATAL] No validation loader for split_seed=%d - check --data_dir and --val_split."
                     % split_seed)

        metrics = run_validation(model, val_loader, args.device, autocast, bool(args.use_amp))
        loss, ade = metrics["val_loss"], metrics.get("val_wp_ade_m", float("nan"))
        losses.append(loss)
        ades.append(ade)
        print("  split_seed %-3d  n=%-6d  val_loss %.4f   ADE %.4f m"
              % (split_seed, len(val_loader.dataset), loss, ade))

    if len(losses) < 2:
        sys.exit("\nNeed at least two splits to measure a spread.")

    mean = statistics.mean(losses)
    spread = max(losses) - min(losses)
    pct = 100.0 * spread / mean

    print()
    print("-" * 74)
    print("  mean val_loss over %d splits : %.4f" % (len(losses), mean))
    print("  spread (max - min)           : %.4f  (%.1f%% of mean)" % (spread, pct))
    print("  ADE spread                   : %.4f m" % (max(ades) - min(ades)))
    print("-" * 74)

    if args.reference_spread is None:
        print("\n  Pass --reference_spread <across-seed spread from compare_wor_runs.py> for a verdict.")
        return

    ref = args.reference_spread
    ratio = spread / ref if ref > 0 else float("inf")
    print("\n  Across-seed training spread  : %.4f" % ref)
    print("  This split spread            : %.4f  (%.2fx the training spread)" % (spread, ratio))
    print()
    if ratio >= 0.7:
        print("  VERDICT: the held-out set is the dominant source of variance. Most of what")
        print("  looks like run-to-run training noise is the evaluation protocol resampling")
        print("  a small number of independent routes. Enlarge or restructure the held-out")
        print("  set before spending compute on architecture arms - until then a real")
        print("  improvement and a lucky split are indistinguishable.")
    elif ratio <= 0.3:
        print("  VERDICT: the held-out set is stable and the across-seed spread is genuine")
        print("  training variance. Weight averaging and more seeds are the levers that")
        print("  lower the detection threshold; the val set is not the problem.")
    else:
        print("  VERDICT: mixed - the split contributes a substantial but not dominant share.")
        print("  Both levers help. Averaging each configuration over several splits is the")
        print("  cheapest immediate win, since it costs no additional training.")


if __name__ == "__main__":
    main()
