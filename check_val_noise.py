"""check_val_noise.py - how far can the held-out number move on its own?

Challenge Group 13.13 measured a 4.3% across-seed spread on `qwen30m`'s held-out loss
against a 5.4% margin over the conv head. The margin only just clears the spread, so
that spread is the detection threshold for every change proposed after it (13.14) - and
it matters where it comes from. Two candidates, opposite fixes:

  training noise     - same data, different initialisation and batch order, different
                       resting place. Fixed by weight averaging, longer schedules, seeds.
  measurement noise  - the model is stable, the held-out SET is not. 15% of 9,632 frames
                       is ~1,445 frames but only ~19 routes, and frames within a route
                       are near-duplicates. Fixed by a larger evaluation set and nothing
                       else: no architectural change is visible until it lands.

WHAT THIS SCRIPT DOES NOT DO, AND WHY
-------------------------------------
The obvious test - score one checkpoint against several `--split_seed` values and look
at the spread - is invalid, and the first version of this script did exactly that. A
model trained under split_seed 0 saw 110 of the 129 routes. Re-splitting with seed k
draws a fresh ~19 val routes from all 129, so roughly 85% of them are routes the model
TRAINED on. The resulting "held-out" loss is mostly training loss, it drops far below
the real number, and it varies wildly with how many genuinely-unseen routes happen to
land in the draw. That measures contamination, not noise.

The valid version resamples only WITHIN the true held-out set. Per-frame losses are
computed once over the routes the model never saw, then bootstrapped at ROUTE
granularity - resampling routes with replacement, because routes are the independent
unit and frames inside one are not. The spread of those resamples is the sampling error
of the held-out estimate, directly comparable to the across-seed training spread.

Read the result against `compare_wor_runs.py`:

  bootstrap spread ~= across-seed spread  -> the val set is the bottleneck. Enlarge or
                                             restructure it before any architecture arm.
  bootstrap spread << across-seed spread  -> the val set is fine; the spread is genuine
                                             training variance. EMA / more seeds.

It also reports the train/held-out gap, which costs one extra pass and is the number
that says whether regularisation or capacity is the live problem at all.

Usage:
  python check_val_noise.py --checkpoint /workspace/checkpoints/wor_sweep/qwen30m_s0/best_model.pth \
      --data_dir /workspace/dataset/wor_trajectories --policy_arch qwen30m --reference_spread 0.0308
"""
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_dataset import WorldOnRailsDataset, route_group_split
from src.training.wor_eval import to_device_batch

try:
    from torch.cuda.amp import autocast
except ImportError:  # pragma: no cover
    from torch.amp import autocast

LATERAL_W = 3.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--policy_arch", type=str, default="cnn")
    p.add_argument("--backbone", type=str, default="resnet34")
    p.add_argument("--val_split", type=float, default=0.15,
                   help="Must match the training run, or the reconstructed split is not its split.")
    p.add_argument("--split_seed", type=int, default=0,
                   help="The split_seed the checkpoint was TRAINED under. Getting this wrong "
                        "silently scores the model on its own training frames.")
    p.add_argument("--bootstrap", type=int, default=2000, help="Route-level bootstrap resamples.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--route_points", type=int, default=4)
    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reference_spread", type=float, default=None,
                   help="Across-seed spread from compare_wor_runs.py, e.g. 0.0308.")
    p.add_argument("--train_subset", type=int, default=2000,
                   help="Frames of the TRAIN split to score for the generalisation gap (0 to skip).")
    return p.parse_args()


@torch.no_grad()
def per_frame_losses(model, dataset, indices, args):
    """Returns the per-frame loss for `indices`, in that order.

    Mirrors `waypoint_losses` with q_loss_weight=0: longitudinal + 3 * lateral, L1,
    meaned over the 5 waypoints of each frame. Kept per-frame rather than aggregated
    because the bootstrap has to regroup them by route afterwards.
    """
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    out_losses = []
    for batch in loader:
        rgb, speed, command, route, _target_q, target_wp = to_device_batch(batch, args.device)
        with autocast(enabled=bool(args.use_amp)):
            out = model(rgb, speed, command, route)
        pred = out["selected_waypoints"].float()
        tgt = target_wp.float()
        per = (pred[..., 0] - tgt[..., 0]).abs() + LATERAL_W * (pred[..., 1] - tgt[..., 1]).abs()
        out_losses.append(per.mean(dim=1).cpu().numpy())  # mean over the 5 waypoints
    return np.concatenate(out_losses) if out_losses else np.array([])


def main():
    args = parse_args()

    print("=" * 76)
    print("  Held-out sampling error - route-level bootstrap within the true val set")
    print("=" * 76)
    print("  checkpoint : %s" % args.checkpoint)
    print("  arch       : %s | backbone %s" % (args.policy_arch, args.backbone))
    print("  split      : val_split %.2f, split_seed %d (must match training)"
          % (args.val_split, args.split_seed))
    print()

    dataset = WorldOnRailsDataset(data_dir=args.data_dir, is_train=False,
                                  synthetic_samples=0, route_points=args.route_points)
    train_idx, val_idx, groups, _keys = route_group_split(dataset, args.val_split, args.split_seed)
    if not val_idx:
        sys.exit("[FATAL] Empty validation split - check --data_dir and --val_split.")

    # Route key per held-out frame, so the bootstrap can resample routes not frames.
    frame_to_route = {}
    for key, idxs in groups.items():
        for i in idxs:
            frame_to_route[i] = key
    val_routes = sorted({frame_to_route[i] for i in val_idx})
    print("  held-out   : %d frames across %d routes (of %d total routes)"
          % (len(val_idx), len(val_routes), len(groups)))

    model = load_wor_model(checkpoint_path=args.checkpoint, backbone_name=args.backbone,
                           policy_arch=args.policy_arch, device=args.device)
    model.eval()

    losses = per_frame_losses(model, dataset, val_idx, args)
    by_route = {}
    for pos, i in enumerate(val_idx):
        by_route.setdefault(frame_to_route[i], []).append(losses[pos])
    route_means = {k: float(np.mean(v)) for k, v in by_route.items()}
    route_counts = {k: len(v) for k, v in by_route.items()}

    point = float(losses.mean())
    print("\n  held-out loss (all %d frames) : %.4f" % (len(losses), point))

    # Bootstrap over routes, weighting each drawn route by its frame count so the
    # resampled statistic estimates the same frame-mean quantity as the point estimate.
    keys = list(route_means)
    means = np.array([route_means[k] for k in keys])
    counts = np.array([route_counts[k] for k in keys], dtype=float)
    rng = np.random.RandomState(0)
    draws = rng.randint(0, len(keys), size=(args.bootstrap, len(keys)))
    boot = (means[draws] * counts[draws]).sum(axis=1) / counts[draws].sum(axis=1)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    spread95 = hi - lo
    print("  bootstrap over %d routes      : sd %.4f | 95%% CI [%.4f, %.4f] | width %.4f"
          % (len(keys), boot.std(), lo, hi, spread95))
    print("  worst / best single route     : %.4f / %.4f"
          % (max(route_means.values()), min(route_means.values())))

    if args.train_subset > 0 and train_idx:
        sub = train_idx[:: max(1, len(train_idx) // args.train_subset)][:args.train_subset]
        train_loss = float(per_frame_losses(model, dataset, sub, args).mean())
        print("\n  train-split loss (%d frames)  : %.4f" % (len(sub), train_loss))
        print("  generalisation gap             : %.4f  (held-out is %.2fx train)"
              % (point - train_loss, point / max(1e-9, train_loss)))
        if point / max(1e-9, train_loss) > 2.0:
            print("  >> The model is memorising. Regularisation and data volume are live")
            print("     levers here; capacity is not. Any architecture arm run in this")
            print("     regime is comparing how gracefully models overfit.")

    if args.reference_spread is None:
        print("\n  Pass --reference_spread <across-seed spread> for a verdict.")
        return

    ref = args.reference_spread
    ratio = spread95 / ref if ref > 0 else float("inf")
    print("\n" + "-" * 76)
    print("  across-seed training spread   : %.4f" % ref)
    print("  held-out 95%% CI width         : %.4f  (%.2fx the training spread)" % (spread95, ratio))
    print("-" * 76)
    if ratio >= 0.7:
        print("\n  VERDICT: the held-out set is a dominant source of variance. Enlarge or")
        print("  restructure it before spending compute on architecture arms - at this")
        print("  size a real improvement and a lucky draw of routes look the same.")
    elif ratio <= 0.3:
        print("\n  VERDICT: the held-out set is stable; the across-seed spread is genuine")
        print("  training variance. Weight averaging and more seeds are the levers that")
        print("  lower the detection threshold. The val set is not the problem.")
    else:
        print("\n  VERDICT: mixed. Both levers help; averaging each configuration over")
        print("  several splits is the cheapest win since it costs no extra training.")


if __name__ == "__main__":
    main()
