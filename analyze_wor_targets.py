#!/usr/bin/env python3
"""Diagnose whether the WoR waypoint targets carry a learnable lateral signal.

Lat Err has sat at ~0.24m across every training configuration tried (frozen and
fine-tuned backbone, lateral loss weight 1.0 and 3.0, epoch 1 and epoch 50)
while Lon Err fell from 2.70m to 0.56m. A quantity that never improves from its
epoch-1 value is usually not a capacity problem - so this checks the targets
themselves.

The decisive number is the "predict a constant" baseline: if the trained model's
lateral MAE matches what you'd get by ignoring the image entirely and always
predicting the dataset's mean lateral offset, then the model has learned nothing
about steering, and the question becomes whether the signal is there to learn.

Usage:
    python analyze_wor_targets.py --data_dir /workspace/dataset/wor_trajectories
"""
import argparse
import numpy as np

from src.training.wor_dataset import WorldOnRailsDataset


def parse_args():
    p = argparse.ArgumentParser(description="Analyze WoR waypoint target distributions")
    p.add_argument("--data_dir", type=str, default="/workspace/dataset/wor_trajectories")
    return p.parse_args()


def main():
    args = parse_args()

    # Waypoints are precomputed at index time, so this needs no images and no GPU.
    ds = WorldOnRailsDataset(data_dir=args.data_dir, cache_decoded=False)
    if getattr(ds, "is_synthetic", False):
        print(f"[Error] No real frames indexed under {args.data_dir} - nothing to analyze.")
        return

    wps = np.array([s["waypoints"] for s in ds.samples if s.get("format") == "pdm_lite"],
                   dtype=np.float32)
    if len(wps) == 0:
        print("[Error] No pdm_lite samples with waypoints found.")
        return

    lon, lat = wps[..., 0], wps[..., 1]          # (N, 5) each
    lat_final, lon_final = lat[:, -1], lon[:, -1]

    print(f"\n=== WoR target analysis: {len(wps)} samples, {wps.shape[1]} waypoints each ===\n")

    for name, arr in (("LONGITUDINAL (x, forward)", lon), ("LATERAL (y, steering)", lat)):
        print(f"{name}")
        print(f"    mean {arr.mean():+.4f}m | std {arr.std():.4f}m | "
              f"MAE-about-mean {np.abs(arr - arr.mean()).mean():.4f}m")
        print(f"    abs percentiles  p50 {np.percentile(np.abs(arr), 50):.3f}m | "
              f"p90 {np.percentile(np.abs(arr), 90):.3f}m | "
              f"p99 {np.percentile(np.abs(arr), 99):.3f}m | max {np.abs(arr).max():.3f}m")

    # How much of the data actually turns? If steering frames are rare, an L1 loss is
    # minimized by predicting ~straight everywhere, and the rare turns never dominate
    # enough gradient to be learned.
    print("\nFraction of samples by final-waypoint lateral offset:")
    for thr in (0.25, 0.5, 1.0, 2.0, 5.0):
        frac = float((np.abs(lat_final) > thr).mean())
        print(f"    |lateral| > {thr:>4.2f}m : {frac * 100:6.2f}%")

    # The baseline the trained model has to beat to have learned anything at all.
    const_zero_mae = float(np.abs(lat).mean())
    const_mean_mae = float(np.abs(lat - lat.mean()).mean())
    print("\n--- Baselines the model must beat on lateral error ---")
    print(f"    predict always 0.0       -> MAE {const_zero_mae:.4f}m")
    print(f"    predict dataset mean     -> MAE {const_mean_mae:.4f}m")
    print(f"    trained model reported   -> MAE ~0.24m (every run this session)")
    if abs(const_zero_mae - 0.24) < 0.05 or abs(const_mean_mae - 0.24) < 0.05:
        print("\n    >>> The model's lateral error matches a constant predictor: it is NOT")
        print("        using the image to steer at all. Either the turning frames are too")
        print("        rare to shape the loss, or image and waypoints aren't aligned.")
    else:
        print("\n    >>> The model beats the constant baseline, so some lateral signal IS")
        print("        being learned - the plateau is then a capacity/feature limit.")

    # Straight-vs-turning split: if the model can't beat a constant, it's worth knowing
    # whether turning frames even exist in useful numbers.
    turning = np.abs(lat_final) > 0.5
    print(f"\nTurning frames (|final lateral| > 0.5m): {turning.sum()} of {len(lat_final)} "
          f"({turning.mean() * 100:.2f}%)")
    if turning.sum():
        print(f"    their mean |lateral| : {np.abs(lat_final[turning]).mean():.3f}m")
        print(f"    their mean |longitudinal| : {np.abs(lon_final[turning]).mean():.3f}m")


if __name__ == "__main__":
    main()
