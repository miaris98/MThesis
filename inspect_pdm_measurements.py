#!/usr/bin/env python3
"""Dump the real schema of PDM-Lite measurement files.

The waypoint analysis showed every one of 9,640 samples carrying command
LANEFOLLOW, in a dataset full of junction-turn scenarios. wor_dataset reads the
command as `meas.get("command", meas.get("next_command", 4))`, so if neither key
exists every frame silently falls back to LANEFOLLOW - which is what the numbers
look like. Those key names were assumed rather than checked; this prints what the
files actually contain so the navigation signal can be wired to the right field.

Usage:
    python inspect_pdm_measurements.py --data_dir /workspace/dataset/wor_trajectories
"""
import argparse
import glob
import gzip
import json
import os
from collections import Counter

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Inspect PDM-Lite measurement schema")
    p.add_argument("--data_dir", type=str, default="/workspace/dataset/wor_trajectories")
    p.add_argument("--sample", type=int, default=2000, help="How many measurement files to sample")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "measurements", "*.json.gz"),
                             recursive=True))
    if not files:
        print(f"[Error] No measurement files under {args.data_dir}")
        return
    print(f"Found {len(files)} measurement files.\n")

    with gzip.open(files[0], "rt") as f:
        first = json.load(f)

    print(f"=== Schema of {files[0]} ===")
    for k in sorted(first.keys()):
        v = first[k]
        if isinstance(v, list):
            arr = np.array(v)
            desc = f"list shape={arr.shape}"
            if arr.size <= 8:
                desc += f" value={v}"
        else:
            desc = f"{type(v).__name__} value={v}"
        print(f"    {k:<28} {desc}")

    # Anything that might carry the navigation intent.
    candidates = [k for k in first
                  if any(t in k.lower() for t in ("command", "target", "route", "junction",
                                                  "waypoint", "next", "steer"))]
    print(f"\n=== Navigation-related keys: {candidates} ===")

    step = max(1, len(files) // args.sample)
    sampled = files[::step][:args.sample]
    print(f"\nSampling {len(sampled)} files for value distributions...\n")

    values = {k: [] for k in candidates}
    for path in sampled:
        try:
            with gzip.open(path, "rt") as f:
                d = json.load(f)
        except Exception:
            continue
        for k in candidates:
            if k in d:
                values[k].append(d[k])

    for k, vals in values.items():
        if not vals:
            continue
        print(f"--- {k} ({len(vals)} samples) ---")
        if isinstance(vals[0], (int, float, bool)):
            counts = Counter(vals)
            if len(counts) <= 12:
                for val, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                    print(f"    {val!r:>12} : {n:>6} ({n / len(vals) * 100:5.1f}%)")
            else:
                arr = np.array(vals, dtype=np.float64)
                print(f"    numeric: mean {arr.mean():+.3f} std {arr.std():.3f} "
                      f"min {arr.min():+.3f} max {arr.max():+.3f} ({len(counts)} distinct)")
        elif isinstance(vals[0], list):
            arr = np.array(vals, dtype=np.float64)
            print(f"    array shape {arr.shape}")
            flat = arr.reshape(len(arr), -1)
            for i in range(min(flat.shape[1], 4)):
                col = flat[:, i]
                print(f"    dim{i}: mean {col.mean():+.3f} std {col.std():.3f} "
                      f"min {col.min():+.3f} max {col.max():+.3f}")
        print()


if __name__ == "__main__":
    main()
