#!/usr/bin/env python3
"""Find which PDM-Lite field actually predicts the lateral offset we need to learn.

The policy has no working navigation input: `command` is 4 (LANEFOLLOW) in 100%
of frames, so its embedding is a constant and the model sees only image + speed.
At a junction that makes left and right indistinguishable, so predicting straight
is the correct response to its inputs - which is exactly the failure observed
(lateral MAE 0.24m == the predict-always-zero baseline).

The schema offers several candidate navigation signals. Rather than assume which
one is right (assumptions about `command` and `next_command` already cost two
training runs), this measures each one's correlation with the lateral target the
model is failing to predict. The winner is what the policy should be conditioned
on.

It also resolves whether `target_point` is stored in global or ego coordinates,
by transforming it through ego_matrix and seeing which version behaves like a
goal that is consistently ahead of the vehicle.

Usage:
    python validate_nav_signal.py --data_dir /workspace/dataset/wor_trajectories
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Validate PDM-Lite navigation signals")
    p.add_argument("--data_dir", type=str, default="/workspace/dataset/wor_trajectories")
    p.add_argument("--max_routes", type=int, default=40)
    p.add_argument("--pred_len", type=int, default=5)
    return p.parse_args()


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    args = parse_args()
    route_dirs = sorted({os.path.dirname(p) for p in glob.glob(
        os.path.join(args.data_dir, "**", "measurements", "*.json.gz"), recursive=True)})
    if not route_dirs:
        print(f"[Error] No measurement dirs under {args.data_dir}")
        return
    route_dirs = route_dirs[:args.max_routes]
    print(f"Scanning {len(route_dirs)} routes (pred_len={args.pred_len})...\n")

    rows = {k: [] for k in ("lat_target", "lon_target", "tp_raw_y", "tp_ego_x", "tp_ego_y",
                            "route_last_y", "route_mid_y", "aim_y", "steer", "junction")}
    pos_matches_ego = []

    for rd in route_dirs:
        files = sorted(glob.glob(os.path.join(rd, "*.json.gz")))
        frames = []
        for p in files:
            try:
                frames.append(load(p))
            except Exception:
                frames.append(None)

        for i in range(5, len(frames) - args.pred_len - 2):
            cur, fut = frames[i], frames[i + args.pred_len]
            if cur is None or fut is None:
                continue
            try:
                ego = np.array(cur["ego_matrix"], dtype=np.float64)
                ego_inv = np.linalg.inv(ego)
                rel = ego_inv @ np.array(fut["ego_matrix"], dtype=np.float64)
                lon_t, lat_t = rel[0, 3], rel[1, 3]

                # Is ego_matrix's translation the same thing as pos_global? If so,
                # ego_matrix is ego->world and a global target_point can be mapped
                # into the ego frame with its inverse.
                pg = np.array(cur["pos_global"], dtype=np.float64)
                pos_matches_ego.append(float(np.abs(ego[:2, 3] - pg).max()))

                tp = np.array(cur["target_point"], dtype=np.float64)
                tp_ego = ego_inv @ np.array([tp[0], tp[1], 0.0, 1.0])

                route = np.array(cur["route"], dtype=np.float64)
                aim = np.array(cur["aim_wp"], dtype=np.float64)

                rows["lat_target"].append(lat_t)
                rows["lon_target"].append(lon_t)
                rows["tp_raw_y"].append(tp[1])
                rows["tp_ego_x"].append(tp_ego[0])
                rows["tp_ego_y"].append(tp_ego[1])
                rows["route_last_y"].append(route[-1, 1])
                rows["route_mid_y"].append(route[len(route) // 2, 1])
                rows["aim_y"].append(aim[1])
                rows["steer"].append(float(cur.get("steer", 0.0)))
                rows["junction"].append(1.0 if cur.get("junction") else 0.0)
            except Exception:
                continue

    n = len(rows["lat_target"])
    if n == 0:
        print("[Error] No usable frames.")
        return
    print(f"Collected {n} frames.\n")

    md = float(np.max(pos_matches_ego))
    print(f"=== Coordinate frame of ego_matrix ===")
    print(f"    max |ego_matrix[:2,3] - pos_global| = {md:.4f}")
    print("    -> ego_matrix translation IS the global position; target_point can be "
          "mapped to ego via its inverse." if md < 1.0 else
          "    -> ego_matrix translation is NOT pos_global; treat the transform with care.")

    print(f"\n=== target_point: global or already ego-frame? ===")
    for label, key in (("raw target_point[1]", "tp_raw_y"),
                       ("ego-transformed x (forward)", "tp_ego_x"),
                       ("ego-transformed y (lateral)", "tp_ego_y")):
        a = np.array(rows[key])
        print(f"    {label:<30} mean {a.mean():+9.3f}  std {a.std():8.3f}  "
              f"min {a.min():+9.3f}  max {a.max():+9.3f}")
    fwd = np.array(rows["tp_ego_x"])
    print(f"    fraction of ego-transformed target points AHEAD of the car: "
          f"{(fwd > 0).mean() * 100:.1f}%")
    print("    (a correct ego-frame transform should put the goal ahead nearly always)")

    print(f"\n=== Which signal predicts the lateral target? (|corr| closer to 1 is better) ===")
    lat = rows["lat_target"]
    cands = [("target_point ego lateral", "tp_ego_y"),
             ("target_point raw [1]", "tp_raw_y"),
             ("route[-1] lateral", "route_last_y"),
             ("route[mid] lateral", "route_mid_y"),
             ("aim_wp lateral", "aim_y"),
             ("expert steer", "steer"),
             ("junction flag", "junction")]
    scored = sorted(((abs(corr(rows[k], lat)), name, corr(rows[k], lat))
                     for name, k in cands), reverse=True)
    for _, name, c in scored:
        print(f"    {name:<28} corr = {c:+.4f}")

    best_abs, best_name, _ = scored[0]
    print(f"\n    >>> Strongest signal: {best_name} (|corr| {best_abs:.3f})")
    if best_abs < 0.3:
        print("        Weak across the board - conditioning alone may not be enough and the")
        print("        turning frames likely need reweighting too.")
    else:
        print("        Conditioning the policy on this should let it learn to steer.")


if __name__ == "__main__":
    main()
