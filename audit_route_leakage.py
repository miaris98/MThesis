#!/usr/bin/env python3
"""How much of the waypoint target is already contained in the model's own inputs?

THE QUESTION
------------
The policy is given a `route` - four ego-frame points from PDM-Lite's planned path - and
asked to predict `waypoints`, the ego vehicle's next five poses. Both are ego-frame (x, y)
sequences pointing in the direction of travel. If the second is largely a function of the
first, then a model can score well by interpolating its own navigation input, and the
entire Group 13 comparison (a 13.6% margin between a conv head and a transformer head)
would be measuring which architecture interpolates a route better - not which one drives.

`train_wor.py --route_points` already records that the full 20-point route correlates
~0.84 with the lateral target. This quantifies what that means for the loss.

IS ROUTE CONDITIONING "CHEATING"?
---------------------------------
Not by itself, and the construction is clean: `route` comes from the `route` field of
PDM-Lite's measurement files (a map-level plan from the route planner), not from the
expert's realised future poses. Targets come from `ego_matrix` of the following frames.
The two are built from different sources. Conditioning a driving policy on a global plan
is also what every comparable method does - World on Rails, TransFuser, LAV, PlanT - and
the CARLA Leaderboard hands agents exactly such a plan, because without one the task is
ill-posed: nothing in a forward camera says whether you were asked to turn left or right.

The leakage risk is not the *existence* of the route input, it is its *sufficiency*. A
route dense enough and near enough becomes the answer rather than the question, and the
expert's realised path necessarily hugs the plan it was following, so the correlation is
structural rather than a bug. What matters for interpreting any architecture result is the
size of the gap between "route alone" and "route plus vision".

WHAT THIS MEASURES
------------------
No-vision baselines fitted on the training split and scored on the *same* held-out routes,
under the *same* loss, via the *same* `route_group_split`. Nothing here opens an image.

  zero              predict (0, 0) - the degenerate optimum Group 9 was stuck at
  speed             constant velocity: x from a fitted per-horizon scale on speed, y = 0
  route_linear      least squares from the 8 route numbers
  route_speed_lin   as above plus speed and command
  route_speed_mlp   a small MLP on the same features - the strongest no-vision model

Read the output against the vision models' held-out loss on this identical split
(qwen30m 0.5793, cnn 0.6616 at seed 0). A no-vision baseline that lands near those makes
the architecture comparison approximately vacuous; one that lands far above them bounds
how much of the result is really vision.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.training.wor_dataset import WorldOnRailsDataset, route_group_split

# Must match waypoint_losses(): longitudinal L1 + lateral_loss_weight * lateral L1.
LATERAL_W = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Quantify route->waypoint leakage")
    p.add_argument("--data_dir", type=str, default="/workspace/dataset/wor_trajectories")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--route_points", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def extract(dataset, indices: List[int]) -> Dict[str, np.ndarray]:
    """Pulls the non-image fields straight out of the sample index.

    Deliberately bypasses __getitem__: that decodes a JPEG per sample, and none of the
    baselines here can see it anyway. It also keeps this honest - a script auditing
    whether vision is needed should be unable to use vision by construction.
    """
    route, speed, command, wp = [], [], [], []
    for i in indices:
        s = dataset.samples[i]
        route.append(np.asarray(s["route"], dtype=np.float32).reshape(-1))
        speed.append(float(s["speed"]))
        command.append(int(s["command"]))
        wp.append(np.asarray(s["waypoints"], dtype=np.float32))
    return {
        "route": np.stack(route),                       # (N, route_points*2)
        "speed": np.asarray(speed, dtype=np.float32),   # (N,)
        "command": np.asarray(command, dtype=np.int64),
        "wp": np.stack(wp),                             # (N, 5, 2)
    }


def per_frame_loss(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N, 5, 2) -> (N,), matching waypoint_losses' per-frame quantity."""
    err = np.abs(pred - target)
    return (err[..., 0] + LATERAL_W * err[..., 1]).mean(axis=1)


def axis_errors(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    err = np.abs(pred - target)
    return float(err[..., 0].mean()), float(err[..., 1].mean())


def _design(d: Dict[str, np.ndarray], use_route: bool, use_speed: bool) -> np.ndarray:
    cols = []
    if use_route:
        cols.append(d["route"])
    if use_speed:
        cols.append(d["speed"][:, None])
        # One-hot the command: it is a 6-way categorical and PDM-Lite pins it to
        # LANEFOLLOW almost everywhere, so this mostly adds a bias column - included so
        # the baseline cannot be dismissed for lacking an input the policy is given.
        oh = np.zeros((len(d["command"]), 6), dtype=np.float32)
        oh[np.arange(len(d["command"])), np.clip(d["command"], 0, 5)] = 1.0
        cols.append(oh)
    if not cols:
        return np.zeros((len(d["speed"]), 0), dtype=np.float32)
    return np.concatenate(cols, axis=1).astype(np.float32)


def fit_linear(Xtr, Ytr, Xva):
    """Least squares with an intercept, solved directly - no iteration to tune."""
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1), dtype=np.float32)], axis=1)
    B = np.concatenate([Xva, np.ones((len(Xva), 1), dtype=np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(A, Ytr.reshape(len(Ytr), -1), rcond=None)
    return (B @ coef).reshape(len(Xva), -1, 2)


def fit_mlp(Xtr, Ytr, Xva, seed: int = 0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=400,
                       early_stopping=True, random_state=seed)
    mlp.fit(scaler.transform(Xtr), Ytr.reshape(len(Ytr), -1))
    return mlp.predict(scaler.transform(Xva)).reshape(len(Xva), -1, 2)


def fit_speed_only(tr, va):
    """Constant-velocity: a per-horizon scale on speed, lateral pinned to zero.

    This is 13.26's argument made concrete. If most of the loss is longitudinal and
    longitudinal displacement is close to the integral of a speed the policy is handed,
    then a single fitted scalar per horizon should already capture most of it.
    """
    scales = []
    for k in range(tr["wp"].shape[1]):
        denom = float((tr["speed"] ** 2).sum())
        scales.append(float((tr["speed"] * tr["wp"][:, k, 0]).sum() / denom) if denom > 0 else 0.0)
    pred = np.zeros_like(va["wp"])
    for k, s in enumerate(scales):
        pred[:, k, 0] = s * va["speed"]
    return pred, scales


def route_level_means(losses: np.ndarray, val_idx: List[int],
                      groups: Dict[str, List[int]]) -> Dict[str, float]:
    """Regroups per-frame losses by route. Routes are the independent unit - frames
    inside one are near-duplicates - so every interval in this project is computed over
    routes, not frames."""
    pos = {idx: k for k, idx in enumerate(val_idx)}
    out = {}
    for key, idxs in groups.items():
        vals = [losses[pos[i]] for i in idxs if i in pos]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def main():
    args = parse_args()
    ds = WorldOnRailsDataset(data_dir=args.data_dir, route_points=args.route_points)
    train_idx, val_idx, groups, _keys = route_group_split(ds, args.val_split, args.split_seed)
    print(f"  frames: {len(ds)}  train {len(train_idx)}  val {len(val_idx)}  "
          f"routes {len(groups)}")

    tr, va = extract(ds, train_idx), extract(ds, val_idx)

    # How much of the target is linearly present in the input, before any model.
    flat_wp = va["wp"].reshape(len(va["wp"]), -1)
    corr_lat = float(np.corrcoef(va["route"][:, 1::2].mean(axis=1),
                                 va["wp"][:, :, 1].mean(axis=1))[0, 1])
    corr_lon = float(np.corrcoef(va["route"][:, 0::2].mean(axis=1),
                                 va["wp"][:, :, 0].mean(axis=1))[0, 1])
    print(f"  corr(route_y, target_y) = {corr_lat:+.3f}   "
          f"corr(route_x, target_x) = {corr_lon:+.3f}")

    results = []

    def record(name: str, pred: np.ndarray, note: str = ""):
        losses = per_frame_loss(pred, va["wp"])
        lon, lat = axis_errors(pred, va["wp"])
        rl = route_level_means(losses, val_idx, groups)
        vals = np.array(list(rl.values()))
        rng = np.random.RandomState(0)
        boot = vals[rng.randint(0, len(vals), size=(args.bootstrap, len(vals)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results.append({
            "name": name, "loss": float(losses.mean()),
            "route_mean": float(vals.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "lon_err_m": lon, "lat_err_m": lat, "note": note,
        })

    record("zero", np.zeros_like(va["wp"]), "predict nothing")

    pred_speed, scales = fit_speed_only(tr, va)
    record("speed", pred_speed, f"x=s*t, scales={[round(s,3) for s in scales]}")

    Xtr_r, Xva_r = _design(tr, True, False), _design(va, True, False)
    record("route_linear", fit_linear(Xtr_r, tr["wp"], Xva_r), "least squares on route")

    Xtr_rs, Xva_rs = _design(tr, True, True), _design(va, True, True)
    record("route_speed_lin", fit_linear(Xtr_rs, tr["wp"], Xva_rs), "+ speed, command")

    try:
        record("route_speed_mlp", fit_mlp(Xtr_rs, tr["wp"], Xva_rs),
               "256x256 MLP, no vision")
    except ImportError:
        print("  [skip] scikit-learn unavailable - MLP baseline not run")

    print()
    print("=" * 84)
    print(f"  {'baseline':<18}{'val_loss':>10}{'95% CI (route-level)':>26}"
          f"{'lon m':>9}{'lat m':>9}")
    print("=" * 84)
    for r in results:
        print(f"  {r['name']:<18}{r['loss']:>10.4f}"
              f"{'[' + format(r['ci_lo'], '.4f') + ', ' + format(r['ci_hi'], '.4f') + ']':>26}"
              f"{r['lon_err_m']:>9.4f}{r['lat_err_m']:>9.4f}")
    print("=" * 84)
    print("  For reference, WITH vision on this identical split (seed 0):")
    print(f"  {'qwen30m':<18}{0.5793:>10.4f}")
    print(f"  {'cnn':<18}{0.6616:>10.4f}")
    print()
    best = min(results, key=lambda r: r["loss"])
    print(f"  Best no-vision baseline: {best['name']} at {best['loss']:.4f}.")
    print(f"  Vision buys {best['loss'] - 0.5793:+.4f} over it "
          f"({100.0 * (best['loss'] - 0.5793) / best['loss']:+.1f}%).")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"corr_lat": corr_lat, "corr_lon": corr_lon,
                       "val_split": args.val_split, "split_seed": args.split_seed,
                       "route_points": args.route_points,
                       "num_val_routes": len(set(groups) & {k for k in groups}),
                       "results": results}, f, indent=2)
        print(f"✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
