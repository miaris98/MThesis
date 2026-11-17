"""How much "is there an obstacle at this point of my path" is readable from the frozen camera features? (2026-11-08)

Reads one or more pooled caches built with --route_windows K --window_labels 1 (optionally --side_windows / --decoder_stats) and fits small
probes from each window's features to its path-relative obstacle label (occupancy: an obstacle footprint within 1.5 m of the window's
ground point; logged boxes are labels only). Frames split by route folder (15% held out, fixed hash), so a route never sits on both
sides. Reports held-out ROC AUC and average precision of
  enc     a linear probe on the window's encoder features (1512 channels)
  enc_mlp a 1-hidden-layer probe on the same features
  dstat   a linear probe on the 10 TF++ decoder statistics alone (depth, depth / road depth, class fractions), when present
  both    encoder features + decoder statistics (linear)
for route windows and side windows separately. A feature set that cannot be probed for obstacles cannot give them to the policy either.

    python scripts/analysis/window_obstacle_probe.py --cache NAME=PREFIX [NAME=PREFIX ...] --k 8 --out_json probe.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from src.training.pooled_feature_cache import DSTAT_CHANNELS, window_count, window_file, window_label_valid_file, window_side_files  # noqa: E402


def route_of(key: str) -> str:
    parts = key.replace("\\", "/").split("/")
    return "/".join(parts[:-2])          # <town>/data/<scenario>/<route>/rgb[_augmented]/<frame> -> the route folder


def is_val(route: str) -> bool:
    return int(hashlib.md5(route.encode()).hexdigest(), 16) % 100 < 15


def auc_ap(score: np.ndarray, y: np.ndarray):
    o = np.argsort(-score)
    y = y[o].astype(np.float64)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    auc = float((getattr(np, "trapezoid", None) or np.trapz)(np.concatenate([[0], tp / P]), np.concatenate([[0], fp / N])))
    prec = tp / np.arange(1, len(y) + 1)
    ap = float((prec * y).sum() / P)
    return auc, ap


def fit(Xtr, ytr, Xva, hidden=0, epochs=6, device="cuda"):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-4
    Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    Xva = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device)
    d = Xtr.shape[1]
    net = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1)) if hidden else nn.Linear(d, 1)).to(device)
    pos = float(ytr.mean().clamp(1e-4, 1 - 1e-4))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            lossf(net(Xtr[b]).squeeze(1), ytr[b]).backward()
            opt.step()
    with torch.no_grad():
        return torch.cat([net(Xva[i:i + 16384]).squeeze(1) for i in range(0, len(Xva), 16384)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", nargs="+", required=True, help="NAME=PREFIX")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=120000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default="")
    a = ap.parse_args()
    results = {}
    for spec in a.cache:
        name, prefix = spec.split("=", 1)
        meta = json.load(open(prefix + ".json"))
        N, C, keys = meta["n"], meta["channels"], meta["keys"]
        rng = np.random.RandomState(0)
        rows = np.sort(rng.choice(N, size=min(a.max_frames, N), replace=False))
        val = np.array([is_val(route_of(keys[r])) for r in rows])
        wlv = np.memmap(window_label_valid_file(prefix), dtype=np.uint8, mode="r", shape=(N,))[rows].astype(bool)
        results[name] = {}
        for kind in ("route", "side"):
            wp = window_file(prefix, a.k, kind)
            if not os.path.exists(wp):
                continue
            M = window_count(a.k, kind)
            sf = window_side_files(prefix, a.k, kind)
            feats = np.memmap(wp, dtype=np.float16, mode="r", shape=(N, C, M))
            lab = np.memmap(sf["lab"], dtype=np.float16, mode="r", shape=(N, 2, M))
            ds = np.memmap(sf["dstat"], dtype=np.float16, mode="r", shape=(N, DSTAT_CHANNELS, M)) if os.path.exists(sf["dstat"]) else None
            X, Y, D, V = [], [], [], []
            for i in range(0, len(rows), 4096):
                r = rows[i:i + 4096]
                keep = wlv[i:i + 4096]
                r, vv = r[keep], val[i:i + 4096][keep]
                X.append(np.asarray(feats[r], dtype=np.float32).transpose(0, 2, 1).reshape(-1, C))
                Y.append((np.asarray(lab[r, 0], dtype=np.float32) > 0.5).reshape(-1))
                if ds is not None:
                    D.append(np.asarray(ds[r], dtype=np.float32).transpose(0, 2, 1).reshape(-1, DSTAT_CHANNELS))
                V.append(np.repeat(vv, M))
            X, Y, V = np.concatenate(X), np.concatenate(Y), np.concatenate(V)
            D = np.concatenate(D) if D else None
            out = {"windows": int(len(Y)), "occupied_frac": float(Y.mean()), "val_windows": int(V.sum())}
            sets = {"enc": X}
            if D is not None:
                sets["dstat"] = D
                sets["both"] = np.concatenate([X, D], 1)
            for sname, F in sets.items():
                s = fit(F[~V], Y[~V], F[V], device=a.device)
                out[f"{sname}_auc"], out[f"{sname}_ap"] = auc_ap(s, Y[V])
            s = fit(X[~V], Y[~V], X[V], hidden=256, device=a.device)
            out["enc_mlp_auc"], out["enc_mlp_ap"] = auc_ap(s, Y[V])
            results[name][kind] = out
            print(f"{name:12s} {kind:5s} windows {out['windows']:8d} occupied {out['occupied_frac']:.3f} | "
                  + " | ".join(f"{k[:-4]} AUC {out[k]:.3f} AP {out[k[:-4] + '_ap']:.3f}" for k in out if k.endswith("_auc")), flush=True)
    if a.out_json:
        json.dump(results, open(a.out_json, "w"), indent=1)
    try:  # MLflow (project rule)
        import mlflow
        mlflow.set_experiment("window_obstacle_probe")
        with mlflow.start_run(run_name="probe"):
            for name, kinds in results.items():
                for kind, out in kinds.items():
                    mlflow.log_metrics({f"{name}_{kind}_{k}": v for k, v in out.items() if isinstance(v, float) and np.isfinite(v)})
    except Exception as exc:
        print(f"[mlflow skipped: {exc}]")


if __name__ == "__main__":
    main()
