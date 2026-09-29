"""How far PDM-Lite's expert-shifted `route` departs from the leaderboard plan (`route_original`), per scenario type.

TODO A1/A14 (S-096): training fed `route` as the route input and overlay, while evaluation feeds the unshifted plan.
This measures, on the downloaded data, how often `changed_route` is set and how far the first route points move
laterally (ego frame, metres) - i.e. how big the train/test mismatch was, and what `--route_key route_original` changes.

    python scripts/analysis/route_shift_stats.py /workspace/dataset/wor_trajectories [--every 10] [--points 4]
"""
import argparse
import collections
import glob
import gzip
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--every", type=int, default=10, help="read every n-th frame per route")
    ap.add_argument("--points", type=int, default=4, help="compare the first n route points (the policy sees 4)")
    a = ap.parse_args()
    stats = collections.defaultdict(lambda: {"frames": 0, "changed": 0, "has_orig": 0, "dev": []})
    for mdir in sorted(glob.glob(os.path.join(a.root, "**", "measurements"), recursive=True)):
        parts = os.path.relpath(mdir, a.root).split(os.sep)
        key = f"{parts[0]}/{parts[2] if len(parts) > 3 else parts[-2]}"  # Town/ScenarioType
        for f in sorted(glob.glob(os.path.join(mdir, "*.json.gz")))[:: a.every]:
            try:
                with gzip.open(f, "rt") as fh:
                    m = json.load(fh)
            except Exception:
                continue
            s = stats[key]; s["frames"] += 1
            s["changed"] += bool(m.get("changed_route"))
            r, ro = m.get("route"), m.get("route_original")
            if r and ro:
                s["has_orig"] += 1
                n = min(a.points, len(r), len(ro))
                d = np.abs(np.asarray(r[:n], float)[:, 1] - np.asarray(ro[:n], float)[:, 1]).max() if n else 0.0
                s["dev"].append(float(d))
    print(f"{'town/scenario':45s} {'frames':>7s} {'changed%':>8s} {'orig%':>6s} {'dev>0.5m%':>9s} {'p95 dev m':>9s}")
    for k in sorted(stats):
        s = stats[k]; dev = np.asarray(s["dev"]) if s["dev"] else np.zeros(1)
        print(f"{k:45s} {s['frames']:7d} {100 * s['changed'] / max(s['frames'], 1):8.1f} "
              f"{100 * s['has_orig'] / max(s['frames'], 1):6.1f} {100 * (dev > 0.5).mean():9.1f} {np.percentile(dev, 95):9.2f}")


if __name__ == "__main__":
    main()
