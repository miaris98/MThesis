"""Cut-in token check (2026-11-19, step 8a of the teacher-ladder plan): is the parked car of ParkingCutIn / StaticCutIn encoded the same way in
the PDM-Lite logs (training tokens) as live (B2D_PRIV_DUMP)?

The teacher deadlocks beside a stationary car next to its lane (live tokens: class "car", speed 0, y about +-3 m, nothing ahead in lane).
If the logs mark the same cars "static_car", or never show them stationary beside the ego, the deadlock is a train/closed-loop mismatch
(cause b), not missing lane information (cause a).

    python scripts/analysis/cutin_token_check.py --prefix /workspace/pooled_cov_ov0/<p> --live_dir /workspace/ladder/privdump/w8

Log side: <prefix>.boxtok20.f16 (all objects, build_box_targets.py) for frames whose key contains a cut-in family. Live side: the per-tick
token dumps (<route tag>_tokens.npz) of routes whose name contains the family. Same counting on both sides.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.training.build_box_targets import BOX_CLASSES, BOX_FEATS, VALID_SLOT  # noqa: E402

FAMILIES = ("ParkingCutIn", "StaticCutIn")


def side_cars(tok: np.ndarray) -> dict:
    """Counts over frames (T, M, BOX_FEATS): vehicles beside the ego lane (0 < x < 12 m, 2 < |y| < 4.5 m), split by class and motion,
    and whether the ego lane ahead (0 < x < 15 m, |y| < 1.5 m) holds any object in the same frame."""
    c = Counter()
    for t in tok:
        v = t[:, VALID_SLOT] > 0.5
        o = t[v]
        if not len(o):
            c["frames"] += 1
            continue
        x, y, spd = o[:, 0] * 20, o[:, 1] * 10, o[:, 6] * 10
        cls = np.array(BOX_CLASSES)[o[:, 9:9 + len(BOX_CLASSES)].argmax(1)]
        veh = np.isin(cls, ("car", "static_car"))
        beside = veh & (x > 0) & (x < 12) & (np.abs(y) > 2) & (np.abs(y) < 4.5)
        ahead = (x > 0) & (x < 15) & (np.abs(y) < 1.5) & ~np.isin(cls, ("traffic_light", "stop_sign"))
        c["frames"] += 1
        if beside.any():
            c["frames_with_side_vehicle"] += 1
            if not ahead.any():
                c["frames_side_vehicle_lane_clear"] += 1
        for k in np.where(beside)[0]:
            c[f"side {cls[k]} {'stationary' if spd[k] < 0.5 else 'moving'}"] += 1
    return c


def report(name: str, c: Counter) -> None:
    n = max(c["frames"], 1)
    print(f"  {name}: {c['frames']} frames; with a vehicle beside the lane {c['frames_with_side_vehicle'] / n:.1%} "
          f"(lane ahead clear in {c['frames_side_vehicle_lane_clear'] / n:.1%})")
    tot = sum(v for k, v in c.items() if k.startswith("side "))
    for k, v in sorted(((k, v) for k, v in c.items() if k.startswith("side ")), key=lambda kv: -kv[1]):
        print(f"      {k[5:]:24s} {v:7d}  ({v / max(tot, 1):.1%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="pooled-cache prefix(es) with <prefix>.boxtok<m>.f16 (all objects), comma-separated")
    ap.add_argument("--live_dir", default="", help="B2D_PRIV_DUMP directory with <route tag>_tokens.npz")
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--out", default="", help="optional JSON with the counts")
    a = ap.parse_args()
    res = {}
    for fam in FAMILIES:
        print(f"== {fam}")
        log = Counter()
        for p in [p for p in a.prefix.split(",") if p]:
            meta = json.load(open(p + ".json"))
            rows = [i for i, k in enumerate(meta["keys"]) if f"/{fam}/" in k]
            if not rows:
                continue
            tm = np.memmap(f"{p}.boxtok{a.m}.f16", dtype=np.float16, mode="r", shape=(meta["n"], a.m, BOX_FEATS))
            log += side_cars(np.asarray(tm[rows], dtype=np.float32))
        report("training logs", log)
        live = Counter()
        for f in sorted(glob.glob(os.path.join(a.live_dir, f"*_{fam}_*_tokens.npz"))) if a.live_dir else []:
            live += side_cars(np.load(f)["tokens"].astype(np.float32))
        if a.live_dir:
            report("live (B2D_PRIV_DUMP)", live)
        res[fam] = {"log": dict(log), "live": dict(live)}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
