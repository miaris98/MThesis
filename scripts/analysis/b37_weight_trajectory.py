#!/usr/bin/env python3
"""TODO B37 step 0: which modules of the EZ-V2 port actually move during training? (offline, CPU, saved checkpoints only)

For one run's checkpoint sequence (e.g. the 10k-100k states of a 100k run), per top-level module of the model:
weight norm, and the relative change between consecutive checkpoints ||W_t - W_prev|| / ||W_prev||. A module whose relative
change is ~0 while its weight norm decays is being driven by weight decay alone (the dead GTrXL mixer of S-116); a module
that keeps changing by a few percent per 10k steps is learning.

    py scripts/analysis/b37_weight_trajectory.py CKPT [CKPT ...] --out traj.json      (checkpoints in training order)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch


def module_of(name: str) -> str:
    return name.split(".")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prev, rows = None, []
    for path in a.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = {k: v.float() for k, v in ck["model"].items() if v.dtype.is_floating_point and v.numel() > 1}
        upd = ck.get("updates", re.findall(r"upd(\d+)", path))
        norm, delta, dprev = defaultdict(float), defaultdict(float), defaultdict(float)
        for k, v in sd.items():
            m = module_of(k)
            norm[m] += float(v.pow(2).sum())
            if prev is not None and k in prev:
                delta[m] += float((v - prev[k]).pow(2).sum())
                dprev[m] += float(prev[k].pow(2).sum())
        row = {"updates": int(upd[0] if isinstance(upd, list) else upd), "norm": {m: n ** 0.5 for m, n in norm.items()},
               "rel_change": {m: (delta[m] / dprev[m]) ** 0.5 for m in delta if dprev[m] > 0}}
        rows.append(row)
        prev = sd
    mods = sorted(rows[0]["norm"], key=lambda m: -rows[0]["norm"][m])
    print(f"{'module':12s}" + "".join(f"{r['updates']:>9d}" for r in rows) + "   (weight norm)")
    for m in mods:
        print(f"{m[:12]:12s}" + "".join(f"{r['norm'].get(m, 0):9.2f}" for r in rows))
    print(f"\n{'module':12s}" + "".join(f"{r['updates']:>9d}" for r in rows[1:]) + "   (relative change vs the previous checkpoint)")
    for m in mods:
        print(f"{m[:12]:12s}" + "".join(f"{r['rel_change'].get(m, 0):9.3f}" for r in rows[1:]))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
