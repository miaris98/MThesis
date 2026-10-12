#!/usr/bin/env python3
"""TODO A56: what do SimLingo-Data's Dreamer labels and bucket index contain? (S-122, 2026-10-04)

`RenzKa/simlingo` (Hugging Face, Wayve non-commercial licence, 1.17 TB, 3.3M frames, the same 1024x512 / FOV 110 / x = -1.5 m / z = 2.0 m front camera as
PDM_Lite_Carla_LB2) ships, next to the driving data, small per-frame label files:

* `dreamer/.../<frame>.json.gz`: counterfactual alternatives for the frame, one entry per mode (target_speed, stop, faster, faster_factor, slower,
  slower_factor, crash, lane_change) with the alternative's 10 waypoints / 20 route points and a kinematic roll-out verdict: `info.dynamic_crash`,
  `info.dynamic_crash_timesteps`, `allowed`, `safe_to_execute`. These are ready-made counterfactual speed / path safety labels (TODO A37) computed with
  privileged state, for every frame, in a few GB.
* `buckets_paths.pkl`: the CarLLaVA / SimLingo sampling buckets (acceleration, brake, vehicle, vehicle_front, leading_object_*, walker_hazard, red_light,
  stop_sign, changed_route, ...): the hazard-bucket index of TODO A38, 2.9M distinct frames.

This script downloads one small Dreamer archive (118 MB) and the bucket pickle (620 MB) into --cache (skipped when present), prints per-mode label statistics and
the bucket sizes. The pickle is scanned for import / call opcodes first and loaded with an unpickler that refuses every global.

    py scripts/analysis/simlingo_dreamer_probe.py --cache <dir> [--no-buckets]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import pickle
import pickletools
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/RenzKa/simlingo/resolve/main/"
DREAMER = "dreamer_simlingo_lb1_split_routes_training_ControlLoss_chunk_001.tar.gz"
BUCKETS = "buckets_paths.pkl"


def fetch(name: str, cache: Path) -> Path:
    dst = cache / name
    if not dst.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...", flush=True)
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
    return dst


class NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--no-buckets", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache)

    path = fetch(DREAMER, cache)
    stats = collections.defaultdict(collections.Counter)
    frames = 0
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".json.gz")):
                continue
            d = json.loads(gzip.decompress(tf.extractfile(m).read()))
            frames += 1
            for mode, entries in d.items():
                for e in entries:
                    c = stats[mode]
                    c["n"] += 1
                    c["safe_to_execute=False"] += e.get("safe_to_execute") is False
                    c["allowed=False"] += e.get("allowed") is False
                    c["dynamic_crash"] += bool(e.get("info", {}).get("dynamic_crash"))
    print(f"\n{DREAMER}: {frames} frames (LB1-split ControlLoss, Town01-05 / Town10HD)")
    print(f"{'mode':14s} {'entries':>8s} {'safe_to_execute=False':>22s} {'allowed=False':>14s} {'dynamic_crash':>14s}")
    for mode, c in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        n = c["n"]
        print(f"{mode:14s} {n:8d} {100 * c['safe_to_execute=False'] / n:21.1f}% {100 * c['allowed=False'] / n:13.1f}% {100 * c['dynamic_crash'] / n:13.1f}%")

    if not a.no_buckets:
        bp = fetch(BUCKETS, cache)
        risky = collections.Counter()
        with open(bp, "rb") as f:
            for op, _, _ in pickletools.genops(f):
                if op.name in ("GLOBAL", "STACK_GLOBAL", "INST", "OBJ", "REDUCE", "NEWOBJ", "NEWOBJ_EX", "BUILD"):
                    risky[op.name] += 1
        if risky:
            raise SystemExit(f"{bp} contains import / call opcodes {dict(risky)}: not loading it")
        with open(bp, "rb") as f:
            d = NoGlobals(f).load()
        allp = set()
        print(f"\n{BUCKETS}: {len(d)} buckets (frames per bucket)")
        for k in sorted(d, key=lambda k: -len(d[k])):
            print(f"{len(d[k]):9d}  {k}")
            allp.update(d[k])
        print(f"distinct frames over all buckets: {len(allp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
