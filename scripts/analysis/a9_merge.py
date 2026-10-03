#!/usr/bin/env python3
"""Merge the full-Bench2Drive runs (TODO_ACTIVE A9) and arm J's evals into one table (S-093, S-106, S-107).

Collects every Leaderboard result JSON matching the label patterns below under the given roots (the live_* pulls on
E:), keeps per (arm, route) the driving records - "couldn't be set up", "Simulation crashed" and "Agent crashed" are
harness failures, not driving outcomes - and reports:

* the official Bench2Drive number: mean DS over all 220 routes, a route with no driving record scoring 0;
* each arm against a reference arm (default WOR), paired on the routes both drove, with a bootstrap CI over routes;
* the same split into towns our arms trained on (Town01-05, Town10HD) and the rest, and into A9's 50 obstacle
  routes (Accident / Obstacle / HazardAtSideLane / InvadingTurn / OpensDoor scenarios) and the rest;
* the routes each arm never drove (crash routes), which the official number counts as 0.

A route driven more than once contributes the mean of its runs.

    py scripts/analysis/a9_merge.py E:/MThesis_EXP/live_20260928_* E:/MThesis_EXP/live_20260929_*
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml"
INFRA = ("couldn't be set up", "Simulation crashed", "Agent crashed")
OBSTACLE_TYPES = ("Accident", "Obstacle", "HazardAtSideLane", "InvadingTurn", "OpensDoor")
TRAIN_TOWNS = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"}
#: result-file label -> arm; first match wins
ARMS = [(r"^a9_E15_", "E15"), (r"^a9_WOR_", "WOR"), (r"^j20_", "J20"), (r"^j18_", "J18"),
        # S-112/S-113: arm J seeds 1-3 (j<s>s<epoch>), arm J e18 re-run on the A40 box (j18H), arm K, A28 median decode
        (r"^j18H_", "J18H"), (r"^j([1-9])s(\d\d)_", None), (r"^k(\d\d)_", None), (r"^a28m18_", "J18med")]


def arm_of(name: str):
    for pat, arm in ARMS:
        m = re.match(pat, name)
        if m:
            if arm is not None:
                return arm
            g = m.groups()  # j<seed>s<epoch> -> J20s2, k<epoch> -> K20
            return f"J{g[1]}s{g[0]}" if len(g) == 2 else f"K{g[0]}"
    return None


def boot_ci(diffs, n=20000, seed=0):
    rnd = random.Random(seed)
    means = sorted(statistics.fmean(rnd.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="directories (globs allowed) searched recursively for result JSONs")
    ap.add_argument("--reference", default="WOR")
    ap.add_argument("--json-out", default=None, help="also write the per-route table here")
    ap.add_argument("--pool", action="append", default=[], metavar="NAME=ARM1,ARM2,..",
                    help="add an arm whose per-route DS is the mean over these arms (several seeds of one recipe); "
                         "only routes all of them drove (repeatable)")
    ap.add_argument("--arms", default=None, help="comma list: only report these arms (default: all)")
    a = ap.parse_args()

    routes = {}
    for r in ET.parse(XML).iter("route"):
        types = [s.get("type") or "" for s in r.iter("scenario")]
        routes[int(r.get("id"))] = {"town": r.get("town"),
                                    "obstacle": any(k in t for t in types for k in OBSTACLE_TYPES)}
    drives = collections.defaultdict(lambda: collections.defaultdict(list))  # arm -> route -> [DS]
    infra = collections.defaultdict(set)
    seen_files = set()
    for root in a.roots:
        for d in glob.glob(root):
            for f in glob.glob(os.path.join(d, "**", "*_x.json"), recursive=True):
                name = os.path.basename(f)
                arm = arm_of(name)
                if arm is None or "abandoned" in f:
                    continue
                try:
                    recs = json.load(open(f))["_checkpoint"]["records"]
                except Exception:
                    continue
                for rec in recs:
                    rid = int(rec["route_id"].split("_")[1])
                    key = (name, rid, rec["scores"]["score_composed"], rec["status"])
                    if key in seen_files:  # the same record pulled into several live_* mirrors
                        continue
                    seen_files.add(key)
                    if any(k in rec["status"] for k in INFRA):
                        infra[arm].add(rid)
                    else:
                        drives[arm][rid].append(float(rec["scores"]["score_composed"]))
    ds = {arm: {r: statistics.fmean(v) for r, v in rr.items()} for arm, rr in drives.items()}
    for spec in a.pool:
        name, members = spec.split("=", 1)
        members = [m for m in members.split(",") if m in ds]
        if members:
            common = set.intersection(*(set(ds[m]) for m in members))
            ds[name] = {r: statistics.fmean(ds[m][r] for m in members) for r in common}
            print(f"pool {name} = mean of {','.join(members)} on {len(common)} routes all of them drove")
    if a.arms:
        keep = set(a.arms.split(",")) | {a.reference}
        ds = {k: v for k, v in ds.items() if k in keep}

    print(f"{'arm':5s} {'driven':>6s} {'official DS (220, missing=0)':>29s} {'mean DS on driven':>18s} {'success %':>9s}")
    for arm in sorted(ds):
        d = ds[arm]
        off = sum(d.values()) / 220
        succ = 100 * sum(v >= 99.99 for v in d.values()) / max(len(d), 1)
        print(f"{arm:5s} {len(d):6d} {off:29.1f} {statistics.fmean(d.values()):18.1f} {succ:9.1f}")

    ref = ds.get(a.reference, {})
    subsets = {
        "all": lambda r: True,
        "train towns": lambda r: routes[r]["town"] in TRAIN_TOWNS,
        "other towns": lambda r: routes[r]["town"] not in TRAIN_TOWNS,
        "obstacle": lambda r: routes[r]["obstacle"],
        "non-obstacle": lambda r: not routes[r]["obstacle"],
    }
    for arm in sorted(ds):
        if arm == a.reference:
            continue
        print(f"\n{arm} vs {a.reference}, paired on routes both drove:")
        for sname, keep in subsets.items():
            common = sorted(r for r in ds[arm] if r in ref and r in routes and keep(r))
            if len(common) < 2:
                continue
            diffs = [ds[arm][r] - ref[r] for r in common]
            lo, hi = boot_ci(diffs)
            print(f"  {sname:13s} n={len(common):3d}  {arm} {statistics.fmean(ds[arm][r] for r in common):5.1f}  "
                  f"{a.reference} {statistics.fmean(ref[r] for r in common):5.1f}  diff {statistics.fmean(diffs):+5.1f} "
                  f"[95% CI {lo:+.1f}, {hi:+.1f}]")

    print("\nRoutes never driven (count 0 in the official number):")
    for arm in sorted(ds):
        miss = sorted(set(routes) - set(ds[arm]))
        crash = [r for r in miss if r in infra[arm]]
        print(f"  {arm}: {len(miss)} ({len(crash)} with only harness failures): {','.join(map(str, miss))}")
    if a.json_out:
        json.dump({"ds": {k: {str(r): v for r, v in d.items()} for k, d in ds.items()},
                   "runs": {k: {str(r): v for r, v in d.items()} for k, d in drives.items()},
                   "routes": {str(r): v for r, v in routes.items()}}, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
