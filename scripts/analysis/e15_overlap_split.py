#!/usr/bin/env python3
"""Is E15's Bench2Drive-220 score inflated by route overlap with its training data? (2026-11-04)

E15 was trained on the six towns Town01-05 and Town10. The route-overlap audit (route_overlap_audit.py) flagged training
folders that follow an evaluation route (>= 30 m within 10 m, or a same-type scenario trigger within 30 m). This script
splits E15's per-route Bench2Drive driving scores by that overlap and compares them with the reference arm (WoR) on the
same routes, paired:

    overlap   six-town routes that a training folder overlaps (43 routes in the audit)
    clean6    six-town routes with no overlap (1 route)
    other     the Town12/13 routes and the rest (E15 never trained on those towns)

Scores come from the same records as a9_merge.py (harness failures excluded; a route with no driving record scores 0 in the
official mean). The official mean is reported first, to check that the replication reproduces the number in our notes.

    py scripts/analysis/e15_overlap_split.py --leak leak_traj_pass1.json --leak leak_traj.json --out_csv e15_overlap_routes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a9_merge as A  # noqa: E402  (collect, route_table, boot_ci, arm labels)
sys.path.insert(0, str(HERE.parents[1]))
from src.config.paths import exp_root  # noqa: E402  (the experiment archive: MTHESIS_EXP_ROOT)

SIX = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10"}
EXP = exp_root().as_posix()
DEFAULT_ROOTS = [f"{EXP}/{g}" for g in ("live_20260928_*", "live_20260929_*", "bench2drive_live_20260928",
                                         "live_20261010_*", "live_20261011_*", "live_20261017_*")]


def town_key(t: str) -> str:
    return "Town10" if (t or "").startswith("Town10") else (t or "")


def overlap_ids(leak_files) -> set:
    """Evaluation route ids a six-town training folder overlaps, from the audit outputs (both passes)."""
    ids = set()
    for path in leak_files:
        rows = json.load(open(path))["routes"]
        for r in rows:
            if not r.get("leak") or town_key(r.get("town")) not in SIX:
                continue
            e = r.get("eval_ids", [])
            if isinstance(e, str):
                e = eval(e)  # the audit wrote lists as their repr in some passes
            ids.update(int(x) for x in e)
    return ids


def route_scores(records, arm: str) -> dict:
    """route id -> mean composed score over that arm's driving records (routes with no record are absent)."""
    out = {}
    for rid, recs in records.get(arm, {}).items():
        out[rid] = statistics.fmean(r["scores"]["score_composed"] for r in recs)
    return out


def official_mean(scores: dict, all_routes) -> float:
    return sum(scores.get(r, 0.0) for r in all_routes) / len(all_routes)


def group_of(rid, routes, overlap) -> str:
    town = town_key(routes[rid]["town"])
    if rid in overlap:
        return "overlap"
    if town in SIX:
        return "clean6"
    return "other"


def paired(a: dict, b: dict, ids) -> tuple:
    diffs = [a[r] - b[r] for r in ids if r in a and r in b]
    if not diffs:
        return None
    lo, hi = A.boot_ci(diffs)
    return len(diffs), statistics.fmean(diffs), lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leak", action="append", required=True, help="route_overlap_audit JSON (pass 1 and pass 2)")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--arms", default="E15,E15HYB,WOR", help="comma list of arm labels from a9_merge")
    ap.add_argument("--reference", default="WOR")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()

    routes = A.route_table()
    all_routes = sorted(routes)
    overlap = overlap_ids(a.leak)
    records, infra = A.collect(a.roots)
    arms = [x for x in a.arms.split(",") if x]

    print(f"{len(all_routes)} evaluation routes; {len(overlap)} overlapped by six-town training folders "
          f"(six-town routes total: {sum(1 for r in all_routes if town_key(routes[r]['town']) in SIX)})")
    groups = {"overlap": [], "clean6": [], "other": []}
    for rid in all_routes:
        groups[group_of(rid, routes, overlap)].append(rid)
    print("group sizes:", {g: len(v) for g, v in groups.items()})

    scores = {arm: route_scores(records, arm) for arm in arms + [a.reference]}
    print("\nofficial mean over all 220 routes (no record = 0):")
    for arm in arms + [a.reference]:
        print(f"  {arm:8s} {official_mean(scores[arm], all_routes):6.2f}   driven routes: {len(scores[arm])}")

    print("\nmean DS by group (official rule: missing = 0)")
    for arm in arms + [a.reference]:
        row = []
        for g in ("overlap", "clean6", "other"):
            ids = groups[g]
            row.append(f"{g}={sum(scores[arm].get(r, 0.0) for r in ids) / max(1, len(ids)):6.2f}")
        print(f"  {arm:8s} " + "  ".join(row))

    print(f"\npaired difference vs {a.reference} on routes both drove (bootstrap 95% CI over routes)")
    for arm in arms:
        if arm == a.reference:
            continue
        for g in ("overlap", "clean6", "other", "all"):
            ids = groups[g] if g != "all" else all_routes
            res = paired(scores[arm], scores[a.reference], ids)
            if res:
                n, d, lo, hi = res
                print(f"  {arm:8s} {g:8s} n={n:3d}  {d:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]")

    if a.out_csv:
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["route_id", "town", "group", "obstacle"] + [f"{arm}_ds" for arm in arms + [a.reference]])
            for rid in all_routes:
                w.writerow([rid, routes[rid]["town"], group_of(rid, routes, overlap),
                            int(routes[rid]["obstacle"])]
                           + [f"{scores[arm][rid]:.2f}" if rid in scores[arm] else "" for arm in arms + [a.reference]])
        print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
