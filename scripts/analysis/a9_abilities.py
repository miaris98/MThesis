#!/usr/bin/env python3
"""Bench2Drive ability scores for the A9 runs (TODO_ACTIVE A9), offline.

Re-implements `Bench2Drive/tools/ability_benchmark.py` without CARLA: a route is a success when its status is Completed
or Perfect and it has no infraction other than min_speed_infractions; each ability is the success rate over the routes
whose first scenario belongs to it (the official scenario lists, imported from the tool); routes without a driving record
(crash routes) are left out, as in the official tool. Merging, Overtaking, Emergency_Brake and Give_Way are exact.
Traffic_Signs needs, per route, the completion fraction at its first junction waypoint, which only CARLA's map gives;
pass `--junctions FILE` (route id -> fraction, from a CARLA box) to score it, otherwise it is left out of the mean.
A route driven more than once contributes the mean of its runs' success (the official tool reads one record per route).

Also reports, per ability, each arm against the reference paired on the routes both drove (bootstrap CI over routes).

    py scripts/analysis/a9_abilities.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_202610*_* --arms E15,WOR
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import ROOT, boot_ci, collect, route_table  # noqa: E402

TOOL = ROOT / "Carla-utils/carla_garage/Bench2Drive/tools/ability_benchmark.py"


def official_abilities() -> dict:
    """The `Ability` dict of the official tool, read from its source (importing it needs the carla module)."""
    tree = ast.parse(TOOL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "Ability" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"no Ability table in {TOOL}")


def success(rec) -> bool:
    if rec["status"] not in ("Completed", "Perfect"):
        return False
    return not any(len(v) > 0 for k, v in rec["infractions"].items() if k != "min_speed_infractions")


def traffic_sign_ok(rec, junction_frac: float) -> bool:
    inf = rec["infractions"]
    return rec["scores"]["score_route"] / 100.0 > junction_frac and not inf["stop_infraction"] and not inf["red_light"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--arms", default="E15,WOR")
    ap.add_argument("--reference", default="WOR")
    ap.add_argument("--junctions", default=None, help="JSON route id -> completion fraction at the first junction (+8 wps)")
    a = ap.parse_args()

    abilities = official_abilities()
    routes = route_table()
    records, _ = collect(a.roots)
    arms = a.arms.split(",")
    junc = {int(k): v for k, v in json.load(open(a.junctions)).items()} if a.junctions else None

    # arm -> ability -> route -> mean success over that route's driving records
    score = {arm: {ab: {} for ab in abilities} for arm in arms}
    for arm in arms:
        for rid, recs in records.get(arm, {}).items():
            scen = routes[rid]["scenario"]
            for ab, scens in abilities.items():
                if scen not in scens:
                    continue
                if ab == "Traffic_Signs":
                    if junc is None or rid not in junc:
                        continue
                    vals = [traffic_sign_ok(r, junc[rid]) for r in recs]
                else:
                    vals = [success(r) for r in recs]
                score[arm][ab][rid] = statistics.fmean(vals)

    shown = [ab for ab in abilities if ab != "Traffic_Signs" or junc is not None]
    print(f"{'ability':16s}" + "".join(f"{arm:>16s}" for arm in arms))
    for ab in shown:
        print(f"{ab:16s}" + "".join(
            f"{100 * statistics.fmean(score[arm][ab].values()):9.1f} (n={len(score[arm][ab]):3d})" if score[arm][ab] else f"{'-':>16s}"
            for arm in arms))
    means = {arm: statistics.fmean(100 * statistics.fmean(score[arm][ab].values()) for ab in shown if score[arm][ab])
             for arm in arms}
    label = "mean (5)" if junc is not None else "mean (4, no TS)"
    print(f"{label:16s}" + "".join(f"{means[arm]:16.1f}" for arm in arms))

    ref = a.reference
    for arm in arms:
        if arm == ref or ref not in score:
            continue
        print(f"\n{arm} vs {ref}, paired success rate (percentage points) on routes both drove:")
        for ab in shown:
            common = sorted(set(score[arm][ab]) & set(score[ref][ab]))
            if len(common) < 2:
                continue
            d = [100 * (score[arm][ab][r] - score[ref][ab][r]) for r in common]
            lo, hi = boot_ci(d)
            print(f"  {ab:16s} n={len(common):3d}  {statistics.fmean(d):+6.1f} [95% CI {lo:+.1f}, {hi:+.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
