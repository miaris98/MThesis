#!/usr/bin/env python3
"""TODO A3: where does each arm lose Bench2Drive score? (first version, 2026-10-04)

Per arm, over every driving record found under the roots (same collection rules as a9_merge.py), split the lost score the
way the metric defines it, `100 - DS = (100 - RC) + RC * (1 - P)`:

* **not finishing** (100 - RC): route completion lost, with the termination reason taken from the record's status
  (blocked / deviated / timed out), and
* **penalties** (RC * (1 - P)): shared between the infraction types in proportion to each type's log-penalty
  (n_i * ln(1 / p_i), the penalty values of Bench2Drive's statistics_manager.py); outside-route-lanes is continuous and
  is counted on its own.

and the same per scenario family of the route XML (`HighwayCutIn_1` -> `HighwayCutIn`), so the biggest remaining losses
and the gap to the reference arm per scenario are visible.

    py scripts/analysis/a3_failure_classes.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_2026100* \
        --pool J=J20,J20s1,J20s2,J20s3 --arms E15,J,WOR --json-out a3_0410.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import collect  # noqa: E402

import xml.etree.ElementTree as ET  # noqa: E402

XML = Path(__file__).resolve().parents[2] / "Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml"
PENALTY = {"collisions_pedestrian": 0.5, "collisions_vehicle": 0.6, "collisions_layout": 0.65, "red_light": 0.7,
           "stop_infraction": 0.8, "scenario_timeouts": 0.7, "yield_emergency_vehicle_infractions": 0.7}
TERMINATION = ("vehicle_blocked", "route_dev", "route_timeout")
#: scenario families (first scenario of the route, trailing version digits stripped) where the other road user's behaviour
#: sets the speed: a lead car that brakes or is slow, a cut-in, a merge, an actor flow
LEAD_FAMILIES = {"HardBreakRoute", "MergerIntoSlowTrafficV", "MergerIntoSlowTraffic", "HighwayExit", "InterurbanActorFlow",
                 "InterurbanAdvancedActorFlow", "StaticCutIn", "ParkingCutIn", "ParkingExit", "CrossingBicycleFlow", "HighwayCutIn",
                 "EnterActorFlow"}
#: families that need a manoeuvre around something blocking the lane (the A14 obstacle set)
OBSTACLE_FAMILIES = {"ConstructionObstacle", "ConstructionObstacleTwoWays", "Accident", "AccidentTwoWays", "ParkedObstacle",
                     "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays", "VehicleOpensDoorTwoWays", "InvadingTurn"}


def family(name: str) -> str:
    return re.sub(r"_?\d+$", "", name or "").rstrip("_") or "none"


def scenario_table():
    out = {}
    for r in ET.parse(XML).iter("route"):
        types = [s.get("type") or "" for s in r.iter("scenario")]
        out[int(r.get("id"))] = {"town": r.get("town"), "family": family(types[0]) if types else "none",
                                 "all": sorted({family(t) for t in types})}
    return out


def split_run(rec) -> dict:
    """One run -> {not_finishing: {reason: pts}, penalty: {type: pts}, ds}"""
    sc = rec["scores"]
    rc, pen, ds = float(sc["score_route"]), float(sc["score_penalty"]), float(sc["score_composed"])
    inf = rec["infractions"]
    lost_rc, lost_pen = 100.0 - rc, rc * (1.0 - pen)
    reason = next((k for k in TERMINATION if inf.get(k)), None)
    if reason is None and lost_rc > 0.5:
        reason = "other_incomplete"
    shares = {k: len(inf.get(k, [])) * math.log(1.0 / p) for k, p in PENALTY.items() if inf.get(k)}
    tot = sum(shares.values())
    pen_split = {k: lost_pen * v / tot for k, v in shares.items()} if tot > 0 else ({"outside_route_lanes_or_other": lost_pen} if lost_pen > 0.05 else {})
    return {"ds": ds, "rc": rc, "lost_rc": lost_rc, "reason": reason, "pen": pen_split, "lost_pen": lost_pen,
            "n_coll_vehicle": len(inf.get("collisions_vehicle", []))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--pool", action="append", default=[], metavar="NAME=ARM1,ARM2,..", help="treat several arms' runs as one arm")
    ap.add_argument("--arms", default="E15,J,WOR")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    records, _ = collect(a.roots)
    for spec in a.pool:
        name, members = spec.split("=", 1)
        for m in members.split(","):
            for rid, runs in records.get(m, {}).items():
                records[name][rid].extend(runs)
    routes = scenario_table()
    arms = [x for x in a.arms.split(",") if x in records]
    runs = {arm: {rid: [split_run(r) for r in rs] for rid, rs in records[arm].items() if rid in routes} for arm in arms}
    out = {}

    print("== where the score goes: mean lost points per route-run (100 - DS), split by cause ==")
    cols = ["ds", "not_finish", "penalties"]
    print(f"{'arm':6s} {'runs':>5s} {'DS':>6s} {'lost':>6s} {'not finishing':>14s} {'penalties':>10s}")
    for arm in arms:
        allr = [x for v in runs[arm].values() for x in v]
        ds = statistics.fmean(x["ds"] for x in allr)
        nf = statistics.fmean(x["lost_rc"] * 1.0 for x in allr)
        pe = statistics.fmean(x["lost_pen"] for x in allr)
        print(f"{arm:6s} {len(allr):5d} {ds:6.1f} {100 - ds:6.1f} {nf:14.1f} {pe:10.1f}")
        by = collections.Counter()
        for x in allr:
            if x["reason"]:
                by[x["reason"]] += x["lost_rc"]
            for k, v in x["pen"].items():
                by["pen:" + k] += v
        n = len(allr)
        out[arm] = {"runs": n, "ds": ds, "lost_by_cause": {k: v / n for k, v in by.items()}}
        print("       " + "  ".join(f"{k} {v / n:.1f}" for k, v in by.most_common(8)))

    print("\n== infraction frequency: share of runs with at least one event ==")
    for arm in arms:
        allr = [r for rs in records[arm].values() for r in rs]
        cnt = collections.Counter(k for r in allr for k, v in r["infractions"].items() if v and k != "min_speed_infractions")
        print(f"{arm:6s} " + "  ".join(f"{k} {100 * c / len(allr):.0f}%" for k, c in cnt.most_common(8)))

    print("\n== per scenario family: mean DS by arm (routes driven by all arms); sorted by the first non-reference arm's loss ==")
    ref = "WOR" if "WOR" in arms else arms[-1]
    fams = collections.defaultdict(list)
    for rid, info in routes.items():
        if all(rid in runs[x] for x in arms):
            fams[info["family"]].append(rid)
    rows = []
    for fam, rids in fams.items():
        row = {"family": fam, "n": len(rids)}
        for arm in arms:
            row[arm] = statistics.fmean(statistics.fmean(x["ds"] for x in runs[arm][r]) for r in rids)
        rows.append(row)
    focus = next((x for x in arms if x not in (ref, "E15")), arms[0])
    rows.sort(key=lambda r: -(100 - r[focus]) * r["n"])
    print(f"{'family':32s} {'n':>3s} " + " ".join(f"{x:>6s}" for x in arms) + f" {'lost pts (' + focus + ')':>16s}")
    for r in rows[: a.top]:
        print(f"{r['family'][:32]:32s} {r['n']:3d} " + " ".join(f"{r[x]:6.1f}" for x in arms) + f" {(100 - r[focus]) * r['n']:16.0f}")
    out["families"] = rows

    print(f"\n== dominant failure per scenario family for {focus} (share of its lost points) ==")
    for r in rows[: a.top]:
        fam_runs = [x for rid in fams[r["family"]] for x in runs[focus][rid]]
        by = collections.Counter()
        for x in fam_runs:
            if x["reason"]:
                by[x["reason"]] += x["lost_rc"]
            for k, v in x["pen"].items():
                by["pen:" + k] += v
        tot = sum(by.values()) or 1.0
        print(f"{r['family'][:30]:30s} " + "  ".join(f"{k} {100 * v / tot:.0f}%" for k, v in by.most_common(3)))
    print("\n== paired DS difference by scenario group (routes both arms drove; 95% bootstrap CI over routes) ==")
    print("lead-vehicle families: " + ", ".join(sorted(LEAD_FAMILIES)))
    print("obstacle families:     " + ", ".join(sorted(OBSTACLE_FAMILIES)))
    groups = {"lead-vehicle": lambda f: f in LEAD_FAMILIES, "obstacle": lambda f: f in OBSTACLE_FAMILIES,
              "all other": lambda f: f not in LEAD_FAMILIES and f not in OBSTACLE_FAMILIES}
    out["groups"] = {}
    for x, y in (("J", "E15"), ("J", "WOR"), ("E15", "WOR")):
        if x not in runs or y not in runs:
            continue
        print(f"{x} - {y}:")
        for gname, fn in groups.items():
            rs = [r for r in runs[x] if r in runs[y] and r in routes and fn(routes[r]["family"])]
            if len(rs) < 3:
                continue
            mx = {r: statistics.fmean(z["ds"] for z in runs[x][r]) for r in rs}
            my = {r: statistics.fmean(z["ds"] for z in runs[y][r]) for r in rs}
            diffs = [mx[r] - my[r] for r in rs]
            rnd = random.Random(0)
            means = sorted(statistics.fmean(rnd.choices(diffs, k=len(diffs))) for _ in range(20000))
            lo, hi = means[500], means[19500]
            print(f"  {gname:13s} n={len(rs):3d}  {x} {statistics.fmean(mx.values()):5.1f}  {y} {statistics.fmean(my.values()):5.1f}  "
                  f"diff {statistics.fmean(diffs):+5.1f} [{lo:+.1f}, {hi:+.1f}]")
            out["groups"][f"{x}-{y}:{gname}"] = {"n": len(rs), "diff": statistics.fmean(diffs), "ci": [lo, hi]}
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
