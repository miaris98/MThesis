#!/usr/bin/env python3
"""Attribute Leaderboard 2.0 infractions to the scenarios that produced them.

WHY THIS EXISTS
---------------
An aggregate LB2 driving score is close to uninformative on its own - the published state of
the art is 6.87 (CarLLaVA, 2024 challenge winner) against 66 for the same group's model on
Leaderboard 1.0, so "near zero" describes almost every learned policy ever submitted. What
distinguishes one near-zero from another is *where* the score is lost.

The obvious move - drop the routes our architecture cannot do - is wrong, and measurably so:
ParkingExit opens 52/112 routes and YieldToEmergencyVehicle appears in 33, so excluding both
leaves 33 of 112, and excluding the cut-in family too leaves essentially nothing. Those
failures are consequences of this system's own design (one forward camera at 100 deg FOV,
pose-delta waypoint targets). They are results, not measurement errors, and removing them
would not produce a more honest number, only a larger one.

So: report every route, and break the infractions down by scenario instead. That is the form
the comparable published analysis takes (Atasever et al., arXiv:2509.22754, Table 3), and it
supports the claim worth making - not "our score is low" but "here is exactly where it goes,
it matches the documented failure pattern for learned policies, and here is the mechanism".

ATTRIBUTION IS APPROXIMATE, AND SAYS SO
---------------------------------------
The results file records infractions with world coordinates; the route XML records each
scenario's trigger point in the same frame. This assigns each infraction to the nearest
trigger point within --radius metres. On a route carrying 50-90 scenarios over ~8 km the
triggers are ~100 m apart on average, so at the default 50 m an infraction is attributed to a
scenario only when it is clearly closest to it; anything further is reported as 'unattributed'
rather than forced onto a scenario. Do not read the per-scenario counts as ground truth about
causation - read them as "infractions that happened where this scenario was staged".

usage:
    scenario_breakdown.py --routes <routes.xml> <results.json> [--radius 50] [--top 20]
"""
import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

COORD_RE = re.compile(r"x=(-?[\d.]+),\s*y=(-?[\d.]+)")

# Infraction keys that carry a location in their message and so can be attributed.
LOCATED = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle",
           "red_light", "stop_infraction", "vehicle_blocked", "route_dev")


def load_routes(path):
    """route_id (as in 'RouteScenario_<id>') -> {'start': (x, y), 'scenarios': [(type, x, y)]}"""
    out = {}
    for r in ET.parse(path).iter("route"):
        wps = [(float(w.attrib["x"]), float(w.attrib["y"])) for w in r.find("waypoints")]
        scen = []
        sc = r.find("scenarios")
        if sc is not None:
            for s in sc:
                tp = s.find("trigger_point")
                if tp is None:
                    continue
                scen.append((s.attrib["type"], float(tp.attrib["x"]), float(tp.attrib["y"])))
        out[r.attrib["id"]] = {"start": wps[0] if wps else None, "scenarios": scen}
    return out


def route_key(route_id):
    """'RouteScenario_12_rep0' -> '12'"""
    core = str(route_id).split("_rep")[0]
    return core.split("_", 1)[1] if "_" in core else core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--routes", required=True, help="the routes XML the run used")
    ap.add_argument("--radius", type=float, default=50.0,
                    help="max distance (m) from a scenario trigger for an infraction to be "
                         "attributed to it (default 50)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    try:
        data = json.load(open(args.results))
        records = data["_checkpoint"]["records"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        sys.exit(f"cannot read {args.results}: {exc}")

    routes = load_routes(args.routes)

    encounters = Counter()          # scenario type -> times staged on an evaluated route
    attributed = defaultdict(Counter)   # scenario type -> infraction kind -> count
    unattributed = Counter()
    park_start, no_park_start = [], []
    blocked_at = Counter()

    for rec in records:
        rid = route_key(rec.get("route_id"))
        info = routes.get(rid)
        if info is None:
            print(f"warning: route {rec.get('route_id')} not found in {args.routes}",
                  file=sys.stderr)
            continue

        for ty, _, _ in info["scenarios"]:
            encounters[ty] += 1

        # Does this route open with a ParkingExit? (it is always scenario #1 when present)
        sx, sy = info["start"]
        starts_parking = any(
            ty == "ParkingExit" and math.hypot(x - sx, y - sy) < 5.0
            for ty, x, y in info["scenarios"])
        completion = rec.get("scores", {}).get("score_route", 0.0)
        (park_start if starts_parking else no_park_start).append(completion)

        for kind, msgs in (rec.get("infractions") or {}).items():
            if kind not in LOCATED:
                continue
            for msg in msgs:
                m = COORD_RE.search(msg)
                if not m:
                    unattributed[kind] += 1
                    continue
                ix, iy = float(m.group(1)), float(m.group(2))
                best, bestd = None, float("inf")
                for ty, x, y in info["scenarios"]:
                    d = math.hypot(ix - x, iy - y)
                    if d < bestd:
                        best, bestd = ty, d
                if best is not None and bestd <= args.radius:
                    attributed[best][kind] += 1
                    if kind == "vehicle_blocked":
                        blocked_at[best] += 1
                else:
                    unattributed[kind] += 1

    n = len(records)
    print(f"routes evaluated: {n}   attribution radius: {args.radius:.0f} m\n")

    print("=" * 78)
    print("INFRACTIONS ATTRIBUTED TO THE NEAREST STAGED SCENARIO")
    print("=" * 78)
    print(f"{'scenario':34s} {'staged':>7s} {'infr':>6s} {'per':>7s}  breakdown")
    rows = []
    for ty, kinds in attributed.items():
        total = sum(kinds.values())
        rows.append((total / max(encounters[ty], 1), total, ty, kinds))
    rows.sort(reverse=True)
    for per, total, ty, kinds in rows[:args.top]:
        brk = ", ".join(f"{k.replace('collisions_', 'col_')}={v}"
                        for k, v in kinds.most_common())
        print(f"{ty:34s} {encounters[ty]:7d} {total:6d} {per:7.2f}  {brk}")
    if unattributed:
        print(f"\nunattributed (no scenario within {args.radius:.0f} m): "
              f"{sum(unattributed.values())} -> {dict(unattributed)}")
        print("  ^ these happened in ordinary driving between staged scenarios, which is "
              "itself a result worth reporting")

    print()
    print("=" * 78)
    print("ROUTE TERMINATION (where the agent got blocked)")
    print("=" * 78)
    if blocked_at:
        for ty, c in blocked_at.most_common():
            print(f"  blocked near {ty:34s} {c:3d} route(s)")
    else:
        print("  no vehicle_blocked infractions recorded")

    print()
    print("=" * 78)
    print("PARKING-EXIT EFFECT (route completion, not excluded - measured)")
    print("=" * 78)

    def stat(v):
        return f"n={len(v):3d}  mean_route_completion={sum(v)/len(v):6.2f}%" if v else "n=  0"
    print(f"  routes STARTING with ParkingExit : {stat(park_start)}")
    print(f"  routes without it                : {stat(no_park_start)}")
    if park_start and no_park_start:
        d = sum(no_park_start) / len(no_park_start) - sum(park_start) / len(park_start)
        print(f"  difference                       : {d:+6.2f} percentage points of route "
              f"completion attributable to opening in a parking bay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
