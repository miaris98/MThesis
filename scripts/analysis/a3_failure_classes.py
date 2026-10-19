#!/usr/bin/env python3
"""TODO A3: where does each arm lose Bench2Drive score, and why does each route fail? (first version 2026-10-04; per-route
classes, paired transitions against any reference arm and the A44 clip taxonomy 2026-10-19)

Per arm, over every driving record found under the roots (same collection rules as a9_merge.py), split the lost score the
way the metric defines it, `100 - DS = (100 - RC) + RC * (1 - P)`:

* **not finishing** (100 - RC): route completion lost, with the termination reason taken from the record (blocked /
  deviated / `time_cap`, the 4,000-tick cap that ends a route as `Failed - TickRuntime`), and
* **penalties** (RC * (1 - P)): shared between the infraction types in proportion to each type's log-penalty
  (n_i * ln(1 / p_i), the penalty values of Bench2Drive's statistics_manager.py); outside-route-lanes is continuous and
  is counted on its own.

and the same per scenario family of the route XML (`HighwayCutIn_1` -> `HighwayCutIn`), so the biggest remaining losses
and the gap to the reference arm per scenario are visible.

**Failure class.** A route-run's class is the cause that cost it the most points (`clean` = DS 100, `minor` = under one point
lost with no cause); an arm's class on a route is its most frequent run class. Against the reference arm (`--reference`,
default WOR) every route both drove falls into one (reference class -> arm class) cell, and the cells are ranked by the
points they move: S-090's "where the points are", automated.

**Clips (`--clips`).** The A44 collision clips (`<B2D_OUT>/clips/<label>/*.npz`, written by every queue_A9c.sh job) give who hit
whom: rear-end into the vehicle ahead, hit from behind, same-direction side contact, crossing or oncoming (the other actor's
position and heading in the ego frame at the impact); the ego and other speeds; and whether, and how early, the speed head
asked to stop before the impact (P(stop bin) >= 0.5 or a target speed <= 1 km/h). The other actor's distance at the moment
the posterior moved is not recorded (actors are logged at the trigger only).

    py scripts/analysis/a3_failure_classes.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_2026100* \
        --pool J=J20,J20s1,J20s2,J20s3 --arms E15,J,WOR --pair J-E15 --json-out a3_0410.json
    py scripts/analysis/a3_failure_classes.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_20261019_fleet \
        --arm A57_220='^c19v4_220(?!.*epic)' --arms A57_220,E15 --reference E15 --clips E:/MThesis_EXP/live_20261019_fleet
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import arm_of, collect, parse_arm_specs  # noqa: E402

import xml.etree.ElementTree as ET  # noqa: E402

XML = Path(__file__).resolve().parents[2] / "Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml"
PENALTY = {"collisions_pedestrian": 0.5, "collisions_vehicle": 0.6, "collisions_layout": 0.65, "red_light": 0.7,
           "stop_infraction": 0.8, "scenario_timeouts": 0.7, "yield_emergency_vehicle_infractions": 0.7}
TERMINATION = ("vehicle_blocked", "route_dev", "route_timeout")
#: cause (termination reason or penalty type) -> failure class
CLASS_OF = {"vehicle_blocked": "blocked", "route_dev": "deviated", "route_timeout": "route_timeout", "time_cap": "time_cap",
            "other_incomplete": "incomplete", "collisions_vehicle": "vehicle_collision", "collisions_pedestrian": "pedestrian_collision",
            "collisions_layout": "layout_collision", "red_light": "red_light", "stop_infraction": "stop_sign",
            "scenario_timeouts": "scenario_timeout", "yield_emergency_vehicle_infractions": "yield_emergency",
            "outside_route_lanes_or_other": "outside_lanes"}
#: tie-break between causes that cost the same points: collisions, then what ends a route, then the rest
CLASS_ORDER = ["vehicle_collision", "pedestrian_collision", "layout_collision", "time_cap", "blocked", "deviated", "route_timeout",
               "incomplete", "red_light", "stop_sign", "scenario_timeout", "yield_emergency", "outside_lanes", "minor", "clean"]
#: scenario families (first scenario of the route, trailing version digits stripped) where the other road user's behaviour
#: sets the speed: a lead car that brakes or is slow, a cut-in, a merge, an actor flow
LEAD_FAMILIES = {"HardBreakRoute", "MergerIntoSlowTrafficV", "MergerIntoSlowTraffic", "HighwayExit", "InterurbanActorFlow",
                 "InterurbanAdvancedActorFlow", "StaticCutIn", "ParkingCutIn", "ParkingExit", "CrossingBicycleFlow", "HighwayCutIn",
                 "EnterActorFlow"}
#: families that need a manoeuvre around something blocking the lane (the A14 obstacle set)
OBSTACLE_FAMILIES = {"ConstructionObstacle", "ConstructionObstacleTwoWays", "Accident", "AccidentTwoWays", "ParkedObstacle",
                     "ParkedObstacleTwoWays", "HazardAtSideLane", "HazardAtSideLaneTwoWays", "VehicleOpensDoorTwoWays", "InvadingTurn"}
#: |lateral offset| of the other actor's centre at contact below which it was in our lane (a lane is ~3.5 m, a car ~1.9 m wide)
LANE_M = 2.0
STOP_P, STOP_KMH = 0.5, 1.0


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
    """One run -> {ds, rc, lost_rc, reason, pen: {type: pts}, lost_pen, n_coll_vehicle}"""
    sc = rec["scores"]
    rc, pen, ds = float(sc["score_route"]), float(sc["score_penalty"]), float(sc["score_composed"])
    inf = rec["infractions"]
    lost_rc, lost_pen = 100.0 - rc, rc * (1.0 - pen)
    reason = next((k for k in TERMINATION if inf.get(k)), None)
    if reason is None and lost_rc > 0.5:
        reason = "time_cap" if "TickRuntime" in str(rec.get("status", "")) else "other_incomplete"
    shares = {k: len(inf.get(k, [])) * math.log(1.0 / p) for k, p in PENALTY.items() if inf.get(k)}
    tot = sum(shares.values())
    pen_split = {k: lost_pen * v / tot for k, v in shares.items()} if tot > 0 else ({"outside_route_lanes_or_other": lost_pen} if lost_pen > 0.05 else {})
    return {"ds": ds, "rc": rc, "lost_rc": lost_rc, "reason": reason, "pen": pen_split, "lost_pen": lost_pen,
            "n_coll_vehicle": len(inf.get("collisions_vehicle", []))}


def _rank(c: str) -> int:
    return CLASS_ORDER.index(c) if c in CLASS_ORDER else len(CLASS_ORDER)


def primary_class(x: dict) -> str:
    """The cause that cost this run the most points."""
    if x["ds"] >= 99.99:
        return "clean"
    pts = collections.Counter()
    if x["reason"]:
        pts[CLASS_OF.get(x["reason"], x["reason"])] += x["lost_rc"]
    for k, v in x["pen"].items():
        pts[CLASS_OF.get(k, k)] += v
    if not pts or max(pts.values()) <= 0:
        return "minor"
    return min(pts, key=lambda c: (-round(pts[c], 6), _rank(c)))


def route_class(xs) -> str:
    """An arm's class on a route: its most frequent run class; a tie goes to the class that lost more points."""
    cls = [primary_class(x) for x in xs]
    count = collections.Counter(cls)
    lost = collections.Counter()
    for c, x in zip(cls, xs):
        lost[c] += 100.0 - x["ds"]
    return min(count, key=lambda c: (-count[c], -round(lost[c], 6), _rank(c)))


def class_table(arm_runs: dict) -> dict:
    """{class: runs, share of runs (%), mean DS of those runs, lost = their contribution to the arm's mean lost points per run}"""
    allr = [x for v in arm_runs.values() for x in v]
    by = collections.defaultdict(list)
    for x in allr:
        by[primary_class(x)].append(x)
    n = max(len(allr), 1)
    rows = {c: {"runs": len(v), "share": 100.0 * len(v) / n, "ds": statistics.fmean(x["ds"] for x in v),
                "lost": sum(100.0 - x["ds"] for x in v) / n} for c, v in by.items()}
    return dict(sorted(rows.items(), key=lambda kv: (-kv[1]["lost"], _rank(kv[0]))))


def transitions(rx: dict, rref: dict) -> list:
    """Routes both arms drove, grouped by (reference class -> arm class): n, mean and summed paired DS difference (arm - ref)."""
    cells = collections.defaultdict(list)
    for r in sorted(set(rx) & set(rref)):
        d = statistics.fmean(z["ds"] for z in rx[r]) - statistics.fmean(z["ds"] for z in rref[r])
        cells[(route_class(rref[r]), route_class(rx[r]))].append((r, d))
    rows = [{"ref": a, "arm": b, "n": len(v), "mean_diff": statistics.fmean(d for _, d in v), "sum_diff": sum(d for _, d in v),
             "routes": [r for r, _ in sorted(v, key=lambda t: t[1])]} for (a, b), v in cells.items()]
    rows.sort(key=lambda row: (-abs(row["sum_diff"]), row["ref"], row["arm"]))
    return rows


def boot_ci(diffs, n=20000, seed=0):
    rnd = random.Random(seed)
    means = sorted(statistics.fmean(rnd.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# ---- A44 clips ---------------------------------------------------------------------------------------------------------------
def _asked_stop(f: dict) -> bool:
    probs, tgt = f.get("probs"), f.get("target_kmh")
    return bool((probs and probs[0] is not None and probs[0] >= STOP_P) or (tgt is not None and tgt <= STOP_KMH))


def _yaw_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def collision_geometry(ev: dict) -> str:
    kind = ev.get("kind")
    fwd, right = ev.get("other_rel_fwd"), ev.get("other_rel_right")
    if kind == "layout":
        return "layout"
    if fwd is None or right is None:
        return "unknown"
    if kind == "walker":
        return "walker_ahead" if fwd > 0 and abs(right) < LANE_M else "walker_side"
    ego, other = float(ev.get("ego_speed") or 0.0), float(ev.get("other_speed") or 0.0)
    if abs(right) < LANE_M and fwd > 0 and ego >= other - 0.5:
        return "rear_end"
    if abs(right) < LANE_M and fwd < 0 and other > ego:
        return "hit_from_behind"
    o = next((a for a in ev.get("actors") or [] if a.get("id") == ev.get("other_id")), None)
    if o is None or not ev.get("ego_xy"):
        return "side_unknown_heading"
    gap = _yaw_gap(float(o["xy"][2]), float(ev["ego_xy"][2]))
    return "side_same_direction" if gap < 30 else "crossing" if gap <= 150 else "oncoming"


def clip_row(path: str) -> dict:
    """One clip -> route, kind, geometry, speeds and what the speed head asked for before the event."""
    import numpy as np
    z = np.load(path, allow_pickle=True)
    ev = json.loads(str(z["event"]))
    frames = json.loads(str(z["scalars"]))
    t0 = float(ev.get("t", 0.0))
    pre = [f for f in frames[: int(z["n_before"])] if float(f.get("t", 0.0)) <= t0 + 1e-6]
    m = re.search(r"RouteScenario_(\d+)", os.path.basename(path))
    row = {"file": path, "route": int(m.group(1)) if m else None, "kind": ev.get("kind"), "t": t0, "other_id": ev.get("other_id"),
           "ego_speed": ev.get("ego_speed"), "other_speed": ev.get("other_speed"), "other_type": ev.get("other_type"),
           "geometry": collision_geometry(ev) if ev.get("kind") in ("vehicle", "walker", "layout") else None,
           "stationary_other": None if ev.get("other_speed") is None else float(ev["other_speed"]) < 0.5}
    last = pre[-1] if pre else {}
    row["p_stop_last"] = (last.get("probs") or [None])[0]
    row["target_kmh_last"] = last.get("target_kmh")
    row["asked_stop"] = _asked_stop(last) if pre else None
    lead_from = t0
    for f in reversed(pre):
        if not _asked_stop(f):
            break
        lead_from = float(f["t"])
    row["stop_lead_s"] = round(t0 - lead_from, 2) if row["asked_stop"] else 0.0
    brakes = [float(f.get("brake") or 0.0) for f in pre if t0 - float(f.get("t", 0.0)) <= 1.0]
    row["max_brake_1s"] = max(brakes) if brakes else None
    return row


def load_clips(roots, extra_arms=()) -> dict:
    """{arm: [clip rows]}; the arm comes from the clip folder's name (the job label), 'unassigned' when it matches no arm."""
    out, seen = collections.defaultdict(list), set()
    for root in roots:
        for d in glob.glob(root):
            for f in sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)):
                if not re.search(r"_t\d{4}\.npz$", f) or os.path.basename(f) in seen:
                    continue
                seen.add(os.path.basename(f))
                try:
                    row = clip_row(f)
                except Exception as e:  # a truncated copy must not stop the table
                    print(f"[clips] unreadable {f}: {e!r}", file=sys.stderr)
                    continue
                out[arm_of(os.path.basename(os.path.dirname(f)), extra_arms) or "unassigned"].append(row)
    # A car pushing a parked car fires the collision sensor every tick, and the recorder writes a clip every 20 s of it (S-126):
    # only the first contact per run (the clip tag ends in the hero id) and other actor is the impact
    for rows in out.values():
        first = {}
        for r in sorted(rows, key=lambda r: r["t"]):
            key = (re.sub(r"_[a-z]+_t\d{4}\.npz$", "", os.path.basename(r["file"])), r["kind"], r.get("other_id"))
            r["first_contact"] = key not in first
            first.setdefault(key, r)
    return out


def clip_summary(rows: list) -> dict:
    """Clip counts by kind; the collision statistics use first contacts only (`first_contact`, set by load_clips)."""
    kinds = collections.Counter(r["kind"] for r in rows)
    veh = [r for r in rows if r["kind"] == "vehicle" and r.get("first_contact", True)]
    repeats = sum(1 for r in rows if r["kind"] in ("vehicle", "walker", "layout") and not r.get("first_contact", True))
    geo = collections.Counter(r["geometry"] for r in veh)
    asked = [r for r in veh if r["asked_stop"] is not None]
    leads = [r["stop_lead_s"] for r in asked if r["asked_stop"]]
    speeds = [float(r["ego_speed"]) for r in veh if r["ego_speed"] is not None]
    rear = [r for r in veh if r["geometry"] == "rear_end"]
    return {"clips": len(rows), "kinds": dict(kinds), "vehicle": len(veh), "repeat_contacts": repeats,
            "geometry": {k: 100.0 * v / len(veh) for k, v in geo.most_common()} if veh else {},
            "rear_end_stationary_pct": 100.0 * sum(bool(r["stationary_other"]) for r in rear) / len(rear) if rear else None,
            "ego_speed_median": statistics.median(speeds) if speeds else None,
            "asked_stop_pct": 100.0 * len(leads) / len(asked) if asked else None,
            "stop_lead_median_s": statistics.median(leads) if leads else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--pool", action="append", default=[], metavar="NAME=ARM1,ARM2,..", help="treat several arms' runs as one arm")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=REGEX", help="an arm for labels a9_merge.py does not know (repeatable)")
    ap.add_argument("--arms", default="E15,J,WOR")
    ap.add_argument("--reference", default="WOR", help="the arm every other arm is paired with (default WOR)")
    ap.add_argument("--pair", action="append", default=[], metavar="X-Y", help="extra paired comparison X - Y (repeatable)")
    ap.add_argument("--clips", action="append", default=[], metavar="ROOT", help="folders (globs allowed) searched for A44 clips")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    extra = parse_arm_specs(a.arm)
    records, _ = collect(a.roots, extra)
    for spec in a.pool:
        name, members = spec.split("=", 1)
        for m in members.split(","):
            for rid, runs in records.get(m, {}).items():
                records[name][rid].extend(runs)
    routes = scenario_table()
    arms = [x for x in a.arms.split(",") if x in records]
    missing = [x for x in a.arms.split(",") if x not in records]
    if missing:
        print(f"(no records for {', '.join(missing)})")
    runs = {arm: {rid: [split_run(r) for r in rs] for rid, rs in records[arm].items() if rid in routes} for arm in arms}
    out = {}

    print("== where the score goes: mean lost points per route-run (100 - DS), split by cause ==")
    print(f"{'arm':10s} {'runs':>5s} {'DS':>6s} {'lost':>6s} {'not finishing':>14s} {'penalties':>10s}")
    for arm in arms:
        allr = [x for v in runs[arm].values() for x in v]
        ds = statistics.fmean(x["ds"] for x in allr)
        nf = statistics.fmean(x["lost_rc"] * 1.0 for x in allr)
        pe = statistics.fmean(x["lost_pen"] for x in allr)
        print(f"{arm:10s} {len(allr):5d} {ds:6.1f} {100 - ds:6.1f} {nf:14.1f} {pe:10.1f}")
        by = collections.Counter()
        for x in allr:
            if x["reason"]:
                by[x["reason"]] += x["lost_rc"]
            for k, v in x["pen"].items():
                by["pen:" + k] += v
        n = len(allr)
        out[arm] = {"runs": n, "ds": ds, "lost_by_cause": {k: v / n for k, v in by.items()}}
        print("           " + "  ".join(f"{k} {v / n:.1f}" for k, v in by.most_common(8)))

    print("\n== infraction frequency: share of runs with at least one event ==")
    for arm in arms:
        allr = [r for rs in records[arm].values() for r in rs]
        cnt = collections.Counter(k for r in allr for k, v in r["infractions"].items() if v and k != "min_speed_infractions")
        print(f"{arm:10s} " + "  ".join(f"{k} {100 * c / len(allr):.0f}%" for k, c in cnt.most_common(8)))

    print("\n== failure class per route-run: the cause that cost the run the most points ==")
    print(f"{'arm':10s} {'class':22s} {'runs':>5s} {'share':>6s} {'mean DS':>8s} {'pts of the lost':>16s}")
    out["classes"] = {}
    for arm in arms:
        tab = class_table(runs[arm])
        out["classes"][arm] = tab
        for c, row in tab.items():
            print(f"{arm:10s} {c:22s} {row['runs']:5d} {row['share']:5.1f}% {row['ds']:8.1f} {row['lost']:16.1f}")

    ref = a.reference if a.reference in arms else ("WOR" if "WOR" in arms else arms[-1])
    print(f"\n== per scenario family: mean DS by arm (routes driven by all arms); sorted by the first non-reference arm's loss ==")
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
    print(f"{'family':32s} {'n':>3s} " + " ".join(f"{x:>8s}" for x in arms) + f" {'lost pts (' + focus + ')':>16s}")
    for r in rows[: a.top]:
        print(f"{r['family'][:32]:32s} {r['n']:3d} " + " ".join(f"{r[x]:8.1f}" for x in arms) + f" {(100 - r[focus]) * r['n']:16.0f}")
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

    pairs = [(x, ref) for x in arms if x != ref] + [tuple(p.split("-", 1)) for p in a.pair]
    pairs = [p for i, p in enumerate(pairs) if p not in pairs[:i] and p[0] in runs and p[1] in runs and p[0] != p[1]]

    print(f"\n== where the points are against the reference: routes grouped by (reference class -> arm class), "
          f"ranked by the points they move ==")
    out["transitions"] = {}
    for x, y in pairs:
        tr = transitions(runs[x], runs[y])
        out["transitions"][f"{x}-{y}"] = tr
        tot, n = sum(t["sum_diff"] for t in tr), sum(t["n"] for t in tr)
        print(f"{x} - {y} on {n} routes both drove: mean paired difference {tot / max(n, 1):+.1f}")
        for t in tr[: a.top]:
            if t["ref"] == t["arm"] and abs(t["mean_diff"]) < 5:
                continue
            ids = ",".join(map(str, t["routes"][:8])) + ("..." if len(t["routes"]) > 8 else "")
            print(f"  {y} {t['ref']:>20s} -> {x} {t['arm']:<20s} {t['n']:3d} routes  {t['mean_diff']:+6.1f} each  {t['sum_diff']:+7.0f} pts  ({ids})")

    print("\n== paired DS difference by scenario group (routes both arms drove; 95% bootstrap CI over routes) ==")
    print("lead-vehicle families: " + ", ".join(sorted(LEAD_FAMILIES)))
    print("obstacle families:     " + ", ".join(sorted(OBSTACLE_FAMILIES)))
    groups = {"lead-vehicle": lambda f: f in LEAD_FAMILIES, "obstacle": lambda f: f in OBSTACLE_FAMILIES,
              "all other": lambda f: f not in LEAD_FAMILIES and f not in OBSTACLE_FAMILIES}
    out["groups"] = {}
    for x, y in pairs:
        print(f"{x} - {y}:")
        for gname, fn in groups.items():
            rs = [r for r in runs[x] if r in runs[y] and r in routes and fn(routes[r]["family"])]
            if len(rs) < 3:
                continue
            mx = {r: statistics.fmean(z["ds"] for z in runs[x][r]) for r in rs}
            my = {r: statistics.fmean(z["ds"] for z in runs[y][r]) for r in rs}
            diffs = [mx[r] - my[r] for r in rs]
            lo, hi = boot_ci(diffs)
            print(f"  {gname:13s} n={len(rs):3d}  {x} {statistics.fmean(mx.values()):5.1f}  {y} {statistics.fmean(my.values()):5.1f}  "
                  f"diff {statistics.fmean(diffs):+5.1f} [{lo:+.1f}, {hi:+.1f}]")
            out["groups"][f"{x}-{y}:{gname}"] = {"n": len(rs), "diff": statistics.fmean(diffs), "ci": [lo, hi]}

    out["route_classes"] = {arm: {str(r): route_class(v) for r, v in runs[arm].items()} for arm in arms}
    if a.clips:
        clips = load_clips(a.clips, extra)
        print("\n== A44 clips: who hit whom, and did the speed head ask to stop before the impact ==")
        out["clips"] = {}
        for arm, rows_ in sorted(clips.items()):
            s = clip_summary(rows_)
            out["clips"][arm] = {"summary": s, "rows": rows_}
            print(f"{arm:12s} {s['clips']:4d} clips  " + "  ".join(f"{k} {v}" for k, v in sorted(s["kinds"].items())))
            if s["vehicle"]:
                geo = "  ".join(f"{k} {v:.0f}%" for k, v in s["geometry"].items())
                print(f"{'':12s} vehicle collisions {s['vehicle']} (first contacts; {s['repeat_contacts']} repeat contacts left out): {geo}")
                extra_txt = [f"ego speed at impact median {s['ego_speed_median']:.1f} m/s" if s["ego_speed_median"] is not None else "",
                             f"rear-ends into a stationary vehicle {s['rear_end_stationary_pct']:.0f}%" if s["rear_end_stationary_pct"] is not None else "",
                             f"speed head asked to stop before impact {s['asked_stop_pct']:.0f}%" if s["asked_stop_pct"] is not None else "",
                             f"median lead {s['stop_lead_median_s']:.2f} s" if s["stop_lead_median_s"] is not None else ""]
                print(f"{'':12s} " + "; ".join(t for t in extra_txt if t))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
