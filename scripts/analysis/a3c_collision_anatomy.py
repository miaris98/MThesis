#!/usr/bin/env python3
"""TODO A1 / A11 from the records alone (S-120): where on the road do the collisions happen?

Every collision infraction of a Bench2Drive record carries the other actor's type and id and the ego position
("Agent collided against object with type=vehicle.mini.cooper_s_2021 and id=3697 at (x=.., y=.., z=..)"). Projecting that
position onto the route polyline of the route XML (the unshifted plan, = what the agent is given) gives

* the **lateral offset** of the car from the route line: ~0 m = it drove straight into something on its line (rear-end,
  hitting an obstacle it did not avoid), >= 1.5 m = it had left the line (a swerve / pass / lane change in progress);
  negative = left of the route (the oncoming lane in right-hand traffic), positive = right;
* the **progress** along the route relative to the scenario's trigger point;
* whether the same actor id is hit again in other runs of the same route (a scenario actor rather than random traffic).

A speed-only fix (A37 / A38 / A39 / A43) can only help the first kind; the second needs the path or the pass decision (A15, A41,
A48, A21).

    py scripts/analysis/a3c_collision_anatomy.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_2026100* \
        --json-out a3c_0410.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import collect  # noqa: E402
from a3_failure_classes import LEAD_FAMILIES, OBSTACLE_FAMILIES, XML, family  # noqa: E402

MSG = re.compile(r"type=(\S+) and id=(\d+) at \(x=(-?[\d.]+), y=(-?[\d.]+), z=(-?[\d.]+)\)")
KINDS = ("collisions_vehicle", "collisions_layout", "collisions_pedestrian")
J_SEEDS = ("J20", "J20s1", "J20s2", "J20s3")


def route_geometry() -> dict:
    out = {}
    for r in ET.parse(XML).getroot().findall("route"):
        pts = [(float(p.get("x")), float(p.get("y"))) for p in r.find("waypoints")]
        cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            cum.append(cum[-1] + math.dist(a, b))
        scen = list(r.find("scenarios"))
        trig = None
        if scen:
            t = scen[0].find("trigger_point")
            if t is not None:
                trig = (float(t.get("x")), float(t.get("y")))
        out[int(r.get("id"))] = {"pts": pts, "cum": cum, "trigger": trig,
                                 "family": family(scen[0].get("type")) if scen else "none"}
    return out


def project(geo: dict, q: tuple) -> tuple[float, float]:
    """(arc length s along the route, lateral offset in metres; positive = right of the travel direction in CARLA's frame)."""
    pts, cum = geo["pts"], geo["cum"]
    best = (1e18, 0.0, 0.0)
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((q[0] - x0) * dx + (q[1] - y0) * dy) / L2))
        px, py = x0 + t * dx, y0 + t * dy
        d = math.hypot(q[0] - px, q[1] - py)
        if d < best[0]:
            L = math.sqrt(L2)
            lat = (-dy * (q[0] - x0) + dx * (q[1] - y0)) / L  # right-hand normal (CARLA: x forward, y right)
            best = (d, cum[i] + t * L, lat)
    return best[1], best[2]


def events(rec):
    for kind in KINDS:
        for msg in rec["infractions"].get(kind, []):
            m = MSG.search(msg)
            if m:
                yield kind, m.group(1), int(m.group(2)), (float(m.group(3)), float(m.group(4)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--off", type=float, default=1.5, help="|lateral offset| in metres above which the car is 'off the line'")
    a = ap.parse_args()

    records, _ = collect(a.roots)
    geo = route_geometry()
    arms = {"E": ["E15"], "J": list(J_SEEDS), "WoR": ["WOR"]}
    grp = lambda f: "lead" if f in LEAD_FAMILIES else "obstacle" if f in OBSTACLE_FAMILIES else "other"  # noqa: E731
    rows = collections.defaultdict(list)  # (arm, group, kind) -> [dict]
    per_route = collections.defaultdict(lambda: collections.defaultdict(list))  # arm -> rid -> [(kind,type,id)]
    nruns = collections.Counter()
    for arm, members in arms.items():
        for m in members:
            for rid, recs in records[m].items():
                if rid not in geo:
                    continue
                g = geo[rid]
                for rec in recs:
                    nruns[(arm, grp(g["family"]))] += 1
                    for kind, typ, aid, q in events(rec):
                        s, lat = project(g, q)
                        st = None
                        if g["trigger"]:
                            st = project(g, g["trigger"])[0]
                        rows[(arm, grp(g["family"]), kind)].append(
                            {"rid": rid, "family": g["family"], "type": typ, "id": aid, "s": s, "lat": lat,
                             "ds_trigger": (s - st) if st is not None else None, "route_len": g["cum"][-1]})
                        per_route[arm][rid].append((kind, typ, aid))

    print(f"collision events by arm and group; 'off line' = |lateral offset from the route| >= {a.off} m "
          f"(runs: " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(nruns.items())) + ")")
    print(f"{'arm':4s} {'group':9s} {'kind':22s} {'events':>6s} {'per run':>8s} {'off line':>9s} {'left':>6s} {'right':>6s} {'median |lat|':>13s} {'median dTrig m':>15s}")
    res = {}
    for g in ("lead", "obstacle", "other"):
        for arm in arms:
            for kind in KINDS:
                ev = rows.get((arm, g, kind), [])
                if not ev:
                    continue
                lat = [e["lat"] for e in ev]
                off = [abs(x) >= a.off for x in lat]
                dt = [e["ds_trigger"] for e in ev if e["ds_trigger"] is not None]
                row = {"events": len(ev), "per_run": len(ev) / nruns[(arm, g)], "off_line": sum(off) / len(ev),
                       "left": sum(x <= -a.off for x in lat) / len(ev), "right": sum(x >= a.off for x in lat) / len(ev),
                       "median_abs_lat": statistics.median(abs(x) for x in lat), "median_d_trigger": statistics.median(dt) if dt else None}
                res[f"{arm}/{g}/{kind}"] = row
                print(f"{arm:4s} {g:9s} {kind:22s} {len(ev):6d} {row['per_run']:8.2f} {100 * row['off_line']:8.0f}% {100 * row['left']:5.0f}% "
                      f"{100 * row['right']:5.0f}% {row['median_abs_lat']:13.2f} {row['median_d_trigger'] if row['median_d_trigger'] is not None else float('nan'):15.1f}")
        print()

    print("== vehicle collisions per family: events per run | off-line share | left share (E / J / WoR) ==")
    fams = sorted({geo[r]["family"] for r in geo})
    for fam in fams:
        if grp(fam) == "other" and not any(rows.get((arm, "other", "collisions_vehicle")) for arm in arms):
            continue
        cells = []
        tot = 0
        for arm in arms:
            ev = [e for e in rows.get((arm, grp(fam), "collisions_vehicle"), []) if e["family"] == fam]
            nr = sum(len(records[m].get(rid, [])) for m in arms[arm] for rid in geo if geo[rid]["family"] == fam)
            tot += len(ev)
            if ev and nr:
                cells.append(f"{arm} {len(ev) / nr:4.2f} | {100 * sum(abs(e['lat']) >= a.off for e in ev) / len(ev):3.0f}% | {100 * sum(e['lat'] <= -a.off for e in ev) / len(ev):3.0f}%")
            else:
                cells.append(f"{arm}  -")
        if tot >= 3:
            print(f"{fam[:34]:34s} {grp(fam):8s} " + "   ".join(cells))

    # multiple events per run, and the same actor hit in several runs of one route
    print("\n== repeated events (vehicle collisions) ==")
    for arm, members in arms.items():
        counts = collections.Counter()
        uniq = collections.Counter()
        for m in members:
            for rid, recs in records[m].items():
                for rec in recs:
                    ev = [(typ, aid) for kind, typ, aid, _ in events(rec) if kind == "collisions_vehicle"]
                    if ev:
                        counts[min(len(ev), 4)] += 1
                        uniq[min(len(set(ev)), 4)] += 1
        n = sum(counts.values())
        print(f"{arm:4s} runs with a vehicle collision {n}: events per run " + ", ".join(f"{k}{'+' if k == 4 else ''}: {100 * v / n:.0f}%" for k, v in sorted(counts.items())) +
              " | distinct actors " + ", ".join(f"{k}{'+' if k == 4 else ''}: {100 * v / n:.0f}%" for k, v in sorted(uniq.items())))
    # points lost to vehicle events after the first one in a run (upper bound for a "stop after the first contact" behaviour:
    # the penalty is 0.6 per registered vehicle collision; routes completion is kept as it was)
    print("\n== points lost to the 2nd, 3rd ... vehicle collision of a run (DS / 0.6^(n-1) - DS), per route-run ==")
    res["excess"] = {}
    for arm, members in arms.items():
        tot = collections.Counter()
        nr = collections.Counter()
        for m in members:
            for rid, recs in records[m].items():
                if rid not in geo:
                    continue
                g = grp(geo[rid]["family"])
                for rec in recs:
                    nr[g] += 1
                    nv = len(rec["infractions"].get("collisions_vehicle", []))
                    ds = float(rec["scores"]["score_composed"])
                    if nv >= 2:
                        tot[g] += ds * (1 / 0.6 ** (nv - 1) - 1)
                    tot["first_event_loss_" + g] += 0.0
        n_all = sum(nr.values())
        res["excess"][arm] = {g: tot[g] / nr[g] for g in nr} | {"all": sum(tot[g] for g in nr) / n_all}
        print(f"{arm:4s} " + "  ".join(f"{g} {tot[g] / nr[g]:5.2f}" for g in ("lead", "obstacle", "other")) + f"   all {sum(tot[g] for g in nr) / n_all:5.2f}")
    gaps = collections.defaultdict(list)
    for arm, members in arms.items():
        for m in members:
            for rid, recs in records[m].items():
                if rid not in geo:
                    continue
                for rec in recs:
                    ss = sorted(project(geo[rid], q)[0] for kind, typ, aid, q in events(rec) if kind == "collisions_vehicle")
                    gaps[arm] += [b - a_ for a_, b in zip(ss, ss[1:])]
    for arm in arms:
        if gaps[arm]:
            q = sorted(gaps[arm])
            print(f"{arm:4s} distance along the route between consecutive vehicle collisions of a run: n={len(q)} median {q[len(q) // 2]:.1f} m, "
                  f"quartiles {q[len(q) // 4]:.1f} / {q[3 * len(q) // 4]:.1f} m")
    same = collections.Counter()
    for arm in ("J",):
        by_route = collections.defaultdict(list)
        for m in arms[arm]:
            for rid, recs in records[m].items():
                for rec in recs:
                    ids = {(typ, aid) for kind, typ, aid, _ in events(rec) if kind == "collisions_vehicle"}
                    if ids:
                        by_route[rid].append(ids)
        for rid, lst in by_route.items():
            if len(lst) < 2:
                continue
            c = collections.Counter(x for ids in lst for x in ids)
            same["routes with >= 2 colliding J runs"] += 1
            if c.most_common(1)[0][1] >= 2:
                same["... of which the same actor id is hit in >= 2 runs"] += 1
    print(dict(same))

    # most frequently hit vehicle models on lead / obstacle routes (J)
    print("\n== vehicle blueprints hit (J, all collisions) ==")
    cb = collections.Counter(e["type"] for (arm, g, k), ev in rows.items() if arm == "J" and k == "collisions_vehicle" for e in ev)
    print(", ".join(f"{k} {v}" for k, v in cb.most_common(12)))
    cl = collections.Counter(e["type"] for (arm, g, k), ev in rows.items() if arm == "J" and k == "collisions_layout" for e in ev)
    print("layout objects hit (J): " + ", ".join(f"{k} {v}" for k, v in cl.most_common(12)))
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
