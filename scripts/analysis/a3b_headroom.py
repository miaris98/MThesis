#!/usr/bin/env python3
"""TODO A3 / A30 follow-up (S-120): where is the headroom left in the 220-route results?

Reads the same driving records as a9_merge.py / a3_failure_classes.py (no box needed) and answers six questions that the
TODO tiers depend on:

1. **Systematic or lottery?** For arm J's four seeds on each route: how many of the four runs have a vehicle collision
   (0..4), and how many of the lost points (collision points, all points) sit on routes where >= 3 of 4 collide.
   Systematic losses need data / representation changes; lottery losses (1 of 4) respond to caution, ensembles, noise.
2. **Seen or unseen scenario types?** E trained on 7 scenario types (S-062), J on those plus the 10 obstacle types
   (A14); the 220 routes use 44. DS of E / J / WoR on routes whose scenarios the arm's data did / did not contain.
3. **Weather and light.** DS by night, heavy rain and fog (route XML), paired against WoR: a camera-only weakness would
   show as a larger drop than WoR's.
4. **Speed.** Mean driving speed (route length x completion / game time) per arm and group, the share of lost points that
   are timeouts / blocked, and whether J's faster runs on the same route are the ones that collide.
5. **Best-of-arms ceiling.** Routes where E, J and WoR all fail; the mean of the per-route best.
6. **Power.** Run-to-run noise from J's four seeds -> the smallest DS difference 220 (or 103) routes can resolve with
   1 / 2 / 4 runs per arm.

    py scripts/analysis/a3b_headroom.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_2026100* \
        --json-out a3b_0410.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import boot_ci, collect  # noqa: E402
from a3_failure_classes import LEAD_FAMILIES, OBSTACLE_FAMILIES, XML, scenario_table, split_run  # noqa: E402

J_SEEDS = ("J20", "J20s1", "J20s2", "J20s3")
#: scenario types in the 6-town PDM-Lite set arm E trained on (S-062); routes without a scenario count as seen
SEEN_E = {"ControlLoss", "DynamicObjectCrossing", "OppositeVehicleRunningRedLight", "SignalizedJunctionLeftTurn",
          "SignalizedJunctionRightTurn", "VehicleTurningRoute", "none"}
SEEN_J = SEEN_E | OBSTACLE_FAMILIES


def weather_table() -> dict:
    out = {}
    for r in ET.parse(XML).getroot().findall("route"):
        w = next(iter(r.iter("weather")), None)
        if w is None:
            continue
        a = {k: float(v) for k, v in w.attrib.items() if k != "route_percentage"}
        out[int(r.get("id"))] = {"night": a["sun_altitude_angle"] < 0, "low_sun": 0 <= a["sun_altitude_angle"] <= 15,
                                 "rain": a["precipitation"] >= 60, "fog": a["fog_density"] >= 50, **a}
    return out


def speed(rec) -> float | None:
    m, rc = rec["meta"], float(rec["scores"]["score_route"])
    if rc < 5 or not m.get("duration_game"):
        return None
    return m["route_length"] * rc / 100.0 / m["duration_game"]


def mean(x):
    x = list(x)
    return statistics.fmean(x) if x else float("nan")


def paired(a: dict, b: dict, rids, label: str, out: dict | None = None):
    rs = [r for r in rids if r in a and r in b]
    if len(rs) < 3:
        return f"{label:34s} n={len(rs):3d}  (too few)"
    d = [a[r] - b[r] for r in rs]
    lo, hi = boot_ci(d, n=10000)
    if out is not None:
        out[label] = {"n": len(rs), "a": mean(a[r] for r in rs), "b": mean(b[r] for r in rs), "diff": mean(d), "ci": [lo, hi]}
    return (f"{label:34s} n={len(rs):3d}  {mean(a[r] for r in rs):5.1f} vs {mean(b[r] for r in rs):5.1f}  "
            f"diff {mean(d):+5.1f} [{lo:+.1f}, {hi:+.1f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    records, _ = collect(a.roots)
    routes = scenario_table()
    wx = weather_table()
    res: dict = {}

    # per-route run lists
    E = {r: [split_run(x) | {"speed": speed(x)} for x in rs] for r, rs in records["E15"].items() if r in routes}
    W = {r: [split_run(x) | {"speed": speed(x)} for x in rs] for r, rs in records["WOR"].items() if r in routes}
    seeds = {s: {r: split_run(rs[0]) | {"speed": speed(rs[0])} for r, rs in records[s].items() if r in routes} for s in J_SEEDS}
    common_j = set.intersection(*(set(v) for v in seeds.values()))
    J = {r: [seeds[s][r] for s in J_SEEDS] for r in common_j}
    allr = sorted(set(E) & set(W) & set(J))
    print(f"routes: E {len(E)} (runs per route {collections.Counter(len(v) for v in E.values())}), "
          f"WoR {len(W)} ({collections.Counter(len(v) for v in W.values())}), J all 4 seeds {len(J)}; all three {len(allr)}")
    dsE = {r: mean(x["ds"] for x in E[r]) for r in E}
    dsW = {r: mean(x["ds"] for x in W[r]) for r in W}
    dsJ = {r: mean(x["ds"] for x in J[r]) for r in J}
    grp = {r: ("lead" if routes[r]["family"] in LEAD_FAMILIES else "obstacle" if routes[r]["family"] in OBSTACLE_FAMILIES else "other")
           for r in routes}

    # 1. systematic vs lottery -------------------------------------------------------------------------------------
    print("\n== 1. J's four seeds per route: how many runs have a vehicle collision? (share of J's lost points) ==")
    bycls = collections.defaultdict(lambda: collections.Counter())
    nr = collections.Counter()
    for r in sorted(J):
        k = sum(1 for x in J[r] if x["n_coll_vehicle"] > 0)
        nr[k] += 1
        for x in J[r]:
            bycls[k]["vehicle_collision_pts"] += x["pen"].get("collisions_vehicle", 0.0)
            bycls[k]["lost_pts"] += 100 - x["ds"]
            bycls[k]["not_finishing_pts"] += x["lost_rc"]
    tot_v = sum(c["vehicle_collision_pts"] for c in bycls.values()) or 1.0
    tot_l = sum(c["lost_pts"] for c in bycls.values()) or 1.0
    print(f"{'runs with collision':>20s} {'routes':>7s} {'veh-collision pts':>18s} {'share':>7s} {'all lost pts':>13s} {'share':>7s}")
    res["j_collision_classes"] = {}
    for k in range(5):
        c = bycls[k]
        print(f"{k:>17d}/4 {nr[k]:7d} {c['vehicle_collision_pts'] / (4 * len(J)):18.2f} {100 * c['vehicle_collision_pts'] / tot_v:6.0f}% "
              f"{c['lost_pts'] / (4 * len(J)):13.2f} {100 * c['lost_pts'] / tot_l:6.0f}%")
        res["j_collision_classes"][k] = {"routes": nr[k], "veh_pts_per_run_avg": c["vehicle_collision_pts"] / (4 * len(J)),
                                        "lost_pts_per_run_avg": c["lost_pts"] / (4 * len(J))}
    print("(points are per route-run averaged over all routes, so the rows add up to J's mean loss)")
    syst = sum(bycls[k]["vehicle_collision_pts"] for k in (3, 4)) / tot_v
    lott = sum(bycls[k]["vehicle_collision_pts"] for k in (1, 2)) / tot_v
    print(f"vehicle-collision points on routes where >= 3 of 4 seeds collide (systematic): {100 * syst:.0f}%; on routes where 1-2 of 4 collide (lottery): {100 * lott:.0f}%")
    res["j_collision_systematic_share"] = syst
    print("by group (share of J's vehicle-collision points that are systematic >=3/4 | mixed 2/4 | rare 1/4):")
    for g in ("lead", "obstacle", "other"):
        t = collections.Counter()
        for r in J:
            if grp[r] != g:
                continue
            k = sum(1 for x in J[r] if x["n_coll_vehicle"] > 0)
            t["sys" if k >= 3 else "mix" if k == 2 else "rare" if k == 1 else "none"] += sum(x["pen"].get("collisions_vehicle", 0.0) for x in J[r])
        s = sum(t.values()) or 1.0
        n = sum(1 for r in J if grp[r] == g)
        print(f"  {g:9s} routes {n:3d}  veh-collision pts per run {s / (4 * n):5.2f}   systematic {100 * t['sys'] / s:3.0f}%  mixed {100 * t['mix'] / s:3.0f}%  rare {100 * t['rare'] / s:3.0f}%")
        res.setdefault("j_collision_by_group", {})[g] = {"routes": n, "pts_per_run": s / (4 * n), "systematic": t["sys"] / s, "mixed": t["mix"] / s, "rare": t["rare"] / s}
    # all-fail
    fail = collections.Counter(sum(1 for x in J[r] if x["ds"] < 50) for r in J)
    lost_by_fail = collections.Counter()
    for r in J:
        k = sum(1 for x in J[r] if x["ds"] < 50)
        lost_by_fail[k] += sum(100 - x["ds"] for x in J[r])
    tl = sum(lost_by_fail.values())
    print("runs with DS < 50 per route (routes | share of J's lost points): " +
          "  ".join(f"{k}/4: {fail[k]} | {100 * lost_by_fail[k] / tl:.0f}%" for k in range(5)))
    res["j_fail_classes"] = {k: {"routes": fail[k], "lost_share": lost_by_fail[k] / tl} for k in range(5)}

    # 2. seen / unseen ---------------------------------------------------------------------------------------------
    print("\n== 2. DS on routes whose scenario types the arm's training data did / did not contain (routes all three drove) ==")
    seenE = lambda r: all(f in SEEN_E for f in routes[r]["all"]) or not routes[r]["all"]  # noqa: E731
    seenJ = lambda r: all(f in SEEN_J for f in routes[r]["all"]) or not routes[r]["all"]  # noqa: E731
    res["seen"] = {}
    for name, fn in (("E: seen types", seenE), ("E: unseen types", lambda r: not seenE(r)),
                     ("J: seen types (+10 obstacle)", seenJ), ("J: unseen types", lambda r: not seenJ(r))):
        rs = [r for r in allr if fn(r)]
        if not rs:
            continue
        row = {"n": len(rs), "E": mean(dsE[r] for r in rs), "J": mean(dsJ[r] for r in rs), "WoR": mean(dsW[r] for r in rs)}
        res["seen"][name] = row
        print(f"{name:30s} n={len(rs):3d}  E {row['E']:5.1f}  J {row['J']:5.1f}  WoR {row['WoR']:5.1f}")
    for fn, nm in ((seenE, "E seen"), (lambda r: not seenE(r), "E unseen")):
        rs = [r for r in allr if fn(r)]
        print("   " + paired(dsE, dsW, rs, f"E - WoR on {nm}", res["seen"]))
    for fn, nm in ((seenJ, "J seen"), (lambda r: not seenJ(r), "J unseen")):
        rs = [r for r in allr if fn(r)]
        print("   " + paired(dsJ, dsW, rs, f"J - WoR on {nm}", res["seen"]))
    fams_unseen = collections.Counter(routes[r]["family"] for r in allr if not seenJ(r))
    print("   unseen-by-J families (routes): " + ", ".join(f"{k} {v}" for k, v in fams_unseen.most_common(40)))

    # 3. weather ---------------------------------------------------------------------------------------------------
    print("\n== 3. weather and light (routes all three drove; paired differences vs WoR) ==")
    res["weather"] = {}
    for nm, fn in (("night", lambda r: wx[r]["night"]), ("day (sun > 15)", lambda r: not wx[r]["night"] and not wx[r]["low_sun"]),
                   ("low sun / dusk", lambda r: wx[r]["low_sun"]), ("heavy rain (>= 60)", lambda r: wx[r]["rain"]),
                   ("dry", lambda r: wx[r]["precipitation"] == 0), ("fog (>= 50)", lambda r: wx[r]["fog"]),
                   ("no fog (< 10)", lambda r: wx[r]["fog_density"] < 10)):
        rs = [r for r in allr if r in wx and fn(r)]
        if len(rs) < 3:
            continue
        print(f"{nm:20s} n={len(rs):3d}  E {mean(dsE[r] for r in rs):5.1f}  J {mean(dsJ[r] for r in rs):5.1f}  WoR {mean(dsW[r] for r in rs):5.1f}")
        res["weather"][nm] = {"n": len(rs), "E": mean(dsE[r] for r in rs), "J": mean(dsJ[r] for r in rs), "WoR": mean(dsW[r] for r in rs)}
    # night vs day for the same arm, and the arm-vs-WoR gap in the two conditions (paired by route, then the difference of the
    # two paired gaps: does the advantage over WoR change with the condition?)
    res["weather_did"] = {}
    for nm, f1, f2 in (("night vs day", lambda r: wx[r]["night"], lambda r: not wx[r]["night"]),
                       ("rain vs dry", lambda r: wx[r]["rain"], lambda r: wx[r]["precipitation"] == 0),
                       ("fog vs no fog", lambda r: wx[r]["fog"], lambda r: wx[r]["fog_density"] < 10)):
        r1 = [r for r in allr if f1(r)]
        r2 = [r for r in allr if f2(r)]
        for arm, d in (("E", dsE), ("J", dsJ), ("WoR", dsW)):
            rnd = random.Random(1)
            diffs = []
            for _ in range(4000):
                diffs.append(mean(d[x] for x in rnd.choices(r1, k=len(r1))) - mean(d[x] for x in rnd.choices(r2, k=len(r2))))
            diffs.sort()
            print(f"   {nm:14s} {arm:3s}: {mean(d[x] for x in r1) - mean(d[x] for x in r2):+6.1f} [{diffs[100]:+.1f}, {diffs[3900]:+.1f}] (n {len(r1)} / {len(r2)}; unpaired, route mix differs)")
        for arm, d in (("E", dsE), ("J", dsJ)):
            g1 = [d[r] - dsW[r] for r in r1]
            g2 = [d[r] - dsW[r] for r in r2]
            rnd = random.Random(2)
            dd = sorted(mean(rnd.choices(g1, k=len(g1))) - mean(rnd.choices(g2, k=len(g2))) for _ in range(4000))
            print(f"   {nm:14s} {arm:3s} - WoR gap: {mean(g1):+5.1f} vs {mean(g2):+5.1f}; change {mean(g1) - mean(g2):+5.1f} [{dd[100]:+.1f}, {dd[3900]:+.1f}]")
            res["weather_did"][f"{nm}/{arm}"] = {"gap_1": mean(g1), "gap_2": mean(g2), "change": mean(g1) - mean(g2), "ci": [dd[100], dd[3900]]}

    # 4. speed -----------------------------------------------------------------------------------------------------
    print("\n== 4. speed (route length x completion / game time, m/s; runs with completion >= 5%) ==")
    res["speed"] = {}
    for g in ("lead", "obstacle", "other", "all"):
        rs = [r for r in allr if g == "all" or grp[r] == g]
        row = {}
        for nm, runs in (("E", E), ("J", J), ("WoR", W)):
            sp = [x["speed"] for r in rs for x in runs[r] if x["speed"] is not None]
            sp_fin = [x["speed"] for r in rs for x in runs[r] if x["speed"] is not None and x["rc"] >= 99.9]
            row[nm] = (mean(sp), mean(sp_fin))
        res["speed"][g] = row
        print(f"{g:9s} n={len(rs):3d}  mean speed E {row['E'][0]:5.2f}  J {row['J'][0]:5.2f}  WoR {row['WoR'][0]:5.2f}   | finished routes only: E {row['E'][1]:5.2f}  J {row['J'][1]:5.2f}  WoR {row['WoR'][1]:5.2f}")
    print("lost points by reason, per route-run (all routes all three drove); 'time cap' = Failed - TickRuntime = more than 4000 ticks (200 s) of game time")
    reasons = ("vehicle_blocked", "route_dev", "route_timeout", "other_incomplete")
    names = {"other_incomplete": "time_cap"}
    for rs_name, runs in (("E", E), ("J", J), ("WoR", W)):
        n = sum(len(runs[r]) for r in allr)
        tot = collections.Counter()
        for r in allr:
            for x in runs[r]:
                if x["reason"]:
                    tot[x["reason"]] += x["lost_rc"]
        res.setdefault("lost_by_reason", {})[rs_name] = {names.get(k, k): tot[k] / n for k in reasons}
        print(f"  {rs_name:3s} " + "  ".join(f"{names.get(k, k)} {tot[k] / n:5.2f}" for k in reasons))
    print("share of runs with a vehicle collision, and mean points lost to vehicle collisions per run, by group (E / J seeds pooled / WoR):")
    res["collision_by_group"] = {}
    for g in ("lead", "obstacle", "other"):
        rs = [r for r in allr if grp[r] == g]
        cells = []
        for nm, runs in (("E", E), ("J", J), ("WoR", W)):
            xs = [x for r in rs for x in runs[r]]
            share = mean(x["n_coll_vehicle"] > 0 for x in xs)
            pts = mean(x["pen"].get("collisions_vehicle", 0.0) for x in xs)
            res["collision_by_group"][f"{g}/{nm}"] = {"runs_with_collision": share, "points": pts}
            cells.append(f"{nm} {100 * share:3.0f}% / {pts:5.1f}")
        print(f"  {g:9s} n={len(rs):3d}  " + "   ".join(cells))
    # within-route: do the faster J runs collide?
    mixed = [r for r in J if 0 < sum(1 for x in J[r] if x["n_coll_vehicle"] > 0) < 4 and all(x["speed"] is not None for x in J[r])]
    dd = []
    for r in mixed:
        c = [x["speed"] for x in J[r] if x["n_coll_vehicle"] > 0]
        n_ = [x["speed"] for x in J[r] if x["n_coll_vehicle"] == 0]
        dd.append(mean(c) - mean(n_))
    lo, hi = boot_ci(dd, n=10000) if len(dd) > 3 else (float("nan"),) * 2
    print(f"routes where J's four seeds disagree on vehicle collision: n={len(mixed)}; mean speed of the colliding runs minus the clean runs on the same route: "
          f"{mean(dd):+.3f} m/s [{lo:+.3f}, {hi:+.3f}]  (route-average speed, so an upper bound on the speed difference at the hazard)")
    res["speed"]["collide_minus_clean_mps"] = {"n": len(mixed), "mean": mean(dd), "ci": [lo, hi]}

    # 5. best-of-arms ceiling --------------------------------------------------------------------------------------
    print("\n== 5. ceilings (routes all three drove; J = 4-seed mean, E single run, WoR run mean: the max over noisy arms is biased upward) ==")
    best3 = {r: max(dsE[r], dsJ[r], dsW[r]) for r in allr}
    bestEJ = {r: max(dsE[r], dsJ[r]) for r in allr}
    print(f"mean DS: E {mean(dsE[r] for r in allr):.1f}  J {mean(dsJ[r] for r in allr):.1f}  WoR {mean(dsW[r] for r in allr):.1f}  "
          f"best of E/J {mean(bestEJ.values()):.1f}  best of E/J/WoR {mean(best3.values()):.1f}")
    allfail = [r for r in allr if max(dsE[r], dsJ[r], dsW[r]) < 50]
    print(f"routes where E, J and WoR all score < 50: {len(allfail)} of {len(allr)} ({100 * len(allfail) / len(allr):.0f}%); they hold "
          f"{sum(100 - best3[r] for r in allfail) / sum(100 - best3[r] for r in allr):.0%} of the points the best-of-three still loses")
    fam_all = collections.Counter(routes[r]["family"] for r in allfail)
    print("   families: " + ", ".join(f"{k} {v}" for k, v in fam_all.most_common(25)))
    res["ceiling"] = {"E": mean(dsE[r] for r in allr), "J": mean(dsJ[r] for r in allr), "WoR": mean(dsW[r] for r in allr),
                      "best_EJ": mean(bestEJ.values()), "best_EJW": mean(best3.values()), "all_fail_routes": allfail,
                      "all_fail_families": dict(fam_all)}
    # lost points of the best-of-three by group
    for g in ("lead", "obstacle", "other"):
        rs = [r for r in allr if grp[r] == g]
        print(f"   {g:9s} n={len(rs):3d}  E {mean(dsE[r] for r in rs):5.1f}  J {mean(dsJ[r] for r in rs):5.1f}  WoR {mean(dsW[r] for r in rs):5.1f}  best-of-3 {mean(best3[r] for r in rs):5.1f}")
    # group-level oracle (honest: three choices, not one per route) and what each group is worth in overall DS if it were solved
    grp_best = {g: max(mean(dsE[r] for r in allr if grp[r] == g), mean(dsJ[r] for r in allr if grp[r] == g)) for g in ("lead", "obstacle", "other")}
    oracle = sum(grp_best[g] * sum(1 for r in allr if grp[r] == g) for g in grp_best) / len(allr)
    print(f"group-level oracle (the better of E / J in each group): {oracle:.1f}; points of overall DS each group still loses for E / J: " +
          "  ".join(f"{g} {sum(100 - dsE[r] for r in allr if grp[r] == g) / len(allr):.1f} / {sum(100 - dsJ[r] for r in allr if grp[r] == g) / len(allr):.1f}" for g in grp_best))
    res["ceiling"]["group_oracle"] = oracle

    # 6. power -----------------------------------------------------------------------------------------------------
    print("\n== 6. power: smallest paired DS difference the route set can resolve (80% power, 5% two-sided; normal approximation) ==")
    var_w = mean(statistics.variance([x["ds"] for x in J[r]]) for r in J)
    d1 = [mean(x["ds"] for x in J[r]) - dsE[r] for r in J if r in dsE]
    var_d = statistics.variance(d1)
    var_int = max(var_d - var_w / 4 - var_w / max(1, mean(len(E[r]) for r in E)), 0.0)
    sd_route = statistics.pstdev([dsJ[r] for r in J])
    print(f"within-route run SD (J seeds) {math.sqrt(var_w):.1f} DS; between-route SD of J's mean {sd_route:.1f}; SD of the paired J-E route difference "
          f"(4 runs vs 1) {math.sqrt(var_d):.1f}; implied route x arm interaction SD {math.sqrt(var_int):.1f}")
    res["power"] = {"sd_within": math.sqrt(var_w), "sd_interaction": math.sqrt(var_int), "mde": {}}
    print(f"{'routes':>8s} " + "".join(f"{'runs/arm=' + str(k):>14s}" for k in (1, 2, 4, 8)))
    for n in (220, 103, 57):
        row = []
        for k in (1, 2, 4, 8):
            se = math.sqrt((var_int + 2 * var_w / k) / n)
            row.append(2.8 * se)
            res["power"]["mde"][f"n{n}_k{k}"] = 2.8 * se
        print(f"{n:8d} " + "".join(f"{v:14.1f}" for v in row))
    # does the group-restricted interaction differ? (lead routes are where the effects are)
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1, default=str)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
