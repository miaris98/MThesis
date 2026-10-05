#!/usr/bin/env python3
"""Stage-1 report for any set of arms on the lead + obstacle routes (TODO A52 protocol, S-126): paired DS against the references with bootstrap CIs, plus the mechanism metrics that
carry far more power than DS (vehicle-collision events per run, share of runs with a collision, events per colliding run, the time-cap share, mean speed), per route group.

References (existing records, free): E15 (arm E e15), J = the four J seeds' mean (J20, J20s1, J20s2, J20s3), WOR. Arms are the labels of `a9_merge.ARMS` (add a pattern there for a new label).

    py scripts/analysis/arm_report.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_20261* \
        --arms M25,M50,M75,MSOUP,MESOUP [--routes lead_obstacle|all] [--json-out report.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import boot_ci, collect  # noqa: E402
from a3_failure_classes import LEAD_FAMILIES, OBSTACLE_FAMILIES, scenario_table, split_run  # noqa: E402

J_SEEDS = ("J20", "J20s1", "J20s2", "J20s3")


def mean(x):
    x = list(x)
    return statistics.fmean(x) if x else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--arms", required=True, help="comma list of arm labels (a9_merge.ARMS)")
    ap.add_argument("--routes", default="lead_obstacle", choices=["lead_obstacle", "all"])
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    arms = a.arms.split(",")

    records, infra = collect(a.roots)
    tab = scenario_table()
    grp = {r: ("lead" if v["family"] in LEAD_FAMILIES else "obstacle" if v["family"] in OBSTACLE_FAMILIES else "other") for r, v in tab.items()}
    pool = lambda r: mean(split_run(x)["ds"] for s in J_SEEDS for x in records[s].get(r, [])[:1])  # noqa: E731

    def runs(arm, r):
        return [split_run(x) for x in records[arm].get(r, [])]

    base_routes = {r for r in tab if a.routes == "all" or grp[r] in ("lead", "obstacle")}
    refs = {"E15": lambda r: mean(x["ds"] for x in runs("E15", r)), "J(4 seeds)": pool, "WoR": lambda r: mean(x["ds"] for x in runs("WOR", r))}
    out = {}
    print(f"routes: {a.routes} ({len(base_routes)} in the set); references E15 / J mean of 4 seeds / WoR\n")
    for arm in arms:
        done = sorted(r for r in base_routes if records[arm].get(r))
        if not done:
            print(f"== {arm}: no records yet\n")
            continue
        print(f"== {arm}: {len(done)} of {len(base_routes)} routes driven, infrastructure failures on {len(infra[arm])} routes (left out)")
        ds = {r: mean(x["ds"] for x in runs(arm, r)) for r in done}
        res = {"n": len(done), "ds": mean(ds.values()), "pairs": {}, "mech": {}, "groups": {}}
        print(f"   mean DS {mean(ds.values()):.1f}")
        for name, ref in refs.items():
            rs = [r for r in done if ref(r) == ref(r)]
            if len(rs) < 5:
                continue
            d = [ds[r] - ref(r) for r in rs]
            lo, hi = boot_ci(d, n=10000)
            res["pairs"][name] = {"n": len(rs), "diff": mean(d), "ci": [lo, hi], "ref": mean(ref(r) for r in rs)}
            print(f"   vs {name:11s} n={len(rs):3d}: {mean(ref(r) for r in rs):5.1f} -> {mean(ds[r] for r in rs):5.1f}  diff {mean(d):+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        # the mechanism metric paired by route, with a bootstrap CI: vehicle-collision events per run (route mean) against each reference
        evr = lambda key, r: mean(x["n_coll_vehicle"] for x in runs(key, r))  # noqa: E731
        jev_r = lambda r: mean(split_run(records[s][r][0])["n_coll_vehicle"] for s in J_SEEDS if records[s].get(r))  # noqa: E731
        for name, ref in (("E15", lambda r: evr("E15", r)), ("J(4 seeds)", jev_r), ("WoR", lambda r: evr("WOR", r))):
            rs = [r for r in done if ref(r) == ref(r)]
            if len(rs) < 5:
                continue
            d = [evr(arm, r) - ref(r) for r in rs]
            lo, hi = boot_ci(d, n=10000)
            res["mech"][name] = {"n": len(rs), "diff": mean(d), "ci": [lo, hi], "ref": mean(ref(r) for r in rs), "arm": mean(evr(arm, r) for r in rs)}
            print(f"   collisions/run vs {name:11s} n={len(rs):3d}: {mean(ref(r) for r in rs):.2f} -> {mean(evr(arm, r) for r in rs):.2f}  diff {mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"   {'group':9s} {'n':>3s} {'DS':>6s} | vehicle-collision events/run, runs with a collision, events per colliding run | time-cap share")
        for g in ("lead", "obstacle", "other", "all"):
            rs = [r for r in done if g == "all" or grp[r] == g]
            if not rs:
                continue
            allruns = [x for r in rs for x in runs(arm, r)]
            ev = [x["n_coll_vehicle"] for x in allruns]
            col = [e for e in ev if e > 0]
            cap = mean(1.0 if x["reason"] == "other_incomplete" else 0.0 for x in allruns)
            line = f"   {g:9s} {len(rs):3d} {mean(ds[r] for r in rs):6.1f} | {mean(ev):.2f}   {100 * len(col) / len(ev):4.0f}%   {mean(col) if col else 0:.2f} | {100 * cap:4.0f}%"
            print(line)
            refs_ds = {nm: mean(v for v in (ref(r) for r in rs) if v == v) for nm, ref in refs.items()}
            print(f"   {'':9s} {'':>3s} references' DS on these routes: " + "  ".join(f"{nm} {v:.1f}" for nm, v in refs_ds.items()))
            res["groups"][g] = {"refs_ds": refs_ds, "n": len(rs), "ds": mean(ds[r] for r in rs), "events_per_run": mean(ev), "share_colliding": len(col) / len(ev), "events_per_colliding": mean(col) if col else 0.0, "time_cap_share": cap}
        # the same mechanism numbers for the references on the same routes
        for name, key in (("E15", "E15"), ("WoR", "WOR")):
            rs = [r for r in done if records[key].get(r)]
            if rs:
                ev = [x["n_coll_vehicle"] for r in rs for x in runs(key, r)]
                col = [e for e in ev if e > 0]
                print(f"   ref {name:4s} {len(rs):3d} {mean(mean(x['ds'] for x in runs(key, r)) for r in rs):6.1f} | {mean(ev):.2f}   {100 * len(col) / len(ev):4.0f}%   {mean(col) if col else 0:.2f}")
        jev = [split_run(records[s][r][0])["n_coll_vehicle"] for r in done for s in J_SEEDS if records[s].get(r)]
        if jev:
            jcol = [e for e in jev if e > 0]
            print(f"   ref J   {len(done):3d} {mean(pool(r) for r in done):6.1f} | {mean(jev):.2f}   {100 * len(jcol) / len(jev):4.0f}%   {mean(jcol) if jcol else 0:.2f}")
        out[arm] = res
        print()
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
        print("wrote", a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
