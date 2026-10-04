#!/usr/bin/env python3
"""TODO A52 follow-up (S-125): which routes are worth running when screening an arm, and what does a cheaper screen resolve?

Reads the same 220-route records as a3b_headroom.py (E, WoR, arm J's four training seeds, with repeat runs of the same checkpoint on ~50 routes) and answers,
with no box:

1. **Where is the information?** Per route group (lead / obstacle / other) and per *clean* route (every run of every arm scores >= 99.5): routes, J's lost DS,
   share of the run-to-run noise.
2. **What kind of noise is it?** Repeat runs of one checkpoint give the evaluation-only variance; the variance of the per-seed means beyond that is the
   training-seed (checkpoint) variance. Extra eval runs of one checkpoint only average the first; only more *trained* seeds average the second.
3. **What does a screen resolve?** A new arm with T trained seeds x k eval runs on a route subset, against J's existing four seeds x two runs (free):
   Var per route = interaction + (ckpt + eval / k) / T + (ckpt / 4 + eval / 8); the smallest gain detectable at 80% power (5% two-sided), in the subset's
   own DS units and as the overall-DS equivalent *if the whole effect sits on that subset*, with its route-runs and box-hours.

    py scripts/analysis/a3d_screen_design.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_2026100* \
        --runs-per-hour 90 --json-out a3d_0410.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import collect  # noqa: E402
from a3_failure_classes import LEAD_FAMILIES, OBSTACLE_FAMILIES, scenario_table, split_run  # noqa: E402

J_SEEDS = ("J20", "J20s1", "J20s2", "J20s3")
CLEAN_DS = 99.5
Z = 2.8  # 80% power, 5% two-sided


def mean(x):
    x = list(x)
    return statistics.fmean(x) if x else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--runs-per-hour", type=float, default=90.0, help="route-runs per hour per 12-lane box (measured 65-128, box_rates.py)")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    records, _ = collect(a.roots)
    routes = scenario_table()
    ds = lambda x: split_run(x)["ds"]  # noqa: E731
    E = {r: [ds(x) for x in rs] for r, rs in records["E15"].items() if r in routes}
    W = {r: [ds(x) for x in rs] for r, rs in records["WOR"].items() if r in routes}
    J1 = {s: {r: ds(rs[0]) for r, rs in records[s].items() if r in routes} for s in J_SEEDS}
    common = sorted(set(E) & set(W) & set.intersection(*(set(v) for v in J1.values())))
    N = len(common)
    grp = {r: ("lead" if routes[r]["family"] in LEAD_FAMILIES else "obstacle" if routes[r]["family"] in OBSTACLE_FAMILIES else "other") for r in common}
    jruns = {r: [J1[s][r] for s in J_SEEDS] for r in common}
    clean = {r: min(jruns[r] + E[r] + W[r]) >= CLEAN_DS for r in common}
    res: dict = {"routes": N}
    print(f"routes with E, WoR and four J seeds: {N}")

    # 1. where is the information ---------------------------------------------------------------------------------------
    lost = {r: 100 - mean(jruns[r]) for r in common}
    var_w = {r: statistics.variance(jruns[r]) for r in common}
    tl, tv = sum(lost.values()), sum(var_w.values())
    sets = {"all": lambda r: True, "lead": lambda r: grp[r] == "lead", "obstacle": lambda r: grp[r] == "obstacle",
            "lead + obstacle": lambda r: grp[r] in ("lead", "obstacle"), "other": lambda r: grp[r] == "other", "clean (all arms 100)": lambda r: clean[r]}
    members = {k: [r for r in common if f(r)] for k, f in sets.items()}
    print("\n== 1. where the lost points and the run-to-run noise sit (J, four seeds) ==")
    print(f"{'set':22s} {'routes':>6s} {'J lost DS/route':>16s} {'share of J lost':>16s} {'share of run noise':>19s}")
    res["sets"] = {}
    for k, rs in members.items():
        print(f"{k:22s} {len(rs):6d} {mean(lost[r] for r in rs):16.1f} {100 * sum(lost[r] for r in rs) / tl:15.0f}% {100 * sum(var_w[r] for r in rs) / tv:18.0f}%")
        res["sets"][k] = {"n": len(rs), "j_lost_share": sum(lost[r] for r in rs) / tl, "noise_share": sum(var_w[r] for r in rs) / tv}
    print("clean routes (no arm ever lost a point) are only a tenth of the set, so dropping them saves little; the saving has to come from the group.")

    # 2. the two kinds of noise -----------------------------------------------------------------------------------------
    rep = sorted(r for r in common if all(len(records[s][r]) >= 2 for s in J_SEEDS))
    ve_j = mean(mean(statistics.variance([ds(z) for z in records[s][r][:2]]) for r in rep) for s in J_SEEDS)
    vm = mean(statistics.variance([mean(ds(z) for z in records[s][r][:2]) for s in J_SEEDS]) for r in rep)
    vc = max(vm - ve_j / 2, 0.0)
    e3 = [r for r in common if len(E[r]) >= 3]
    w3 = [r for r in common if len(W[r]) >= 3]
    ve_e = mean(statistics.variance(E[r][:3]) for r in e3)
    ve_w = mean(statistics.variance(W[r][:3]) for r in w3)
    print(f"\n== 2. noise split on the {len(rep)} routes where all four J seeds have a repeat run (obstacle-heavy) ==")
    print(f"evaluation-only (same checkpoint, repeat run): J {ve_j:.0f} (SD {math.sqrt(ve_j):.1f} DS), E {ve_e:.0f} (SD {math.sqrt(ve_e):.1f}, {len(e3)} routes x 3 runs), "
          f"WoR {ve_w:.0f} (SD {math.sqrt(ve_w):.1f}, {len(w3)} routes x 3 runs)")
    print(f"training-seed (checkpoint) variance of J: {vc:.0f} (SD {math.sqrt(vc):.1f}); single-run total {vc + ve_j:.0f}; training seed = {100 * vc / (vc + ve_j):.0f}% of it")
    print("-> repeat runs of one checkpoint remove the larger part; the rest needs more trained seeds (cheap once training is cached).")
    res["noise"] = {"routes": len(rep), "eval_var_J": ve_j, "eval_var_E": ve_e, "eval_var_WoR": ve_w, "ckpt_var_J": vc}
    tot_all = mean(var_w[r] for r in common)  # all routes, single run per seed: ckpt + eval
    ve_all, vc_all = tot_all * ve_j / (ve_j + vc), tot_all * vc / (ve_j + vc)

    # 3. screen designs -------------------------------------------------------------------------------------------------
    d1 = {r: mean(jruns[r]) - mean(E[r]) for r in common}
    print("\n== 3. new arm vs J (4 seeds x 2 runs): smallest gain resolved at 80% power ==")
    print("cells: overall-DS equivalent if the effect sits on the subset (the subset's own DS units) | route-runs | box-hours at "
          f"{a.runs_per_hour:.0f} runs/h")
    res["design"] = {}
    designs = ((1, 1), (1, 2), (2, 1), (3, 1))
    for vi_sd in (0.0, 10.0, math.sqrt(max(statistics.variance([d1[r] for r in common]) - tot_all / 4 - tot_all / 3, 0.0))):
        vi = vi_sd ** 2
        print(f"\n-- arm x route interaction SD {vi_sd:.0f}{' (E vs J: a very different policy)' if vi_sd > 12 else ''} --")
        print(f"{'subset':22s} {'n':>4s} " + "".join(f"{'T=%d k=%d' % d:>26s}" for d in designs))
        for name in ("all", "lead + obstacle", "obstacle"):
            rs = members[name]
            n = len(rs)
            if name == "obstacle":
                ve, vc_ = ve_j, vc  # measured on this very kind of route
            else:
                ve, vc_ = ve_all, vc_all
            cells = []
            for T, k in designs:
                var = vi + (vc_ + ve / k) / T + (vc_ / 4 + ve / 8)
                se = math.sqrt(var / n)
                runs = n * T * k
                cells.append(f"{Z * se * n / N:4.1f} ({Z * se:4.1f}) | {runs:4d} | {runs / a.runs_per_hour:4.1f}h")
                res["design"][f"si{vi_sd:.0f}/{name}/T{T}k{k}"] = {"mde_overall": Z * se * n / N, "mde_subset": Z * se, "runs": runs}
            print(f"{name:22s} {n:4d} " + "".join(f"{c:>26s}" for c in cells))
    print("\nreading: a screen on the lead + obstacle routes costs half the route-runs of a full run and resolves a smaller overall gain, but only for arms whose effect "
          "sits on those routes; a finalist then gets 3 trained seeds on all routes (and the clean routes are the regression check).")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1, default=str)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
