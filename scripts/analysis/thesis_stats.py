#!/usr/bin/env python3
"""Thesis statistics for the Bench2Drive claims (supervisor report M4; TODO A30, A52): which interval answers which
question, the training-seed variance, the family of screened ideas, and the smallest effect a design can resolve.

Three estimands for "X beats Y", paired on the routes both drove:
* **checkpoint, these routes**: this X checkpoint beats this Y checkpoint on these routes, in expectation over the
  simulator's run-to-run noise (sigma_run per route-run);
* **recipe, these routes**: X's training recipe (any seed) beats Y here. It adds X's seed noise per route (sigma_sr)
  and as one shift shared by every route (sigma_g); a single seed carries all of sigma_g, which no number of routes or
  runs averages away;
* **recipe, routes like these**: the same over the route population: a paired bootstrap over routes plus sigma_g^2 / s_X,
  or, when X is a pool of seeds, a two-level bootstrap (seeds, then routes) that contains it.
Y is a fixed checkpoint (WoR's release) unless it is a pool as well.

Variance components (one run per seed and route in the seed pool, --seeds; default the first --pool with >= 3 members):
sigma_w^2 = mean over routes of the variance across seeds (= sigma_sr^2 + sigma_run^2); sigma_g^2 = var(seed means on the
common routes) - sigma_w^2 / N, floored at 0. sigma_run^2 is the pooled within-route variance of repeated runs of one
checkpoint (--repeats ARM, else every non-pool arm with repeats; without any repeats sigma_run = sigma_w, an upper bound).

p-values are two-sided sign-flip randomisation tests over routes (the route-population question). --family NAME=X-Y
collects the screened ideas: Holm (family-wise) and Benjamini-Hochberg (false discovery) adjustments over them.
The MDE table is the smallest difference resolved with 80% power at 5% two-sided, 2.8 SE, for N routes, s seeds and
k runs per seed against a fixed checkpoint with one run per route (Y fixed) and between two recipes.

    py scripts/analysis/thesis_stats.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_2026093* E:/MThesis_EXP/live_2026100* \
        --pool J=J20,J20s1,J20s2,J20s3 --compare E15-WOR --compare J-WOR --compare J-E15
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a9_merge import collect, parse_arm_specs  # noqa: E402

Z95, Z_MDE = 1.959964, 2.8016  # two-sided 95%; 1.96 + 0.8416 (80% power)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def runs_by_route(records_arm: dict) -> dict:
    return {r: [float(x["scores"]["score_composed"]) for x in rs] for r, rs in records_arm.items() if rs}


def pooled_within(arms: list) -> tuple:
    """Pooled within-route variance of repeated runs over several {route: [DS]} maps -> (variance, dof)."""
    ss = dof = 0
    for runs in arms:
        for xs in runs.values():
            if len(xs) >= 2:
                m = mean(xs)
                ss += sum((x - m) ** 2 for x in xs)
                dof += len(xs) - 1
    return (ss / dof if dof else float("nan")), dof


def components(seeds: dict, repeats: list | None = None, unbiased_sg: bool = False) -> dict:
    """seeds = {seed: {route: DS}} (one run each) -> sigma_w^2, sigma_g^2, sigma_run^2, sigma_sr^2 and their support.
    sigma_g^2 is the raw variance of the seed means by default (conservative: it still holds sigma_w^2 / N, which a few seeds
    cannot separate from a real shared shift); unbiased_sg subtracts it (often 0 with 4 seeds)."""
    common = sorted(set.intersection(*(set(v) for v in seeds.values())))
    s, n = len(seeds), len(common)
    if s < 2 or n < 2:
        raise ValueError(f"need >= 2 seeds on >= 2 common routes, got {s} seeds on {n}")
    sw2 = mean(statistics.variance([seeds[k][r] for k in seeds]) for r in common)
    means = [mean(seeds[k][r] for r in common) for k in seeds]
    sg2_raw = statistics.variance(means)
    sg2_unb = max(sg2_raw - sw2 / n, 0.0)
    sg2 = sg2_unb if unbiased_sg else sg2_raw
    srun2, dof = pooled_within(repeats or [])
    if not dof:
        srun2 = sw2
    return {"seeds": s, "routes": n, "sw2": sw2, "sg2": sg2, "sg2_raw": sg2_raw, "sg2_unbiased": sg2_unb,
            "srun2": min(srun2, sw2) if dof else srun2, "srun_dof": dof, "sr2": max(sw2 - (srun2 if dof else sw2), 0.0), "seed_means": means}


def boot_ci(d: list, n: int = 20000, seed: int = 0) -> tuple:
    rnd = random.Random(seed)
    ms = sorted(mean(rnd.choices(d, k=len(d))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def signflip_p(d: list, n: int = 20000, seed: int = 0) -> float:
    rnd = random.Random(seed)
    obs = abs(mean(d))
    hits = sum(abs(mean(x if rnd.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def hier_boot(pool: dict, ref: dict, rids: list, n: int = 5000, seed: int = 0) -> tuple:
    """Two-level bootstrap of mean(pool recipe - ref) over seeds, then routes."""
    rnd = random.Random(seed)
    keys = list(pool)
    ms = []
    for _ in range(n):
        ks = rnd.choices(keys, k=len(keys))
        rs = rnd.choices(rids, k=len(rids))
        ms.append(mean(mean(pool[k][r] for k in ks) - ref[r] for r in rs))
    ms.sort()
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def holm(ps: list) -> list:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def bh(ps: list) -> list:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adj, running = [0.0] * m, 1.0
    for rank, i in enumerate(order):
        running = min(running, ps[i] * m / (m - rank))
        adj[i] = running
    return adj


def compare(x: dict, y: dict, comp: dict, x_pool: dict | None = None, y_pool: dict | None = None, boot: int = 20000) -> dict:
    """x, y = {route: [DS runs]} (a pool's per-route runs are its seeds). Returns the three estimands' intervals."""
    rids = sorted(set(x) & set(y))
    n = len(rids)
    mx = {r: mean(x[r]) for r in rids}
    my = {r: mean(y[r]) for r in rids}
    d = [mx[r] - my[r] for r in rids]
    diff = mean(d)
    srun2, sw2, sg2 = comp["srun2"], comp["sw2"], comp["sg2"]

    def noise(runs: dict, pool, recipe: bool) -> float:   # sum over routes of the variance of that side's route mean
        if pool:      # one run per seed
            return sum(sw2 / len(runs[r]) for r in rids)
        if recipe:    # one seed and k runs of it: the seed's route deviation is not averaged by the runs
            return sum(comp["sr2"] + srun2 / len(runs[r]) for r in rids)
        return sum(srun2 / len(runs[r]) for r in rids)

    s_x = len(x_pool) if x_pool else 1
    s_y = len(y_pool) if y_pool else 1
    out = {"n": n, "diff": diff, "p_route": signflip_p(d, n=boot), "ci_route_boot": boot_ci(d, n=boot)}
    if not x_pool and not y_pool:
        se = math.sqrt(noise(x, None, False) + noise(y, None, False)) / n
        out["ci_checkpoint_fixed"] = (diff - Z95 * se, diff + Z95 * se)
    # recipe: X (and Y if a pool) carries seed noise; a fixed Y keeps run noise only
    se2 = (noise(x, x_pool, True) + noise(y, y_pool, bool(y_pool))) / n ** 2 + sg2 / s_x + (sg2 / s_y if y_pool else 0.0)
    out["ci_recipe_fixed"] = (diff - Z95 * math.sqrt(se2), diff + Z95 * math.sqrt(se2))
    if x_pool and not y_pool:
        out["ci_recipe_population"] = hier_boot({k: {r: v for r, v in s.items()} for k, s in x_pool.items()}, my,
                                                [r for r in rids if all(r in s for s in x_pool.values())], n=min(boot, 5000))
    else:
        lo, hi = out["ci_route_boot"]
        half = Z95 * math.sqrt(((hi - lo) / (2 * Z95)) ** 2 + sg2 / s_x + (sg2 / s_y if y_pool else 0.0))
        out["ci_recipe_population"] = (diff - half, diff + half)
    lo, hi = out["ci_recipe_population"]
    se_pop = (hi - lo) / (2 * Z95)
    out["p_recipe"] = math.erfc(abs(diff) / (se_pop * math.sqrt(2))) if se_pop > 0 else float("nan")   # normal approximation
    return out


def mde(comp: dict, sint2: float, n: int, s: int, k: int, y_fixed: bool = True, population: bool = True) -> float:
    """Smallest resolvable mean difference (2.8 SE): recipe with s seeds x k runs vs a fixed checkpoint (1 run per route), or
    vs another recipe with the same s and k; over routes like these (population) or on these routes (no interaction term)."""
    sr2, srun2, sg2 = comp["sr2"], comp["srun2"], comp["sg2"]
    side = sr2 / s + srun2 / (s * k)
    si2 = sint2 if population else 0.0
    if y_fixed:
        se2 = (si2 + side + srun2) / n + sg2 / s
    else:
        se2 = (si2 + 2 * side) / n + 2 * sg2 / s
    return Z_MDE * math.sqrt(se2)


def mde_switch(comp: dict, sint2: float, n: int, k: int, population: bool) -> float:
    """An evaluation-time switch on one checkpoint, k runs on and k off per route: no seed term. These routes (population
    False) leave only run noise, which more runs average away; routes like these add the switch x route interaction."""
    return Z_MDE * math.sqrt(((sint2 if population else 0.0) + 2 * comp["srun2"] / k) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=REGEX")
    ap.add_argument("--pool", action="append", default=[], metavar="NAME=SEED1,SEED2,..", help="seeds of one recipe (each member arm is one seed)")
    ap.add_argument("--seeds", default=None, help="the pool the seed components come from (default: first pool with >= 3 members)")
    ap.add_argument("--repeats", action="append", default=[], metavar="ARM", help="arm(s) with repeated runs of one checkpoint, for sigma_run")
    ap.add_argument("--compare", action="append", default=[], metavar="X-Y")
    ap.add_argument("--family", action="append", default=[], metavar="NAME=X-Y", help="screened ideas for the multiplicity table")
    ap.add_argument("--interaction", default=None, metavar="X-Y", help="recipe vs fixed checkpoint pair whose route x arm interaction sizes the recipe table (default: first --compare)")
    ap.add_argument("--interaction-recipe", default=None, metavar="X-Y", help="two of our recipes, for the recipe-vs-recipe column (default: --interaction)")
    ap.add_argument("--interaction-switch", default=None, metavar="X-Y", help="a switch on vs off on one checkpoint, for the switch table (skipped if absent)")
    ap.add_argument("--sg-unbiased", action="store_true", help="sigma_g^2 minus the route noise of a seed mean (default: the raw, conservative variance)")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    records, _ = collect(a.roots, parse_arm_specs(a.arm))
    runs = {arm: runs_by_route(rr) for arm, rr in records.items()}
    pools = {}
    for spec in a.pool:
        name, members = spec.split("=", 1)
        mem = [m for m in members.split(",") if m in runs]
        if len(mem) >= 2:
            pools[name] = {m: {r: mean(v) for r, v in runs[m].items()} for m in mem}
            common = set.intersection(*(set(v) for v in pools[name].values()))
            runs[name] = {r: [pools[name][m][r] for m in mem] for r in common}
            print(f"pool {name}: {len(mem)} seeds ({', '.join(mem)}), {len(common)} routes all of them drove")
        elif mem:   # one seed present: the idea still counts, as a single arm
            runs[name] = runs[mem[0]]
            print(f"pool {name}: only {mem[0]} has records; used as a single arm")
    seed_pool = a.seeds or next((p for p, v in pools.items() if len(v) >= 3), next(iter(pools), None))
    if seed_pool is None or seed_pool not in pools:
        ap.error("needs a --pool of >= 2 seeds with records for the seed components")
    rep = [runs[x] for x in (a.repeats or [k for k in runs if k not in pools and not any(k in v for v in pools.values())]) if x in runs]
    comp = components(pools[seed_pool], rep, unbiased_sg=a.sg_unbiased)
    res = {"components": {k: v for k, v in comp.items()}}
    print(f"\n== variance components (seed pool {seed_pool}: {comp['seeds']} seeds on {comp['routes']} routes; run repeats: dof {comp['srun_dof']}) ==")
    print(f"sigma_w (per route, across seeds) {math.sqrt(comp['sw2']):5.1f}   sigma_run (repeats of one checkpoint) {math.sqrt(comp['srun2']):5.1f}"
          f"{'' if comp['srun_dof'] else ' (no repeats: = sigma_w)'}   sigma_sr {math.sqrt(comp['sr2']):5.1f}   seed means " + ", ".join(f"{m:.1f}" for m in comp["seed_means"]))
    print(f"sigma_g (shift shared by all routes): raw spread of the seed means {math.sqrt(comp['sg2_raw']):.2f}, minus route noise "
          f"{math.sqrt(comp['sg2_unbiased']):.2f}; used: {math.sqrt(comp['sg2']):.2f} ({'unbiased' if a.sg_unbiased else 'raw, conservative'})")

    res["compare"] = {}
    print("\n== X - Y: three estimands (95% intervals) and the route sign-flip p ==")
    print(f"{'X - Y':22s} {'n':>4s} {'diff':>6s}  {'checkpoint, these routes':>25s}  {'recipe, these routes':>21s}  {'recipe, routes like these':>26s}"
          f"  {'p route':>8s} {'p recipe':>8s}")
    for spec in a.compare:
        x, y = spec.split("-", 1)
        if x not in runs or y not in runs:
            print(f"{spec:22s} (no records for {x if x not in runs else y})")
            continue
        o = compare(runs[x], runs[y], comp, pools.get(x), pools.get(y), boot=a.bootstrap)
        res["compare"][spec] = o
        f = lambda ci: f"[{ci[0]:+5.1f}, {ci[1]:+5.1f}]" if ci else "-"
        print(f"{spec:22s} {o['n']:4d} {o['diff']:+6.1f}  {f(o.get('ci_checkpoint_fixed')):>25s}  {f(o['ci_recipe_fixed']):>21s}  "
              f"{f(o['ci_recipe_population']):>26s}  {o['p_route']:8.4f} {o['p_recipe']:8.4f}")

    if a.family:
        print(f"\n== screened ideas as one family ({len(a.family)} comparisons): Holm (family-wise) and BH (false discovery) ==")
        rows = []
        for spec in a.family:
            name, pair = spec.split("=", 1)
            x, y = pair.split("-", 1)
            if x in runs and y in runs:
                rids = sorted(set(runs[x]) & set(runs[y]))
                d = [mean(runs[x][r]) - mean(runs[y][r]) for r in rids]
                rows.append({"name": name, "pair": pair, "n": len(rids), "diff": mean(d), "p": signflip_p(d, n=a.bootstrap)})
            else:
                print(f"  {name}: no records for {pair}")
        for r_, h, b in zip(rows, holm([r["p"] for r in rows]), bh([r["p"] for r in rows])):
            r_["holm"], r_["bh"] = h, b
            flag = "survives Holm" if h < 0.05 else ("BH only" if b < 0.05 else "")
            print(f"  {r_['name']:22s} {r_['pair']:28s} n={r_['n']:3d} diff {r_['diff']:+6.1f}  p {r_['p']:.4f}  Holm {h:.4f}  BH {b:.4f}  {flag}")
        res["family"] = rows

    def interaction(pair: str, kind: str):
        """sigma_int^2 = var of the paired route differences minus their noise. kind: switch (one checkpoint on both sides: the
        seed's route deviation cancels), vs_fixed (our recipe vs a fixed checkpoint), recipe (two of our recipes)."""
        x, y = pair.split("-", 1)
        if x not in runs or y not in runs:
            return None
        rids = sorted(set(runs[x]) & set(runs[y]))

        def nz(arm, r, recipe):
            if arm in pools:
                return comp["sw2"] / len(runs[arm][r])
            return (comp["sr2"] if recipe else 0.0) + comp["srun2"] / len(runs[arm][r])
        rx, ry = {"switch": (False, False), "vs_fixed": (True, False), "recipe": (True, True)}[kind]
        d = [mean(runs[x][r]) - mean(runs[y][r]) for r in rids]
        return max(statistics.variance(d) - mean(nz(x, r, rx) + nz(y, r, ry) for r in rids), 0.0)

    res["mde"] = {}
    if a.interaction_switch and (si2 := interaction(a.interaction_switch, "switch")) is not None:
        res["mde"]["switch"] = {"pair": a.interaction_switch, "sigma_int": math.sqrt(si2), "table": {}}
        print(f"\n== smallest resolvable effect of a switch on one checkpoint (80% power, 5% two-sided; switch x route interaction from "
              f"{a.interaction_switch}: sigma_int {math.sqrt(si2):.1f}) ==")
        print(f"{'routes':>7s} {'runs on + off':>14s} {'these routes':>13s} {'routes like these':>18s}")
        for n in (220, 103, 19):
            for k in (1, 2, 4, 8):
                v1, v2 = mde_switch(comp, si2, n, k, False), mde_switch(comp, si2, n, k, True)
                res["mde"]["switch"]["table"][f"n{n}_k{k}"] = {"these_routes": v1, "routes_like_these": v2}
                print(f"{n:7d} {k:14d} {v1:13.1f} {v2:18.1f}")
    pair = a.interaction or (a.compare[0] if a.compare else None)
    sint2 = interaction(pair, "vs_fixed") if pair else None
    if sint2 is not None:
        rp = a.interaction_recipe
        sir2 = interaction(rp, "recipe") if rp else None
        sir2 = sint2 if sir2 is None else sir2
        res["mde"]["recipe"] = {"pair_vs_fixed": pair, "sigma_int": math.sqrt(sint2), "pair_recipe": rp or pair, "sigma_int_recipe": math.sqrt(sir2), "table": {}}
        print(f"\n== smallest resolvable difference for a recipe (80% power, 5% two-sided; interaction vs a fixed checkpoint "
              f"from {pair}: {math.sqrt(sint2):.1f}; recipe vs recipe from {rp or pair}: {math.sqrt(sir2):.1f}) ==")
        print(f"{'':18s} {'vs a fixed checkpoint':>27s} {'recipe vs recipe':>27s}")
        print(f"{'routes':>7s} {'seeds':>5s} {'runs':>4s} {'these routes':>13s} {'like these':>13s} {'these routes':>13s} {'like these':>13s}")
        for n in (220, 103, 19):
            for s in (1, 3, 5):
                for k in (1, 2):
                    v = [mde(comp, sint2, n, s, k, True, False), mde(comp, sint2, n, s, k, True, True),
                         mde(comp, sir2, n, s, k, False, False), mde(comp, sir2, n, s, k, False, True)]
                    res["mde"]["recipe"]["table"][f"n{n}_s{s}_k{k}"] = {"vs_fixed_these": v[0], "vs_fixed": v[1], "recipe_vs_recipe_these": v[2],
                                                                       "recipe_vs_recipe": v[3]}
                    print(f"{n:7d} {s:5d} {k:4d} " + " ".join(f"{x:13.1f}" for x in v))
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
