#!/usr/bin/env python3
"""TODO B2: GTrXL vs ResNet on the 16 B35 checkpoints (env30000, upd28004) scored with 30 episodes, no sticky actions.

Reads every `eval_ep30_sim<S>_st0_env30000_upd28004.json` under the given roots (default: the live_20261003_* Atari pulls on
E:) and reports, per trunk: the mean of the seed means, the median, tunnel rate P(score >= 200), a bootstrap CI over seeds
and a permutation test for the trunk difference; the seed x episode variance components (one-way random effects) and what
more seeds or more episodes buy (the Neyman allocation for the next GPU-hour); the same runs at 64 simulations (B24).

    py scripts/analysis/b2_trunk_analysis.py [--sims 16] [--vs 64]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st

HOME = "E:/MThesis_EXP"


def load(sims: int) -> dict:
    out = {}
    pat = f"eval_ep30_sim{sims}_st0_env30000_upd28004.json"
    for f in glob.glob(f"{HOME}/live_20261003_box*_atari/**/{pat}", recursive=True):
        out[f.replace("\\", "/").split("/")[-3]] = json.load(open(f))
    return out


def trunk(run: str) -> str:
    return "GTrXL" if "gtrxl" in run else "ResNet"


def boot(vals, n=20000, seed=0):
    rnd = random.Random(seed)
    m = sorted(st.fmean(rnd.choices(vals, k=len(vals))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def boot_diff(a, b, n=20000, seed=0):
    rnd = random.Random(seed)
    d = sorted(st.fmean(rnd.choices(a, k=len(a))) - st.fmean(rnd.choices(b, k=len(b))) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n)]


def perm_p(a, b, n=20000, seed=0):
    rnd = random.Random(seed)
    obs = abs(st.fmean(a) - st.fmean(b)); pool = a + b; k = len(a); hits = 0
    for _ in range(n):
        rnd.shuffle(pool)
        hits += abs(st.fmean(pool[:k]) - st.fmean(pool[k:])) >= obs
    return hits / n


def components(groups):
    """One-way random effects: per-episode within-run variance and between-run variance of the run means."""
    k = [len(g) for g in groups]; n = len(groups); kbar = st.fmean(k)
    within = st.fmean(st.variance(g) for g in groups)
    means = [st.fmean(g) for g in groups]
    between = max(st.variance(means) - within / kbar, 0.0)
    return between, within


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--vs", type=int, default=64, help="compare against this simulation count where available")
    a = ap.parse_args()
    d = load(a.sims)
    by = {t: {r: v for r, v in d.items() if trunk(r) == t} for t in ("GTrXL", "ResNet")}
    print(f"B2 (30 episodes, {a.sims} sims, no sticky): {len(by['GTrXL'])} GTrXL, {len(by['ResNet'])} ResNet checkpoints\n")
    means = {}
    for t, runs in by.items():
        m = [v["mean"] for v in runs.values()]; means[t] = m
        sc = [x for v in runs.values() for x in v["scores"]]
        lo, hi = boot(m)
        print(f"{t:6s} seed means: mean {st.fmean(m):6.1f} [{lo:6.1f}, {hi:6.1f}]  median {st.median(m):6.1f}  "
              f"SD {st.stdev(m):6.1f}  P(score>=200) {100 * sum(x >= 200 for x in sc) / len(sc):4.1f}%  "
              f"runs with mean>=100: {sum(x >= 100 for x in m)}/{len(m)}  runs at 0: {sum(x == 0 for x in m)}")
    g, r = means["GTrXL"], means["ResNet"]
    lo, hi = boot_diff(g, r)
    print(f"\nGTrXL - ResNet: {st.fmean(g) - st.fmean(r):+.1f}  bootstrap over seeds [{lo:+.1f}, {hi:+.1f}]  "
          f"permutation p = {perm_p(list(g), list(r)):.2f}")
    print("distinct scores per checkpoint (of 30 episodes): " +
          ", ".join(f"{len(set(v['scores']))}" for v in d.values()) + "  -> episodes are not independent draws")

    print("\nVariance components (per trunk): between-seed SD, within-seed (episode) SD, and the SD of a trunk mean")
    for t, runs in by.items():
        groups = [v["scores"] for v in runs.values()]
        b, w = components(groups); n = len(groups)
        print(f"{t:6s} between-seed SD {b ** 0.5:6.1f}  within-seed SD {w ** 0.5:6.1f}  "
              f"ICC {b / (b + w) if b + w else 0:4.2f}")
        for nseeds, k in ((n, 10), (n, 30), (n, 100), (2 * n, 10), (2 * n, 30), (4 * n, 10)):
            sd = (b / nseeds + w / (nseeds * k)) ** 0.5
            print(f"        {nseeds:3d} seeds x {k:3d} episodes: SD of the trunk mean {sd:5.1f}")

    d2 = load(a.vs)
    both = sorted(set(d) & set(d2))
    if both:
        print(f"\nB24: {a.vs} vs {a.sims} simulations, same checkpoints and games ({len(both)} done so far; "
              f"the ones that finish first are the better-playing agents, so this sample is biased):")
        diffs = []
        for r_ in both:
            diffs.append(d2[r_]["mean"] - d[r_]["mean"])
            print(f"  {r_:32s} {d[r_]['mean']:6.1f} -> {d2[r_]['mean']:6.1f}  ({diffs[-1]:+6.1f})  "
                  f"tunnel {100 * d[r_]['tunnel_rate']:3.0f}% -> {100 * d2[r_]['tunnel_rate']:3.0f}%")
        print(f"  mean change {st.fmean(diffs):+.1f}" + (f" bootstrap [{boot(diffs)[0]:+.1f}, {boot(diffs)[1]:+.1f}]" if len(diffs) > 2 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
