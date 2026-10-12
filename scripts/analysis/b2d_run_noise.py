#!/usr/bin/env python3
"""Run-to-run noise, estimands and the collision lottery in Bench2Drive results (S-100, S-101).

Given several closed-loop runs per agent (Leaderboard result JSONs), it reports:

* per-agent pooled within-route SD across repeated runs of one checkpoint, and per route;
* each agent minus a reference agent (default WoR), with a route-population CI (paired bootstrap over routes)
  and a fixed-route-set CI (run noise only, analytic, pooled variances; TODO_ACTIVE A30);
* where the DS goes: 100 - DS = (100 - RC) + RC * (1 - penalty);
* the collision lottery: per-route mean/var of the vehicle-collision count k and the Poisson prediction
  E[DS] = RC exp(-0.4 lam), SD[DS] = RC sqrt(exp(-0.64 lam) - exp(-0.8 lam)), since each collision is x0.60.

Runs come from --run AGENT=PATH (repeatable; paths relative to the experiment root, src/config/paths.py) or from
--preset b2d20_2026-09-28, the file set S-100/S-101 used. Harness failures ("couldn't be set up") are dropped;
every other status is a driving outcome.

    python scripts/analysis/b2d_run_noise.py --preset b2d20_2026-09-28
    python scripts/analysis/b2d_run_noise.py --run WoR=a.json --run WoR=b.json --run E=c.json --reference WoR
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import paths  # noqa: E402

VEHICLE_COLLISION_PENALTY = 0.6  # Bench2Drive statistics_manager.PENALTY_VALUE_DICT

_B, _S, _V, _X = ("live_20260925_boxB_5070_carlaeval", "live_20260926_boxS_pro4000_carlaeval",
                  "live_20260927_boxV_pro4000_carla", "live_20260927_boxX2_3090_carlaeval")
#: The valid real-backbone 19-route runs of S-100/S-101 (b19_* = random backbone and brkcreep excluded).
PRESETS = {
    "b2d20_2026-09-28": {
        "WoR": [f"{_V}/bench2drive_out/v2_wor_lb_x.json", f"{_X}/bench2drive_out/v2_wor_lb_r2_x.json",
                f"{_S}/bench2drive_out/w_wor_lb_x.json"],
        "E_e15": [f"{_S}/bench2drive_out/w_armE_e15_x.json"],
        "D_e15": [f"{_B}/bench2drive_out/v_armD_e15_x.json", f"{_S}/bench2drive_out/w_armD_e15_r2_x.json",
                  f"{_S}/bench2drive_out/w_armD_e15_r3_x.json"],
        "A_e15": [f"{_B}/bench2drive_out/v_armA_e15_r12_x.json", f"{_S}/bench2drive_out/w_armA_e15_r12b_x.json",
                  f"{_B}/bench2drive_out/x13_armA_e15_x.json", f"{_B}/bench2drive_out/x2_armA_e15_x.json",
                  f"{_B}/bench2drive_out/tel_armA_e15_x.json"],
        "B_e15": [f"{_B}/bench2drive_out/v_armB_e15_r12_x.json", f"{_S}/bench2drive_out/w_armB_e15_r12b_x.json",
                  f"{_B}/bench2drive_out/x13_armB_e15_x.json", f"{_B}/bench2drive_out/x2_armB_e15_x.json"],
        "A_swa": [f"{_B}/bench2drive_out/v_armA_swa_x.json", f"{_S}/bench2drive_out/w_armA_swa_x.json"],
        "G_e20": [f"{_B}/bench2drive_out/v_armG_e20_x.json", f"{_S}/bench2drive_out/w_armG_e20_x.json"],
    },
}


def load_records(path: Path) -> Dict[str, dict]:
    """route id -> record, harness failures dropped."""
    out = {}
    for r in json.load(open(path))["_checkpoint"]["records"]:
        if "set up" in r["status"]:
            continue
        out[r["route_id"].split("_")[1]] = r
    return out


def poisson_ds(rc: float, lam: float) -> tuple:
    """(mean, sd) of DS when vehicle collisions are Poisson(lam) and each multiplies the score by 0.6."""
    p = VEHICLE_COLLISION_PENALTY
    mean = rc * math.exp(-(1 - p) * lam)
    var = rc ** 2 * (math.exp(-(1 - p * p) * lam) - math.exp(-2 * (1 - p) * lam))
    return mean, math.sqrt(max(0.0, var))


def pooled_var(runs_by_route: Dict[str, List[float]], routes=None) -> tuple:
    ss, dof = 0.0, 0
    for r, xs in runs_by_route.items():
        if routes is not None and r not in routes:
            continue
        if len(xs) >= 2:
            m = sum(xs) / len(xs)
            ss += sum((x - m) ** 2 for x in xs)
            dof += len(xs) - 1
    return (ss / dof if dof else float("nan")), dof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="AGENT=PATH")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--reference", default="WoR")
    ap.add_argument("--root", type=Path, default=None, help="base for relative paths (default: experiment root)")
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()

    base = args.root or paths.exp_root()
    spec = collections.defaultdict(list)
    if args.preset:
        for agent, files in PRESETS[args.preset].items():
            spec[agent] += files
    for item in args.run:
        agent, _, path = item.partition("=")
        spec[agent].append(path)
    if args.reference not in spec:
        ap.error(f"no runs for the reference agent {args.reference!r}")

    ds = {a: collections.defaultdict(list) for a in spec}      # agent -> route -> [DS]
    recs = {a: collections.defaultdict(list) for a in spec}    # agent -> route -> [record]
    for agent, files in spec.items():
        for f in files:
            p = Path(f) if Path(f).is_absolute() else base / f
            for rid, rec in load_records(p).items():
                ds[agent][rid].append(rec["scores"]["score_composed"])
                recs[agent][rid].append(rec)
    ref = args.reference
    routes = sorted(ds[ref], key=int)
    ours = [a for a in spec if a != ref]

    print("== within-route run SD (pooled over repeated runs of one checkpoint)")
    for a in spec:
        v, dof = pooled_var(ds[a])
        print(f"  {a:8s} sd_run {math.sqrt(v) if dof else float('nan'):5.1f}  (dof {dof})")

    def route_var(r, agents):
        ss = dof = 0
        for a in agents:
            xs = ds[a].get(r, [])
            if len(xs) >= 2:
                m = sum(xs) / len(xs); ss += sum((x - m) ** 2 for x in xs); dof += len(xs) - 1
        return ss, dof
    tot = [route_var(r, ours) for r in routes]
    glob_ours = sum(s for s, _ in tot) / max(1, sum(d for _, d in tot))
    glob_ref = pooled_var(ds[ref])[0]

    print(f"\n== agent - {ref}: route-population CI (paired bootstrap over routes) vs fixed-route-set CI (run noise)")
    rng = random.Random(0)
    for a in ours:
        rs = [r for r in routes if ds[a].get(r)]
        n = len(rs)
        d = [sum(ds[a][r]) / len(ds[a][r]) - sum(ds[ref][r]) / len(ds[ref][r]) for r in rs]
        mean = sum(d) / n
        boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.bootstrap))
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        var = sum(glob_ours / len(ds[a][r]) + glob_ref / len(ds[ref][r]) for r in rs)
        half = 1.96 * math.sqrt(var) / n
        print(f"  {a:8s} n={n:2d} diff {mean:+5.1f} | route-population [{lo:+5.1f}, {hi:+5.1f}]"
              f" | fixed-set [{mean - half:+5.1f}, {mean + half:+5.1f}]")

    print("\n== where the DS goes (per route): 100 - DS = not finishing + penalties")
    for a in spec:
        rc_loss = pen = n = 0
        for rid, rl in recs[a].items():
            for r in rl:
                s = r["scores"]; n += 1
                rc_loss += 100 - s["score_route"]; pen += s["score_route"] * (1 - s["score_penalty"])
        if n:
            print(f"  {a:8s} lost {(rc_loss + pen) / n:5.1f} = not finishing {rc_loss / n:5.1f} + penalties {pen / n:5.1f}  ({n} route-runs)")

    print("\n== collision lottery over our agents' runs (k = vehicle collisions per run)")
    print("   route  runs  mean k  var k   obs sd(DS)  Poisson sd(DS)   obs mean  Poisson mean")
    gain = cnt = 0
    for r in routes:
        rows = [(len(x["infractions"].get("collisions_vehicle", [])), x["scores"]["score_composed"], x["scores"]["score_route"])
                for a in ours for x in recs[a].get(r, [])]
        cnt += len(rows)
        gain += sum(d / (VEHICLE_COLLISION_PENALTY ** k) - d for k, d, _ in rows if k)
        if len(rows) < 2:
            continue
        ks = [k for k, _, _ in rows]; dsr = [d for _, d, _ in rows]
        lam = sum(ks) / len(ks); vk = sum((k - lam) ** 2 for k in ks) / (len(ks) - 1)
        md = sum(dsr) / len(dsr); sd = math.sqrt(sum((x - md) ** 2 for x in dsr) / (len(dsr) - 1))
        pm, psd = poisson_ds(sum(c for _, _, c in rows) / len(rows), lam)
        if lam > 0 or sd > 0:
            kind = "systematic" if lam > 0.3 and vk < 0.5 * lam else ("lottery" if lam > 0.3 else "")
            print(f"  {r:>6s} {len(rows):5d}  {lam:5.2f}  {vk:5.2f}   {sd:8.1f}   {psd:10.1f}   {md:10.1f}  {pm:10.1f}  {kind}")
    print(f"\n  ceiling: removing vehicle collisions at unchanged RC adds {gain / max(1, cnt):+.1f} DS per route-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
