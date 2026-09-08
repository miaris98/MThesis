#!/usr/bin/env python3
"""Paired route-level comparison of two closed-loop evaluation runs.

The open-loop half of this project already learned, twice, that an unpaired comparison
over a small route set resolves nothing (13.17, 13.19, and again in 13.29 where three
same-signed point estimates all turned out to be coin flips). Closed-loop routes are
fewer and noisier than held-out frames, so the same discipline applies with more force:
this compares two runs **paired over the shared route ids**, which is only meaningful
because `eval_wor_closed_loop.py` drives byte-identical routes and traffic for a given
`--route_seed`.

The metric compared is the Driving Score by default, but route completion and the
infraction penalty can be compared separately - as 13.29 showed, a change can move one
component of a composite metric a long way while leaving the composite still.

    python compare_closed_loop.py --a baseline_s0.json --b geom_s0.json
    python compare_closed_loop.py --a baseline_s0.json --b geom_s0.json --metric route_completion
"""
import argparse
import json
from typing import Dict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Paired bootstrap over closed-loop routes")
    p.add_argument("--a", type=str, required=True, help="First run's JSON (the reference arm)")
    p.add_argument("--b", type=str, required=True, help="Second run's JSON")
    p.add_argument("--metric", type=str, default="driving_score",
                   choices=["driving_score", "route_completion", "infraction_penalty",
                            "raw_route_completion", "offroad_fraction"])
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--higher_is_better", type=int, default=1,
                   help="1 for scores, 0 for error-like metrics such as offroad_fraction")
    return p.parse_args()


def load(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        payload = json.load(f)
    return {r["route_id"]: r for r in payload["routes"]}, payload


def main():
    args = parse_args()
    routes_a, payload_a = load(args.a)
    routes_b, payload_b = load(args.b)

    if payload_a.get("route_seed") != payload_b.get("route_seed"):
        print(f"[WARNING] route_seed differs ({payload_a.get('route_seed')} vs "
              f"{payload_b.get('route_seed')}). The two runs did not drive the same routes, "
              f"so pairing them is invalid - compare them unpaired, or re-run with a shared seed.")
    if payload_a.get("town") != payload_b.get("town"):
        print(f"[WARNING] town differs ({payload_a.get('town')} vs {payload_b.get('town')}).")

    shared = sorted(set(routes_a) & set(routes_b))
    if not shared:
        raise SystemExit("No route ids in common - nothing to pair.")
    if len(shared) < len(routes_a) or len(shared) < len(routes_b):
        print(f"[NOTE] pairing on {len(shared)} shared routes "
              f"(A has {len(routes_a)}, B has {len(routes_b)}).")

    a = np.array([routes_a[r][args.metric] for r in shared], dtype=float)
    b = np.array([routes_b[r][args.metric] for r in shared], dtype=float)
    d = a - b

    rng = np.random.RandomState(args.seed)
    idx = rng.randint(0, len(shared), size=(args.bootstrap, len(shared)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed = float(d.mean())

    better = ">" if args.higher_is_better else "<"
    a_wins = int((d > 0).sum()) if args.higher_is_better else int((d < 0).sum())

    print("=" * 76)
    print(f"  PAIRED closed-loop comparison on '{args.metric}' over {len(shared)} routes")
    print("=" * 76)
    print(f"  A : {a.mean():.4f}   ({args.a})")
    print(f"  B : {b.mean():.4f}   ({args.b})")
    print(f"  paired difference A - B       : {observed:+.4f}")
    print(f"  bootstrap 95% CI on difference: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  A {better} B on {a_wins} of {len(shared)} routes")
    print()

    a_better = (lo > 0) if args.higher_is_better else (hi < 0)
    b_better = (hi < 0) if args.higher_is_better else (lo > 0)

    if a_better:
        print("  VERDICT: A is better, and the margin survives the route draw.")
    elif b_better:
        print("  VERDICT: B is better, and the margin survives the route draw.")
    else:
        print("  VERDICT: NOT SUPPORTED. The CI on the difference spans zero, so with")
        print(f"  {len(shared)} routes this margin is indistinguishable from which routes")
        print("  happened to be drawn. Reporting it as a result would be an error.")
    print("=" * 76)

    # Termination reasons are reported alongside because a driving-score difference is
    # uninterpretable without them: the same score can come from short clean routes or
    # from long ones that ended in a wall.
    for label, payload in (("A", payload_a), ("B", payload_b)):
        s = payload.get("summary", {})
        print(f"  {label} terminations: {s.get('termination_reasons')}  "
              f"| km driven {s.get('total_km')}")


if __name__ == "__main__":
    main()
