#!/usr/bin/env python3
"""Merge a Leaderboard results file with the results of retry passes over its failed routes.

WHY THIS EXISTS
---------------
`check_leaderboard_results.py` can tell that some routes failed for a reason that is not a
measurement of the policy (the agent raised, or the simulator died). Retrying those routes
produces a second, smaller results file covering only them. This script puts the two back
together into one file with the shape the evaluator itself would have produced.

Retrying cannot be done in place. `RouteIndexer.validate_and_resume()` resumes at a single
`progress[0]` index and requires `records[i]['route_id']` to match config `i` positionally, so
there is no way to ask one run to redo routes 7, 23 and 55. Hence: re-run them as their own
pass with `--routes-subset`, into their own checkpoint file, and merge afterwards.

The global record is NOT recomputed by hand here. `StatisticsManager.compute_global_statistics()`
already does it, including things that are easy to get subtly wrong - the per-kilometre
normalisation of infraction rates, the `outside_route_lanes` percentage parsed back out of its
message string, the 'Perfect' -> 'Completed' -> 'Failed' downgrade chain. This script builds the
merged record list and then hands it to that method, so the output is the evaluator's own
arithmetic rather than an approximation of it.

usage:
    merge_leaderboard_results.py --out merged.json --total 112 base.json retry1.json [retry2.json ...]

Later files win: a route present in a retry file replaces the same route in the base, but only
when the retry actually produced a driving outcome - a retry that crashed again leaves the
earlier record in place, so merging can never make a file look worse-founded than its inputs.
"""
import argparse
import os
import sys

# The leaderboard/scenario_runner packages must already be importable; run this with the same
# interpreter and PYTHONPATH as the evaluator (run_leaderboard_official.sh exports both).
try:
    from leaderboard.utils.statistics_manager import StatisticsManager, to_route_record
    from leaderboard.utils.checkpoint_tools import fetch_dict
except ImportError as exc:
    sys.exit(f"cannot import the leaderboard package ({exc}).\n"
             "Run this with /workspace/venv_carla/bin/python and the PYTHONPATH that "
             "run_leaderboard_official.sh sets.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_leaderboard_results import classify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, required=True,
                    help="total number of routes in the full evaluation - the denominator the "
                         "means are computed over. Must be the full route count, not the "
                         "number of records being merged, or every mean comes out too high.")
    ap.add_argument("inputs", nargs="+", help="base results file, then each retry file in order")
    args = ap.parse_args()

    # route_id -> (record dict, source file). Later inputs overwrite earlier ones, but only if
    # the newer record is a real driving outcome.
    merged = {}
    order = []
    for path in args.inputs:
        data = fetch_dict(path)
        if not data:
            print(f"warning: {path} is empty or unreadable, skipping", file=sys.stderr)
            continue
        for rec in (data.get("_checkpoint", {}).get("records") or []):
            rid = rec.get("route_id")
            if rid is None:
                continue
            if rid not in merged:
                order.append(rid)
                merged[rid] = (rec, path)
                continue
            prev_rec, prev_path = merged[rid]
            if classify(rec.get("status")) == "driving":
                merged[rid] = (rec, path)
                print(f"  {rid}: {prev_rec.get('status')!r} ({os.path.basename(prev_path)})"
                      f" -> {rec.get('status')!r} ({os.path.basename(path)})")
            else:
                print(f"  {rid}: retry in {os.path.basename(path)} also failed "
                      f"({rec.get('status')!r}); keeping earlier record")

    records = [merged[rid][0] for rid in order]

    still_bad = [r for r in records if classify(r.get("status")) != "driving"]
    print(f"\nmerged {len(records)} routes from {len(args.inputs)} file(s); "
          f"{len(records) - len(still_bad)} are driving outcomes, {len(still_bad)} still are not")

    manager = StatisticsManager(args.out, args.out.replace(".json", "_live.txt"))
    for rec in records:
        manager._results.checkpoint.records.append(to_route_record(rec))
    manager.sort_records()           # reindexes 0..n-1 in route order
    manager._total_routes = args.total
    manager.save_progress(len(records), args.total)
    manager.save_entry_status("Finished")
    manager.compute_global_statistics()
    manager.write_statistics()

    print(f"wrote {args.out}")
    if still_bad:
        print("\nNOTE: this file still contains routes that did not measure the policy. "
              "Run check_leaderboard_results.py on it before quoting any number from it.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
