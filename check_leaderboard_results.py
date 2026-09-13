#!/usr/bin/env python3
"""Refuse to accept a Leaderboard results file whose scores came from an agent that crashed.

WHY THIS EXISTS
---------------
A crashed agent does not make the Leaderboard produce an error. `leaderboard_evaluator.py`
catches `AgentError`, tags the route with FAILURE_MESSAGES["Agent_runtime"] -> "Agent crashed",
stops it, and then *scores it anyway*: `score_route` 0, `score_composed` 0, most criteria
SUCCESS because nothing ever had a chance to go wrong. That zero is then averaged into
`global_record.scores_mean` alongside genuine driving results.

The evaluator's own escalation does not catch it either. In `statistics_manager.py`'s
`compute_global_statistics()`, only two route statuses raise the entry status:

    if 'Simulation crashed' in route_status:              entry_status = 'Crashed'
    elif "Agent's sensors were invalid" in route_status:  entry_status = 'Rejected'

"Agent crashed" is absent, so a run in which the agent crashed on every single route still
reports `entry_status: "Finished"`, `eligible: true`, and a well-formed mean driving score.
That is exactly what happened here on the first official-harness run (11.14): a one-line
AttributeError on the first `run_step` of the route, and a results file that looked like a
policy which drove badly rather than one which never drove at all.

The distinction this script draws is between a route the agent *lost* and a route the agent
never *ran*. Both score 0. Only the first is a measurement.

usage:
    check_leaderboard_results.py results.json [--expect-routes N] [--quiet]

exit codes:
    0  clean - every recorded route is a real driving outcome
    1  contaminated - at least one route scored an agent-side crash, or a count mismatch
    2  unusable - file missing, truncated, or not a leaderboard results file
"""
import argparse
import json
import sys

# Statuses where the number is not a measurement of driving: the agent process failed, so the
# route's 0 reflects our bug rather than the policy's behaviour. Matched case-insensitively as
# substrings, because the evaluator writes them as "Failed - <message>".
AGENT_FAULT = (
    "agent crashed",             # FAILURE_MESSAGES["Agent_runtime"] - exception in run_step
    "agent couldn't be set up",  # FAILURE_MESSAGES["Agent_init"]    - exception in setup()
    "agent's sensors were invalid",
)

# Simulator-side failures. Also not a measurement of the policy, but distinguished from
# AGENT_FAULT because their usual causes - a CARLA server that died, an RPC timeout, a map that
# failed to stream in - are transient, so a retry on a fresh server has a real chance of turning
# one into a genuine result. (The first smoke run produced exactly this: "time-out of 600000ms
# while waiting for the simulator".)
SIM_FAULT = (
    "simulation crashed",
    "couldn't be set up",   # route/scenario setup failure not already matched as Agent_init
)

# Statuses that are legitimate, scoreable driving outcomes, including bad ones. Listed
# explicitly so that a status matching none of the tuples is reported as unrecognised rather
# than silently assumed to be fine - this evaluator's vocabulary may grow.
#
# These are NEVER retried, and that is a correctness property, not an oversight: a route the
# policy lost by getting blocked, deviating or timing out is a measurement of the policy. Re-
# running those until they come out better would not be error recovery, it would be selecting
# the favourable half of a noisy distribution and reporting it as the result.
DRIVING_OUTCOME = (
    "completed",
    "perfect",
    "agent got blocked",
    "agent timed out",
    "agent deviated from the route",
    "route timeout",
)


def classify(status):
    s = (status or "").lower()
    for marker in AGENT_FAULT:
        if marker in s:
            return "agent_fault"
    for marker in DRIVING_OUTCOME:
        if marker in s:
            return "driving"
    for marker in SIM_FAULT:
        if marker in s:
            return "sim_fault"
    return "unknown"


def subset_id(route_id):
    """'RouteScenario_12_rep0' -> '12', the id --routes-subset expects.

    route_parser builds the name as "RouteScenario_{route_id}" from the XML `id` attribute, and
    the statistics manager appends "_rep{n}", so this inverts exactly that construction.
    """
    core = str(route_id).split("_rep")[0]
    return core.split("_", 1)[1] if "_" in core else core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--expect-routes", type=int, default=None,
                    help="fail if the file does not hold exactly this many route records")
    ap.add_argument("--abort-after", type=int, default=None, metavar="N",
                    help="watchdog mode: exit 3 once at least N routes are recorded and EVERY "
                         "one of them is an agent-side failure. Used mid-run to kill a doomed "
                         "112-route pass early instead of letting it grind out zeros.")
    ap.add_argument("--list-retryable", action="store_true",
                    help="print a --routes-subset string naming the routes that failed for a "
                         "reason that is not a measurement of the policy (agent-side or "
                         "simulator-side), and so are worth re-running. Prints nothing and "
                         "exits 0 when there is nothing to retry. Driving outcomes are never "
                         "listed - see DRIVING_OUTCOME.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        with open(args.results) as fh:
            data = json.load(fh)
        records = data["_checkpoint"]["records"]
    except FileNotFoundError:
        if args.abort_after is not None:
            return 0  # the run has not written its first checkpoint yet; nothing to judge
        print(f"UNUSABLE: {args.results} does not exist", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # A run killed mid-write leaves truncated JSON. That is not a clean result either.
        # In watchdog mode this is usually just a read racing the evaluator's own write, so
        # say nothing and let the next poll settle it rather than killing a healthy run.
        if args.abort_after is not None:
            return 0
        print(f"UNUSABLE: {args.results} is not a readable leaderboard results file: {exc}",
              file=sys.stderr)
        return 2

    buckets = {"agent_fault": [], "sim_fault": [], "driving": [], "unknown": []}
    for rec in records:
        buckets[classify(rec.get("status"))].append(rec)

    entry_status = data.get("entry_status")
    n_total = len(records)
    n_fault = len(buckets["agent_fault"]) + len(buckets["sim_fault"])
    n_unknown = len(buckets["unknown"])

    if args.list_retryable:
        # Only agent- and simulator-side failures. A route the policy lost stays lost.
        ids = [subset_id(r.get("route_id"))
               for r in buckets["agent_fault"] + buckets["sim_fault"]]
        # Preserve route order and drop duplicates without needing them sorted numerically.
        seen, ordered = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        if ordered:
            print(",".join(ordered))
        return 0

    if args.abort_after is not None:
        # Deliberately requires *every* recorded route to have failed agent-side. A policy that
        # genuinely drives badly still produces "driving" statuses, so this cannot fire on a
        # merely bad model - only on one that is not running at all.
        if n_total >= args.abort_after and n_fault == n_total:
            print(f"ABORT: first {n_total} routes all failed agent-side "
                  f"({records[0].get('status')!r}). The agent is not driving; killing the run "
                  f"rather than scoring zeros for the remainder.", file=sys.stderr)
            return 3
        return 0

    if not args.quiet:
        print(f"routes recorded : {n_total}")
        print(f"  driving results: {len(buckets['driving'])}")
        print(f"  agent-side fail: {len(buckets['agent_fault'])}")
        print(f"  simulator fail : {len(buckets['sim_fault'])}")
        print(f"  unrecognised   : {n_unknown}")
        print(f"entry_status    : {entry_status}")

        scored = [r["scores"]["score_composed"] for r in buckets["driving"]
                  if "scores" in r and "score_composed" in r["scores"]]
        if scored:
            print(f"mean score_composed over driving routes only: "
                  f"{sum(scored) / len(scored):.2f}  (n={len(scored)})")
        reported = data.get("_checkpoint", {}).get("global_record", {}) \
                       .get("scores_mean", {}).get("score_composed")
        if reported is not None:
            print(f"mean score_composed as reported by the evaluator: {reported}")
            if n_fault:
                print("  ^ this figure includes the agent-side failures as zeros and is NOT "
                      "a valid measurement of the policy")

    failed = False

    if n_fault:
        failed = True
        print(f"\nCONTAMINATED: {n_fault}/{n_total} routes did not measure the policy - the "
              f"agent or the simulator failed before or during driving.", file=sys.stderr)
        for rec in (buckets["agent_fault"] + buckets["sim_fault"])[:10]:
            print(f"  route {rec.get('route_id')}: {rec.get('status')}", file=sys.stderr)
        if n_fault > 10:
            print(f"  ... and {n_fault - 10} more", file=sys.stderr)
        print("These routes are scored 0 and averaged into the reported mean. Fix the agent "
              "and re-run them; do not report this file's global_record.", file=sys.stderr)

    if n_unknown:
        failed = True
        print(f"\nUNRECOGNISED: {n_unknown} routes carry a status this checker does not know "
              f"how to classify. Treating as a failure rather than assuming it is fine:",
              file=sys.stderr)
        for rec in buckets["unknown"][:10]:
            print(f"  route {rec.get('route_id')}: {rec.get('status')!r}", file=sys.stderr)

    if args.expect_routes is not None and n_total != args.expect_routes:
        failed = True
        print(f"\nINCOMPLETE: expected {args.expect_routes} route records, found {n_total}.",
              file=sys.stderr)

    if failed:
        return 1

    if not args.quiet:
        print("\nOK: every recorded route is a real driving outcome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
