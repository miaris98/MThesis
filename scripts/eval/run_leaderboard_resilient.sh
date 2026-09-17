#!/usr/bin/env bash
# Runs one arm over a route set and RETRIES the routes that failed for reasons which are not a
# measurement of the policy, restarting CARLA fresh for each attempt, then merges everything
# into a single results file.
#
# usage: scripts/eval/run_leaderboard_resilient.sh <label> <arch> <ckpt> <routes_xml> [subset] [port] [tm_port] [max_attempts]
#
# WHAT GETS RETRIED, AND WHAT DELIBERATELY DOES NOT
# --------------------------------------------------
# Retried: "Agent crashed", "Agent couldn't be set up", "Simulation crashed" - the agent process
# or the simulator failed, so the route's 0 measures our infrastructure, not the driving. On this
# hardware the common ones are transient (a CARLA server that wedged, the 600s RPC timeout seen
# on the first smoke run), which is exactly what a fresh server and a second go can fix.
#
# NOT retried: "Completed", "Agent got blocked", "Agent timed out", "Agent deviated from the
# route". These are the policy's actual behaviour. Re-running them until they come out better
# would not be error recovery - it would be sampling a noisy distribution and keeping the
# favourable draws, which is how you manufacture a result rather than measure one.
#
# The loop also stops as soon as an attempt stops making progress: if a retry pass fails on
# exactly the same routes as the pass before it, the fault is deterministic (a real bug, like
# 11.14's AttributeError) and further attempts would only burn hours reproducing it.
set -u
LABEL="$1"; ARCH="$2"; CKPT="$3"; ROUTES="$4"
SUBSET="${5:-}"; PORT="${6:-2000}"; TM_PORT="${7:-8000}"; MAX_ATTEMPTS="${8:-3}"

MTHESIS=/workspace/MThesis
OUT=/workspace/leaderboard_official_out
PY=/workspace/venv_carla/bin/python
GARAGE=/workspace/carla_garage
REAL_CARLA=/workspace/carla
mkdir -p "$OUT"

# merge_leaderboard_results.py imports leaderboard.utils.statistics_manager directly (it reuses
# the evaluator's own compute_global_statistics() rather than reimplementing it - see that
# file's docstring). run_leaderboard_official.sh exports this PYTHONPATH internally for the
# evaluator subprocess it launches, but that does not escape its own process, so this script
# needs the same export for its own merge step or the import fails after every attempt has
# already spent hours driving - the exact way this was first caught.
export PYTHONPATH="$MTHESIS:$REAL_CARLA/PythonAPI:$REAL_CARLA/PythonAPI/carla:$GARAGE/leaderboard:$GARAGE/scenario_runner${PYTHONPATH:+:$PYTHONPATH}"

CHECK="$MTHESIS/scripts/eval/check_leaderboard_results.py"
ATTEMPT_FILES=()
attempt=1
subset="$SUBSET"
prev_retry=""

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  alabel="${LABEL}_a${attempt}"
  afile="$OUT/${alabel}.json"
  echo "############ $LABEL attempt $attempt/$MAX_ATTEMPTS ############"
  if [ -n "$subset" ]; then
    echo "retrying routes: $subset"
  else
    echo "running the full route set"
  fi

  # Each attempt gets its own CARLA. run_leaderboard_official.sh starts one and the trap in it
  # cleans up, so a server left wedged by the previous attempt cannot poison this one.
  "$MTHESIS/scripts/eval/run_leaderboard_official.sh" \
      "$alabel" "$ARCH" "$CKPT" "$ROUTES" "$subset" "$PORT" "$TM_PORT"
  echo "--- attempt $attempt exited $? ---"

  if [ ! -f "$afile" ]; then
    echo "attempt $attempt produced no results file; stopping."
    break
  fi
  ATTEMPT_FILES+=("$afile")

  retry="$($PY "$CHECK" "$afile" --list-retryable)"
  if [ -z "$retry" ]; then
    echo "nothing left to retry after attempt $attempt."
    break
  fi

  # Same routes failing again means the cause is not transient; another pass buys nothing.
  if [ "$retry" = "$prev_retry" ]; then
    echo "attempt $attempt failed on exactly the same routes as the previous attempt ($retry)."
    echo "This is a deterministic failure, not a flake - stopping rather than retrying it."
    break
  fi

  prev_retry="$retry"
  subset="$retry"
  attempt=$((attempt + 1))
done

# Merge in attempt order so later, better records win. --total must be the size of the full
# route set, not of the last retry pass, or every mean is divided by too small a number.
TOTAL="$($PY - "$ROUTES" "$SUBSET" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
tree = ET.parse(sys.argv[1])
ids = [r.attrib["id"] for r in tree.iter("route")]
subset = sys.argv[2].strip() if len(sys.argv) > 2 else ""
if not subset:
    print(len(ids)); raise SystemExit
n = 0
for group in subset.replace(" ", "").split(","):
    if "-" in group:
        start, end = group.split("-")
        n += len(ids[ids.index(start):ids.index(end) + 1])
    else:
        n += 1
print(n)
PYEOF
)"

echo "############ merging ${#ATTEMPT_FILES[@]} attempt file(s), total routes = $TOTAL ############"
if [ "${#ATTEMPT_FILES[@]}" -eq 0 ]; then
  echo "no attempt produced results; nothing to merge."
  exit 2
fi

$PY "$MTHESIS/scripts/eval/merge_leaderboard_results.py" \
    --out "$OUT/${LABEL}.json" --total "$TOTAL" "${ATTEMPT_FILES[@]}"

echo "############ final validation of $OUT/${LABEL}.json ############"
$PY "$CHECK" "$OUT/${LABEL}.json" --expect-routes "$TOTAL"
exit $?
