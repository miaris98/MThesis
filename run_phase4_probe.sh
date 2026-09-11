#!/usr/bin/env bash
# Pre-flight probe: tests each of the 51 town-filtered "safe" Bench2Drive routes (challenges_03
# 3.12) in its own isolated leaderboard_evaluator.py invocation, using a single fast arm (cnn,
# near-real-time per 3.11/3.12's timing data) rather than all three concurrently.
#
# Why per-route isolation: a hard engine crash (Signal 11) kills the whole evaluator process
# outright rather than skipping to the next route, discovered live when route 24206 - previously
# untested, on a town never flagged bad - crashed all three Phase 4 arms independently at world
# load. A single invocation covering all 51 routes would have silently stopped probing at the
# first bad one. This only detects the load-time hard-crash failure class (27494's and 24206's
# shape); it will not catch 28198's kind of failure (agent-dependent low score, no crash) since
# that requires actually driving the route with a specific policy, which is what the real 3-arm
# run is for - this probe's only job is to keep the real run from tripping over another landmine
# like 24206 mid-flight.
set -x
MTHESIS=/workspace/MThesis
CKPT=/workspace/checkpoints/wor_cl_arms/cnn_s0/best_model.pth
ROUTES=/workspace/Bench2Drive/leaderboard/data/bench2drive220.xml
PORT=2000
TM_PORT=8000
IDS="24206,24211,24224,24240,24258,24294,24330,24333,24340,24367,24757,24781,24784,24785,24795,24816,24841,25300,25358,25378,25383,25439,25845,25854,25863,25896,25951,25968,25975,26393,26394,26401,26405,26406,26408,26435,26456,26458,26944,26950,26990,27018,27515,27529,28035,28048,28087,28093,28099,28111,28154"

RESULTS=/workspace/phase4_probe_results.txt
> "$RESULTS"

IFS=',' read -ra ROUTE_ARR <<< "$IDS"
for rid in "${ROUTE_ARR[@]}"; do
  rm -f /workspace/bench2drive_out/probe.json
  rm -rf /workspace/bench2drive_out/probe
  pkill -9 -f CarlaUE4 2>/dev/null
  pkill -9 -f leaderboard_evaluator 2>/dev/null
  sleep 2
  LOG="/workspace/phase4_probe_${rid}.log"
  timeout 90 bash "$MTHESIS/run_bench2drive.sh" probe cnn "$CKPT" "$ROUTES" "$PORT" "$TM_PORT" 0 "$rid" > "$LOG" 2>&1
  if grep -q "Segmentation fault" "$LOG"; then
    echo "${rid} CRASH" >> "$RESULTS"
  elif grep -q "Wallclock" "$LOG"; then
    echo "${rid} OK" >> "$RESULTS"
  else
    echo "${rid} UNKNOWN" >> "$RESULTS"
  fi
  pkill -9 -f leaderboard_evaluator 2>/dev/null
  pkill -9 -f CarlaUE4 2>/dev/null
  sleep 2
done
echo PROBE_ALL_DONE
