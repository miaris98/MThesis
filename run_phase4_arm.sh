#!/usr/bin/env bash
# Runs ONE Phase 4 arm: the 38-route validated-safe subset of Bench2Drive's published 220
# (challenges_03 3.12 - Town05/11/12/13 excluded; 3.13 - Town03 excluded after its 210MB
# BuiltData asset was found ~130-6000x larger than every other town's and confirmed to
# deterministically segfault CARLA's engine at world-load; routes 27494/28198/28048/26394
# excluded individually as deterministic per-route crashes, cause unidentified).
#
# Runs each route in its OWN evaluator invocation, with a full CarlaUE4 kill+fresh-restart
# before it - deliberately not one continuous multi-route session. Live evidence for why: route
# 24224 (Town02), which passed cleanly in the pre-flight probe (always a cold boot per route),
# crashed identically twice in a row when it followed 24211 (Town01) inside one continuous
# evaluator process - a live map *switch* crash, not a property of 24224 in isolation. Giving
# every route a fresh boot removes map-switching as a variable, confirmed: 24224 then passed on
# both retries once isolated.
#
# EACH ROUTE GETS ITS OWN --checkpoint FILE, merged into the arm-level JSON only at the end.
# This was discovered the hard way: the first version pointed every invocation's --checkpoint at
# one shared arm-level file with --resume=True, on the assumption that --resume=True accumulates
# across invocations the way it accumulates across a crash-restart of the SAME --routes-subset.
# It does not, when --routes-subset differs each call: Bench2Drive's checkpoint resume logic
# matches against the current invocation's subset, and a subset of one different route each time
# looks like a fresh 0-of-1-routes run to it - it silently reset progress=[0,1] and 0 records on
# every single route, discarding every previously-completed route's result. Caught only by
# checking the JSON directly mid-run (route count did not match the number of ATTEMPTED log
# lines already seen) rather than trusting the watchdog's own success reports - the same
# "verify against the artifact, not the process's reported status" lesson as 11.8/11.9, just
# hitting the checkpoint file instead of a bare exit code this time.
#
# Deliberately does NOT pre-launch CARLA via start_carla_multi.sh the way run_closed_loop_arms.sh
# does for eval_wor_closed_loop.py: leaderboard_evaluator.py manages its own CarlaUE4 process
# internally through the CARLA_ROOT=carla_shim trick (see run_bench2drive.sh/run_tcp.sh).
# Pre-launching a server on the same port collided with the evaluator's own spawn and silently
# rebound it to port+3 - tried once, made things worse, removed.
#
# usage: run_phase4_arm.sh <label> wor <arch> <ckpt> <port> <tm_port> [gpu_rank] [stagger]
#        run_phase4_arm.sh <label> tcp <ckpt> -  <port> <tm_port> [gpu_rank] [stagger]
set -x
LABEL="$1"; KIND="$2"; ARG3="$3"; ARG4="$4"; PORT="$5"; TM_PORT="$6"; GPU_RANK="${7:-0}"; STAGGER="${8:-0}"

MTHESIS=/workspace/MThesis
ROUTES=/workspace/Bench2Drive/leaderboard/data/bench2drive220.xml
IDS="24211,24224,24240,24258,24294,24330,24333,24340,24367,24757,24781,24784,24785,24795,24841,25300,25358,25383,25439,25845,25863,25951,25968,25975,26401,26405,26406,26408,26458,26944,27018,27529,28035,28087,28093,28099,28111,28154"
EXPECTED=38
OUT=/workspace/bench2drive_out
PERROUTE_DIR="${OUT}/${LABEL}_perroute"
mkdir -p "$PERROUTE_DIR"
PER_ROUTE_TIMEOUT=1500

is_done() {
  local rid="$1"
  local f="${PERROUTE_DIR}/${rid}.json"
  python3 -c "
import json
try:
    d = json.load(open('$f'))
    print(1 if len(d['_checkpoint']['records']) > 0 else 0)
except Exception:
    print(0)
"
}

sleep "$STAGGER"

IFS=',' read -ra ROUTE_ARR <<< "$IDS"
for rid in "${ROUTE_ARR[@]}"; do
  if [ "$(is_done "$rid")" = "1" ]; then
    echo "PHASE4_ROUTE_ALREADY_DONE label=$LABEL route=$rid"
    continue
  fi
  ROUTE_JSON="${PERROUTE_DIR}/${rid}.json"
  rm -f "$ROUTE_JSON"
  pkill -9 -f "carla-rpc-port=${PORT} " 2>/dev/null
  pkill -9 -f "leaderboard_evaluator.*--port=${PORT} " 2>/dev/null
  sleep 2
  if [ "$KIND" = "tcp" ]; then
    TCP_SKIP_SAVE=1 timeout "$PER_ROUTE_TIMEOUT" bash "$MTHESIS/run_tcp.sh" "${LABEL}_perroute/${rid}" "$ARG3" "$ROUTES" "$PORT" "$TM_PORT" "$GPU_RANK" "$rid" \
      >> "/workspace/phase4_${LABEL}.log" 2>&1
  else
    timeout "$PER_ROUTE_TIMEOUT" bash "$MTHESIS/run_bench2drive.sh" "${LABEL}_perroute/${rid}" "$ARG3" "$ARG4" "$ROUTES" "$PORT" "$TM_PORT" "$GPU_RANK" "$rid" \
      >> "/workspace/phase4_${LABEL}.log" 2>&1
  fi
  echo "PHASE4_ROUTE_ATTEMPTED label=$LABEL route=$rid done=$(is_done "$rid")"
done

pkill -9 -f "carla-rpc-port=${PORT} " 2>/dev/null
pkill -9 -f "leaderboard_evaluator.*--port=${PORT} " 2>/dev/null

# Merge every per-route checkpoint's single record into one arm-level JSON.
python3 -c "
import json, glob
records = []
for path in sorted(glob.glob('${PERROUTE_DIR}/*.json')):
    try:
        d = json.load(open(path))
        records.extend(d['_checkpoint']['records'])
    except Exception:
        pass
out = {'_checkpoint': {'records': records, 'progress': [len(records), $EXPECTED]}}
json.dump(out, open('${OUT}/${LABEL}.json', 'w'), indent=2)
print('MERGED', len(records), 'records into ${OUT}/${LABEL}.json')
"

final=$(python3 -c "
import json
try:
    d = json.load(open('${OUT}/${LABEL}.json'))
    print(len(d['_checkpoint']['records']))
except Exception:
    print(0)
")
echo "PHASE4_ARM_DONE label=$LABEL routes_recorded=${final}/${EXPECTED}"
