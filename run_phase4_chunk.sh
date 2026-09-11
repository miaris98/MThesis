#!/usr/bin/env bash
# Runs a caller-specified subset of an arm's Phase 4 routes against a SHARED
# per-route output directory (same LABEL_perroute dir as run_phase4_arm.sh uses),
# with no merge/PHASE4_ARM_DONE step of its own. Built to parallelize TCP's
# remaining routes across the GPU/CPU capacity freed up once cnn and qwen
# finished: launch several of these at once on different ports, each given a
# disjoint route-id chunk, all writing into one shared perroute dir (no file
# collision since route ids don't overlap across chunks) - see
# run_phase4_tcp_parallel.sh, which launches N of these and does the one real
# merge + PHASE4_ARM_DONE echo only after all of them finish.
#
# usage: run_phase4_chunk.sh <label> wor <arch> <ckpt> <port> <tm_port> <gpu_rank> <ids_csv>
#        run_phase4_chunk.sh <label> tcp <ckpt> -      <port> <tm_port> <gpu_rank> <ids_csv>
set -x
LABEL="$1"; KIND="$2"; ARG3="$3"; ARG4="$4"; PORT="$5"; TM_PORT="$6"; GPU_RANK="$7"; IDS_CSV="$8"

MTHESIS=/workspace/MThesis
ROUTES=/workspace/Bench2Drive/leaderboard/data/bench2drive220.xml
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

IFS=',' read -ra ROUTE_ARR <<< "$IDS_CSV"
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
      >> "/workspace/phase4_${LABEL}_chunk_${PORT}.log" 2>&1
  else
    timeout "$PER_ROUTE_TIMEOUT" bash "$MTHESIS/run_bench2drive.sh" "${LABEL}_perroute/${rid}" "$ARG3" "$ARG4" "$ROUTES" "$PORT" "$TM_PORT" "$GPU_RANK" "$rid" \
      >> "/workspace/phase4_${LABEL}_chunk_${PORT}.log" 2>&1
  fi
  echo "PHASE4_ROUTE_ATTEMPTED label=$LABEL route=$rid done=$(is_done "$rid")"
done

pkill -9 -f "carla-rpc-port=${PORT} " 2>/dev/null
pkill -9 -f "leaderboard_evaluator.*--port=${PORT} " 2>/dev/null
echo "PHASE4_CHUNK_DONE label=$LABEL port=$PORT"
