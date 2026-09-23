#!/usr/bin/env bash
# Keeps one Bench2Drive arm alive until its checkpoint JSON reports all routes done.
#
# usage: scripts/eval/b2d_guardian.sh <label> <checkpoint_json> <total_routes> <cmd...>
#
# The evaluator runs with --resume=True, so relaunching after a CARLA crash/hang continues
# from the last finished route instead of restarting. Adapted from the guardian.sh used for the
# 2026-09-16 b2d20 baseline run (E:\MThesis_EXP\bench2drive_eval\run_20260916_final), with two
# changes: the route total is a parameter (that copy hard-coded 20, which never terminates for an
# arm that only runs a subset), and attempts are capped so a route that deterministically kills
# the server cannot relaunch forever.
LABEL="$1"; CKPT_JSON="$2"; TOTAL="$3"; shift 3
MAX_ATTEMPTS="${MAX_ATTEMPTS:-25}"
LOGDIR=/workspace/guardian_logs
mkdir -p "$LOGDIR"
ATTEMPT=0
while true; do
  PROGRESS=$(python3 -c "
import json
try:
    d = json.load(open('$CKPT_JSON'))
    print(d.get('_checkpoint', {}).get('progress', [0, $TOTAL])[0])
except Exception:
    print(0)
" 2>/dev/null)
  if [ "${PROGRESS:-0}" -ge "$TOTAL" ] 2>/dev/null; then
    echo "$(date) [$LABEL] progress=$PROGRESS/$TOTAL - DONE" >> "$LOGDIR/${LABEL}.log"
    break
  fi
  if ! pgrep -f "checkpoint=$CKPT_JSON" >/dev/null 2>&1; then
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
      echo "$(date) [$LABEL] giving up after $ATTEMPT attempts at progress=${PROGRESS:-0}/$TOTAL" >> "$LOGDIR/${LABEL}.log"
      exit 1
    fi
    ATTEMPT=$((ATTEMPT+1))
    echo "$(date) [$LABEL] not running, progress=${PROGRESS:-0}/$TOTAL, launch attempt=$ATTEMPT" >> "$LOGDIR/${LABEL}.log"
    nohup "$@" > "$LOGDIR/${LABEL}_attempt${ATTEMPT}.log" 2>&1 &
    sleep 60
  fi
  sleep 15
done
