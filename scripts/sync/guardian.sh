#!/usr/bin/env bash
# usage: guardian.sh <label> <ckpt_json> <total_routes> <cmd...>
# total_routes must match whatever --routes-subset resolves to for this run - a mismatch here
# reproduces the infinite relaunch-and-immediately-exit loop documented in struggle-solutions.md
# S-021/challenges_03 3.15 (guardian's completion check never matching a trimmed subset's total).
LABEL="$1"; CKPT_JSON="$2"; TOTAL_ROUTES="$3"; shift 3
LOGDIR=/workspace/guardian_logs
mkdir -p "$LOGDIR"
ATTEMPT=0
while true; do
  PROGRESS=$(python3 -c "
import json
try:
    d = json.load(open('$CKPT_JSON'))
    print(d.get('_checkpoint', {}).get('progress', [0,$TOTAL_ROUTES])[0])
except Exception:
    print(0)
" 2>/dev/null)
  if [ "${PROGRESS:-0}" -ge "$TOTAL_ROUTES" ] 2>/dev/null; then
    echo "$(date) [$LABEL] progress=$PROGRESS/$TOTAL_ROUTES - DONE, guardian exiting" >> "$LOGDIR/${LABEL}.log"
    break
  fi
  if ! pgrep -f "checkpoint=$CKPT_JSON" >/dev/null 2>&1; then
    ATTEMPT=$((ATTEMPT+1))
    echo "$(date) [$LABEL] dead, progress=${PROGRESS:-0}/$TOTAL_ROUTES, relaunch attempt=$ATTEMPT" >> "$LOGDIR/${LABEL}.log"
    nohup "$@" > "$LOGDIR/${LABEL}_attempt${ATTEMPT}.log" 2>&1 &
    sleep 60
  fi
  sleep 15
done
