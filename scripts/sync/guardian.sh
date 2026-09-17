#!/usr/bin/env bash
LABEL="$1"; CKPT_JSON="$2"; shift 2
LOGDIR=/workspace/guardian_logs
mkdir -p "$LOGDIR"
ATTEMPT=0
while true; do
  PROGRESS=$(python3 -c "
import json
try:
    d = json.load(open('$CKPT_JSON'))
    print(d.get('_checkpoint', {}).get('progress', [0,20])[0])
except Exception:
    print(0)
" 2>/dev/null)
  if [ "${PROGRESS:-0}" -ge 20 ] 2>/dev/null; then
    echo "$(date) [$LABEL] progress=$PROGRESS/20 - DONE, guardian exiting" >> "$LOGDIR/${LABEL}.log"
    break
  fi
  if ! pgrep -f "checkpoint=$CKPT_JSON" >/dev/null 2>&1; then
    ATTEMPT=$((ATTEMPT+1))
    echo "$(date) [$LABEL] dead, progress=${PROGRESS:-0}/20, relaunch attempt=$ATTEMPT" >> "$LOGDIR/${LABEL}.log"
    nohup "$@" > "$LOGDIR/${LABEL}_attempt${ATTEMPT}.log" 2>&1 &
    sleep 60
  fi
  sleep 15
done
