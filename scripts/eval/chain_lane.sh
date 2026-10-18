#!/usr/bin/env bash
# Chain extra CARLA jobs behind a lane (2026-10-18): wait until the lane's queue_A9c.sh on PORT has ended, then start it again with the
# extra jobs (CKE/CKL syntax of queue_A9c.sh). Starts nothing after CUTOFF (UTC HHMM), so a variant can still finish before the final save.
# usage: chain_lane.sh PORT CUTOFF_HHMM_UTC JOB [JOB ...]       (run it detached)
cd /workspace/MThesis
PORT=$1; CUTOFF=$2; shift 2
while ps -eo args | grep -q "[q]ueue_A9c.sh $PORT "; do sleep 60; done
if [ "$(date -u +%H%M)" -ge "$CUTOFF" ]; then echo "$(date -u) lane $PORT ended after the cutoff, nothing chained" >> /workspace/q/carla_chain_$PORT.out; exit 0; fi
echo "$(date -u) lane $PORT ended, chaining: $*" >> /workspace/q/carla_chain_$PORT.out
setsid nohup bash scripts/eval/queue_A9c.sh $PORT 0 "$@" >> /workspace/q/carla_chain_$PORT.out 2>&1 < /dev/null &
