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
# the server cannot relaunch forever. scripts/sync/guardian.sh now delegates here.
# total_routes must match whatever --routes-subset resolves to for this run - a mismatch reproduces
# the infinite relaunch-and-immediately-exit loop of S-021 / challenges_03 3.15.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
LABEL="$1"; CKPT_JSON="$2"; TOTAL="$3"; shift 3
MAX_ATTEMPTS="${MAX_ATTEMPTS:-25}"
mkdir -p "$GUARDIAN_LOGS"
ATTEMPT=0
while true; do
  PROGRESS=$(b2d_progress "$CKPT_JSON")
  if [ "${PROGRESS:-0}" -ge "$TOTAL" ] 2>/dev/null; then
    echo "$(date) [$LABEL] progress=$PROGRESS/$TOTAL - DONE" >> "$GUARDIAN_LOGS/${LABEL}.log"
    break
  fi
  # "checkpoint=" + the json path matches only the evaluator, never this guardian (its argv has the bare path, S-019)
  if ! pgrep -f "checkpoint=$CKPT_JSON" >/dev/null 2>&1; then
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
      echo "$(date) [$LABEL] giving up after $ATTEMPT attempts at progress=${PROGRESS:-0}/$TOTAL" >> "$GUARDIAN_LOGS/${LABEL}.log"
      exit 1
    fi
    ATTEMPT=$((ATTEMPT+1))
    echo "$(date) [$LABEL] not running, progress=${PROGRESS:-0}/$TOTAL, launch attempt=$ATTEMPT" >> "$GUARDIAN_LOGS/${LABEL}.log"
    nohup "$@" > "$GUARDIAN_LOGS/${LABEL}_attempt${ATTEMPT}.log" 2>&1 &
    sleep 60
  fi
  sleep 15
done
