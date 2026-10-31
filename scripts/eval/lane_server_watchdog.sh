#!/usr/bin/env bash
# Box side (2026-10-28, A69): a Bench2Drive lane can hang with its leaderboard evaluator alive and its CARLA server gone (seen twice on
# the TF++ 220-route lanes). b2d_guardian.sh only relaunches when the evaluator exits, so this kills an evaluator whose server
# (-carla-rpc-port=PORT) has been missing on two checks in a row; the guardian then resumes the lane.
# usage: setsid nohup bash scripts/eval/lane_server_watchdog.sh LABEL:PORT [LABEL:PORT ...] > /workspace/a69/watchdog.log 2>&1 &
# PITFALL (2026-10-28 evening): a lane's server came up on PORT+1 (2101 for lane 2100) after a crash relaunch; this script then killed a
# healthy evaluator every 4 min and used up the guardian's 25 attempts. Do not run it unattended until it matches the server by the
# evaluator's own child process (or by PORT..PORT+2) instead of the exact port.
miss=()
while true; do
  i=0
  for spec in "$@"; do
    label=${spec%%:*}; port=${spec##*:}
    ev=$(ps -eo pid,args | grep "[l]eaderboard_evaluator" | grep -- "--port=$port" | awk '{print $1}' | head -1)
    srv=$(ps -eo args | grep "[C]arlaUE4-Linux" | grep -c -- "-carla-rpc-port=$port\b")
    if [ -n "$ev" ] && [ "$srv" -eq 0 ]; then
      miss[$i]=$(( ${miss[$i]:-0} + 1 ))
      if [ "${miss[$i]}" -ge 2 ]; then
        echo "$(date -u +%H:%M:%S) $label: evaluator $ev has had no CARLA server on port $port for 2 checks, killing it (guardian relaunches)"
        kill "$ev"; miss[$i]=0
      fi
    else
      miss[$i]=0
    fi
    i=$((i + 1))
  done
  sleep 120
done
