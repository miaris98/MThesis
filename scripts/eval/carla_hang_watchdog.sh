#!/usr/bin/env bash
# Kills leaderboard clients whose CARLA server has died, so b2d_guardian.sh relaunches the arm.
#
# usage: nohup scripts/eval/carla_hang_watchdog.sh [grace_s=90] [poll_s=30] &
#
# Why: when CarlaUE4 segfaults while loading the next route's map (signal 11, seen on every box so
# far), leaderboard_evaluator.py keeps waiting on its RPC port for client_timeout = 600 s, and on
# 2026-09-24 one client never gave up at all - the guardian only relaunches after the client exits,
# so a GPU sat idle for 25+ min twice. A client is "orphaned" when no CarlaUE4 process serves its
# --port for longer than the grace period.
GRACE="${1:-90}"; POLL="${2:-30}"
declare -A since
while true; do
  now=$(date +%s)
  for pid in $(pgrep -f "leaderboard_evaluator.py"); do
    port=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -oE -- "--port[= ][0-9]+" | grep -oE "[0-9]+$")
    [ -z "$port" ] && continue
    if pgrep -f "CarlaUE4-Linux-Shipping.*-carla-rpc-port=$port( |$)" > /dev/null; then
      unset "since[$pid]"
    else
      since[$pid]=${since[$pid]:-$now}
      if (( now - since[$pid] > GRACE )); then
        echo "$(date '+%F %T') client pid $pid lost its CARLA server on port $port for >${GRACE}s - killing it"
        kill "$pid"; unset "since[$pid]"
      fi
    fi
  done
  sleep "$POLL"
done
