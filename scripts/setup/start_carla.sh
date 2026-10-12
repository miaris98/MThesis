#!/usr/bin/env bash
# (Re)starts the CARLA server on one port and waits until it actually answers, rather than assuming
# a launched process is a ready one. Group 3's lesson: the server can die at any point,
# so every consumer needs a liveness check rather than a sleep.
#
# usage: start_carla.sh [port=2000]      env: LOG (default $WORKSPACE/carla_server.log), KILL_ALL_CARLA=1
#
# Only the server on this port is restarted. This used to `pkill -9 -f CarlaUE4`, which also killed every
# other lane's server on a multi-lane box; KILL_ALL_CARLA=1 keeps that behaviour for a deliberate reset.
# Space concurrent ports by at least 10: a server claims port..port+2 (RPC, streaming, secondary), and the
# client's --tm_port is a separate collision on the same symptom (run_closed_loop_arms.sh).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
PORT="${1:-2000}"
LOG="${LOG:-$WORKSPACE/carla_server.log}"
if [ "${KILL_ALL_CARLA:-0}" = 1 ]; then
  pkill -9 -f CarlaUE4 2>/dev/null
else
  # `ss -tlnp` cannot resolve a socket's pid in these containers, so match the command line via /proc
  clear_carla_ports "$PORT"
fi
sleep 3
su carlauser -c "$CARLA_DIR/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -vulkan -quality-level=Low" \
  > "$LOG" 2>&1 &
for i in $(seq 1 60); do
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "CARLA_LISTENING_ON_${PORT}_AFTER_${i}s"
    sleep 5   # listening != ready to serve a map query
    exit 0
  fi
  sleep 1
done
echo "CARLA_FAILED_TO_LISTEN_ON_${PORT}"
tail -20 "$LOG"
exit 1
