#!/usr/bin/env bash
# Like start_carla.sh but does not pkill every CarlaUE4 process - only restarts the
# server on the given port, so multiple instances (one per port) can coexist for
# concurrent closed-loop arms. Only ever kill by exact port match.
#
# Space ports by at least 10 when calling this for multiple concurrent servers: each
# CARLA server actually claims a 3-port range (port, port+1, port+2 for RPC, streaming,
# and a secondary channel), so anything closer than 10 apart risks collision. See
# run_closed_loop_arms.sh for the matching --tm_port requirement on the client side,
# which is a separate, easier-to-miss port collision on the SAME symptom.
PORT="${1:-2000}"
EXISTING_PID=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$EXISTING_PID" ]; then
  kill -9 "$EXISTING_PID" 2>/dev/null
  sleep 2
fi
su carlauser -c "/workspace/carla/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -vulkan -quality-level=Low" \
  > "/workspace/carla_server_${PORT}.log" 2>&1 &
for i in $(seq 1 60); do
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "CARLA_LISTENING_ON_${PORT}_AFTER_${i}s"
    sleep 5
    exit 0
  fi
  sleep 1
done
echo "CARLA_FAILED_TO_LISTEN_ON_${PORT}"
tail -20 "/workspace/carla_server_${PORT}.log"
exit 1
