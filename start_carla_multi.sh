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
# `ss -tlnp` cannot resolve a listening socket's pid in this container (its `pid=` field
# is simply absent from the output - confirmed, not assumed: `fuser -n tcp` and `lsof -i`
# both fail to find the process too). The kill below silently never fired as a result, so
# every restart landed on a server already running from a prior invocation instead of a
# fresh one - accidentally harmless here, because eval_wor_closed_loop.py's traffic is
# reseeded per-run regardless of server freshness (spawn_traffic()'s
# set_random_device_seed call), but the new server process still launched, still crashed
# on the port bind, and still cost a full engine-init cycle for nothing. `pgrep -af` shows
# cmdline matching works fine in this container (it does not depend on socket-to-pid
# resolution), so match on the command line instead of the port's socket.
pkill -f "carla-port=${PORT} " 2>/dev/null || true
sleep 2
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
