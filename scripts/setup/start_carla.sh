#!/usr/bin/env bash
# (Re)starts the CARLA server and waits until it actually answers, rather than assuming
# a launched process is a ready one. Group 3's lesson: the server can die at any point,
# so every consumer needs a liveness check rather than a sleep.
PORT="${1:-2000}"
pkill -9 -f CarlaUE4 2>/dev/null
sleep 3
su carlauser -c "/workspace/carla/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -vulkan -quality-level=Low" \
  > /workspace/carla_server.log 2>&1 &
for i in $(seq 1 60); do
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "CARLA_LISTENING_AFTER_${i}s"
    sleep 5   # listening != ready to serve a map query
    exit 0
  fi
  sleep 1
done
echo "CARLA_FAILED_TO_LISTEN"
tail -20 /workspace/carla_server.log
exit 1
