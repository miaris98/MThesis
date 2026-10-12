#!/usr/bin/env bash
# Standalone diagnostic: does THIS instance's CARLA 0.9.15 install load Town11/12/13 cleanly?
#
# Reuses the exact method challenges_03 3.9 used to first establish these towns' failure
# mechanisms (Town11: server-side Signal 11 segfault at map load; Town12/Town13: silent hang,
# "Operation canceled" / exit 143, server process alive but map switch never completes) --
# a fresh CarlaUE4 server per town, torch-free, load_world() + a few ticks, isolated so one
# town's crash can't hide whether the next town would have worked.
#
# Why re-run this on every new rental rather than trust the old finding: challenges_03 3.10
# found town reliability does NOT transfer between rented instances -- a town that hangs or
# segfaults on one box is not guaranteed to do either on a different one, even similar hardware.
# This probe is the cheap way to find out before committing to a download or a real run.
#
# usage: scripts/eval/run_town_probe.sh [town1,town2,...]   (default: Town11,Town12,Town13)
set -x
TOWNS="${1:-Town11,Town12,Town13}"
REAL_CARLA=/workspace/carla
SHIM=/workspace/carla_shim
PORT=3000
TM_PORT=9000
PER_TOWN_TIMEOUT=150
RESULTS=/workspace/town_probe_results.txt

mkdir -p "$SHIM"
# CARLA/UE4 refuses to run as root (hence su carlauser) and separately requires a valid,
# carlauser-owned XDG_RUNTIME_DIR or it logs "XDG_RUNTIME_DIR is invalid or not set" and the
# RPC server never comes up - this bit run_leaderboard_official.sh once already (it sets this
# up explicitly) and the first version of this probe script omitted it, producing a false
# "UNKNOWN rc=1" for Town11/Town12 that was actually this setup bug, not a map-load failure.
export XDG_RUNTIME_DIR=/tmp/runtime-carlauser-probe
mkdir -p "$XDG_RUNTIME_DIR"
chown carlauser:carlauser "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
# -RenderOffScreen -nosound are not optional extras - run_leaderboard_official.sh always
# passes them, and this shim originally didn't, which stalled load_world() for 60s+ on every
# town (UE4 trying to create a display context with no X server present) and looked exactly
# like the "silent hang" failure mode this probe exists to detect. Match the real invocation
# byte-for-byte rather than approximate it, so a probe pass actually predicts the real run.
cat > "$SHIM/CarlaUE4.sh" <<SHIMEOF
#!/usr/bin/env bash
exec su carlauser -c "export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR; /workspace/carla/CarlaUE4.sh \$* -RenderOffScreen -nosound -vulkan -quality-level=Low"
SHIMEOF
chmod +x "$SHIM/CarlaUE4.sh"

> "$RESULTS"

check_one_town() {
  local town="$1"
  local log="/workspace/town_probe_${town}.log"

  pkill -9 -f "carla-rpc-port=${PORT} " 2>/dev/null
  sleep 2
  "$SHIM/CarlaUE4.sh" -carla-rpc-port="$PORT" -carla-streaming-port=0 > "$log" 2>&1 &
  SERVER_PID=$!

  # Wait for the RPC port to actually accept connections rather than a fixed sleep --
  # the whole point is to isolate map-load behaviour, not startup-time variance.
  for i in $(seq 1 30); do
    /workspace/venv_carla/bin/python -c "
import carla
c = carla.Client('localhost', $PORT)
c.set_timeout(2.0)
c.get_server_version()
" >/dev/null 2>&1 && break
    sleep 2
  done

  timeout "$PER_TOWN_TIMEOUT" /workspace/venv_carla/bin/python -c "
import carla
c = carla.Client('localhost', $PORT)
c.set_timeout(120.0)
world = c.load_world('$town')
for _ in range(5):
    world.tick()
print('PROBE_TOWN_OK')
" >> "$log" 2>&1
  RC=$?

  pkill -9 -f "carla-rpc-port=${PORT} " 2>/dev/null
  sleep 2

  if grep -q "PROBE_TOWN_OK" "$log"; then
    echo "${town} OK" | tee -a "$RESULTS"
  elif grep -qi "Segmentation fault\|Signal 11" "$log"; then
    echo "${town} SEGFAULT" | tee -a "$RESULTS"
  elif [ "$RC" -eq 124 ]; then
    echo "${town} HANG (timeout)" | tee -a "$RESULTS"
  else
    echo "${town} UNKNOWN rc=$RC (see $log)" | tee -a "$RESULTS"
  fi
}

IFS=',' read -ra TOWN_ARR <<< "$TOWNS"
for t in "${TOWN_ARR[@]}"; do
  check_one_town "$t"
done

echo "TOWN_PROBE_ALL_DONE"
cat "$RESULTS"
