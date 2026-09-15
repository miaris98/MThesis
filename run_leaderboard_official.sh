#!/usr/bin/env bash
# Runs ONE arm against the *official* CARLA Leaderboard 2.0 route files, using the vanilla
# evaluator vendored in carla_garage rather than Bench2Drive's fork.
#
# usage: run_leaderboard_official.sh <label> <arch> <ckpt> <routes_xml> [routes_subset] [port] [tm_port]
#
# Differences from run_bench2drive.sh that are not cosmetic:
#
#   1. The vanilla evaluator never launches a simulator - it connects to one. Bench2Drive's
#      fork spawns CarlaUE4.sh itself, which is the only reason that script needs the
#      CARLA_ROOT privilege-drop shim. Here the server is started below instead, and the
#      shim is not needed at all.
#   2. No --gpu-rank. That flag exists only in Bench2Drive's fork; passing it to the vanilla
#      parser is an error.
#   3. IS_BENCH2DRIVE is deliberately unset. It is read by Bench2Drive's evaluator, never by
#      bench2drive_agent.py, so it means nothing here and setting it would only mislead.
#
# The agent file is shared with the Bench2Drive path unchanged: WorB2DAgent imports only
# leaderboard.autoagents.autonomous_agent and srunner.scenariomanager.carla_data_provider,
# both of which the vanilla checkout provides under the same module paths.
set -u
LABEL="$1"; ARCH="$2"; CKPT="$3"; ROUTES="$4"
ROUTES_SUBSET="${5:-}"; PORT="${6:-2000}"; TM_PORT="${7:-8000}"

GARAGE=/workspace/carla_garage
REAL_CARLA=/workspace/carla
MTHESIS=/workspace/MThesis
OUT=/workspace/leaderboard_official_out
mkdir -p "$OUT"

# The heads are useless without the vision weights they were trained on, and the loader only
# finds those as a sibling of the checkpoint (wor_loader.py:150). A checkpoint copied out of
# its directory still loads, still drives, and silently runs a different perception stack
# (11.10) - so refuse to start rather than produce a plausible wrong number.
if [ ! -f "$(dirname "$CKPT")/frozen_backbone.pth" ]; then
  echo "FATAL: no frozen_backbone.pth beside $CKPT - these heads were trained on a frozen"
  echo "backbone and would silently run on freshly initialized vision weights."
  exit 1
fi

export SCENARIO_RUNNER_ROOT="$GARAGE/scenario_runner"
export LEADERBOARD_ROOT="$GARAGE/leaderboard"
export MTHESIS_ROOT="$MTHESIS"

# Note what is NOT on this path: carla-0.9.15-py3.7-linux-x86_64.egg, which
# leaderboard/scripts/run_evaluation.sh would append. venv_carla is Python 3.10 with carla
# installed from pip; putting a py3.7 egg ahead of that shadows a working client with one
# whose compiled extension segfaults on first use rather than failing to import (11.7, 1.9).
export PYTHONPATH="$MTHESIS:$REAL_CARLA/PythonAPI:$REAL_CARLA/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 2026-09-15: a fresh box's venv_carla had carla/torch/timm but none of the vanilla
# leaderboard/scenario_runner runtime deps (six, py-trees, opencv, ...) - checking "torch and
# timm import" was not the same as checking "the evaluator itself imports", so both arms
# spent a full CARLA cold-start (60-100s) only to crash instantly on `from six import
# iteritems`. Import the real entry points *before* paying that cost, and if it fails, install
# the known dependency set and re-check once - never assume a fresh box already has these just
# because the training venv did (they are two separate venvs: /venv/main vs venv_carla).
cd "$MTHESIS"
PREFLIGHT_LOG=/tmp/eval_preflight_${LABEL}.log
preflight_ok() {
  /workspace/venv_carla/bin/python -c "
import leaderboard.leaderboard_evaluator
import bench2drive_agent
" >"$PREFLIGHT_LOG" 2>&1
}
echo "=== preflight: verifying venv_carla can import the evaluator + agent ==="
if ! preflight_ok; then
  echo "preflight failed - installing known eval dependencies into venv_carla (see $PREFLIGHT_LOG)"
  /workspace/venv_carla/bin/pip install -q six "py-trees==0.8.3" Shapely xmlschema ephem \
    tabulate opencv-python matplotlib psutil pygame pexpect dictor transforms3d \
    simple-watchdog-timer requests
  if ! preflight_ok; then
    echo "FATAL: venv_carla still cannot import the evaluator/agent after installing the known"
    echo "dependency set - a NEW missing module, not one of the ones this script already knows"
    echo "about. See $PREFLIGHT_LOG:"
    cat "$PREFLIGHT_LOG"
    exit 1
  fi
  echo "preflight fixed by installing the known dependency set"
fi
echo "=== preflight OK ==="

echo "=== starting CARLA on port $PORT ==="
# Keyed by LABEL, not a fixed path: two arms launched concurrently (different ports, same
# carlauser account) would otherwise share one XDG_RUNTIME_DIR and corrupt each other's
# Vulkan ICD/session state - this only surfaced once someone tried running two arms at once.
export XDG_RUNTIME_DIR="/tmp/runtime-carlauser-${LABEL}"
mkdir -p "$XDG_RUNTIME_DIR"
chown carlauser:carlauser "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
# CARLA refuses to run as root in this container (1.1/1.2), hence su carlauser everywhere.
setsid su carlauser -c "ulimit -n 65536 2>/dev/null; export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR; \
  $REAL_CARLA/CarlaUE4.sh -carla-rpc-port=$PORT -carla-streaming-port=0 \
  -RenderOffScreen -nosound -vulkan -quality-level=Low" \
  </dev/null >"$OUT/carla_${LABEL}.log" 2>&1 &

# This script starts a simulator, so it owns killing it on EVERY exit path - including the
# `exit 1` below when the port never opens, and including an OOM that takes out the evaluator.
# It used to trap only the results-watchdog, and on 2026-09-15 that leaked a CarlaUE4 server
# holding 5.7GB of VRAM for over an hour after its arm died. Worse, eval_watchdog.sh was at the
# same time refusing to relaunch that very arm because free VRAM was below MIN_FREE_MIB - the
# arm was blocked by its own leaked server. A resource gate needs a matching reaper.
#
# Matching on the comm *prefix* via /proc is deliberate, and neither obvious alternative works:
#   - `pkill -f "carla-rpc-port=$PORT"` matches any command line containing that string,
#     including the shell running this cleanup, and kills the caller.
#   - `pgrep -x CarlaUE4-Linux-Shipping` never matches: the kernel truncates /proc/PID/comm to
#     15 chars, so the name is "CarlaUE4-Linux-" and the exact match silently finds nothing.
# A shell's comm is bash/sh, never CarlaUE4*, so this cannot match the caller. The trailing
# space in the port test stops port 2000 matching 20000.
WATCHDOG=""
cleanup() {
  [ -n "$WATCHDOG" ] && kill "$WATCHDOG" 2>/dev/null
  local pid cmd
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ -r "/proc/$pid/comm" ] || continue
    case "$(cat "/proc/$pid/comm" 2>/dev/null)" in CarlaUE4*) ;; *) continue ;; esac
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) "
    case "$cmd" in *"carla-rpc-port=$PORT "*) kill -9 "$pid" 2>/dev/null ;; esac
  done
}
trap cleanup EXIT INT TERM

# Wait on the port actually accepting connections rather than a fixed sleep: cold-starting
# this server takes 60-100s here, and the probe that concluded "this box cannot run CARLA"
# used a 60s ceiling that it sometimes simply did not clear.
echo "waiting for RPC port $PORT ..."
for i in $(seq 1 90); do
  if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "port $PORT up after ${i}0s"
    break
  fi
  sleep 10
done
if ! (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "FATAL: CARLA never opened port $PORT - see $OUT/carla_${LABEL}.log"
  exit 1
fi
sleep 15  # the port opens slightly before the server will answer RPCs

export CHALLENGE_TRACK_CODENAME=SENSORS
export SAVE_PATH="$OUT/$LABEL/"
mkdir -p "$SAVE_PATH"

EXTRA_ARGS=()
if [ -n "$ROUTES_SUBSET" ]; then
  EXTRA_ARGS+=(--routes-subset="$ROUTES_SUBSET")
fi

echo "=== evaluating $LABEL: $ARCH @ $CKPT on $(basename "$ROUTES") ==="

# A crashed agent is not an error to this evaluator: it catches AgentError, tags the route
# "Agent crashed", scores it 0 and carries on, and compute_global_statistics() does not
# escalate that status the way it escalates 'Simulation crashed'. So a completely broken agent
# yields entry_status "Finished" and a tidy mean of zeros (11.14). Watch the live results file
# and kill the run once it is clear nothing is driving, rather than spending the remaining
# ~110 routes discovering it. It fires only if EVERY route so far failed agent-side, so a
# genuinely poor policy - which still produces driving statuses - will never trip it.
RESULTS="$OUT/${LABEL}.json"
(
  while sleep 60; do
    /workspace/venv_carla/bin/python "$MTHESIS/check_leaderboard_results.py" \
        "$RESULTS" --abort-after 3 --quiet
    if [ $? -eq 3 ]; then
      echo "=== WATCHDOG: aborting $LABEL, see above ==="
      pkill -TERM -f "leaderboard_evaluator.py.*${LABEL}.json"
      exit 0
    fi
  done
) &
WATCHDOG=$!
# The EXIT/INT/TERM trap installed right after the CARLA launch already reaps both this
# watchdog and the simulator, so it is deliberately not re-installed (and not narrowed) here.

/workspace/venv_carla/bin/python "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py" \
  --routes="$ROUTES" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="$OUT/${LABEL}.json" \
  --debug-checkpoint="$OUT/${LABEL}_live.txt" \
  --agent="$MTHESIS/bench2drive_agent.py" \
  --agent-config="${ARCH}:${CKPT}" \
  --debug=0 \
  --resume=True \
  --timeout=600 \
  --port="$PORT" \
  --traffic-manager-port="$TM_PORT" \
  "${EXTRA_ARGS[@]}"
STATUS=$?
kill $WATCHDOG 2>/dev/null

# Exiting 0 here would mean "the evaluator ran", which is not the same as "these numbers mean
# something". Gate the result on every recorded route being a real driving outcome, so a
# contaminated file cannot be quietly picked up and reported as a score.
echo "=== $LABEL finished with status $STATUS; validating results ==="
/workspace/venv_carla/bin/python "$MTHESIS/check_leaderboard_results.py" "$OUT/${LABEL}.json"
VALID=$?
if [ $VALID -ne 0 ]; then
  echo "=== $LABEL: results at $OUT/${LABEL}.json are NOT a usable measurement (see above) ==="
  exit 2
fi

echo "=== $LABEL finished with status $STATUS; results: $OUT/${LABEL}.json ==="
exit $STATUS
