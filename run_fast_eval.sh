#!/usr/bin/env bash
# Fast closed-loop driving score - the iteration-speed tier that sits under the official
# Leaderboard 2.0 run (run_leaderboard_official.sh).
#
# usage: run_fast_eval.sh <label> <arch> <ckpt> [preset] [town] [port]
#          preset: smoke | hp | wide      (default: hp)
#          town:   Town01/02/03/05/10HD   (default: Town01)
#
# WHY THIS EXISTS
# ---------------
# On 2026-09-15 the official evaluator was measured at a 0.216x sim-to-wall ratio on
# routes_validation.xml, which is 20 routes x ~9.2 km x ~89 scenarios, all in Town13 - CARLA's
# largest map. That is ~50 h for one arm, which is a perfectly good *final* number and a useless
# *fitness function*: you cannot search hyperparameters against a signal that takes two days per
# sample. The cost is not the model (the GPU sits at ~40 W of 140 W and 7-30% utilisation with
# two arms running) - it is Town13's map size and the ~89 scenario behaviour-trees per route
# ticking in lock-step with the agent.
#
# This script keeps the part that carries the signal - closed-loop driving in synchronous mode at
# the same 20 Hz, scored with the same driving_score = route_completion x infraction_penalty as
# the leaderboard (src/eval/driving_metrics.py) - and drops the part that only costs wall-clock:
# the huge map and the scenario swarm. Same metric family, ~100x cheaper.
#
# WHAT IT IS NOT
# --------------
# Not a substitute for the official number, and must never be reported as one. There are no
# scenario_runner scenarios here, so the infraction mix is different and the absolute score will
# not match routes_validation.xml. Use it to *rank* configurations, then confirm the top one or
# two with run_leaderboard_official.sh.
set -u

LABEL="${1:?usage: run_fast_eval.sh <label> <arch> <ckpt> [preset] [town] [port]}"
ARCH="${2:?missing arch (cnn|qwen30m)}"
CKPT="${3:?missing checkpoint path}"
PRESET="${4:-hp}"
TOWN="${5:-Town01}"
# Default 2030, deliberately outside the 2000/2010/2020 block that eval_watchdog.sh manages for
# the official arms. Sharing a port with a watchdog-managed arm means the watchdog can start a
# CARLA server underneath this script at any moment; that happened, and this script then
# "reused" a server that was still cold-booting and timed out 60 s later.
PORT="${6:-2030}"

MTHESIS=/workspace/MThesis
REAL_CARLA=/workspace/carla
GARAGE=/workspace/carla_garage
OUT=/workspace/fast_eval_out
mkdir -p "$OUT"

# 2026-09-15: same pluggable-agent mechanism as run_leaderboard_official.sh's EVAL_AGENT /
# EVAL_AGENT_CONFIG / EVAL_PYTHON, added when eval_wor_closed_loop.py was refactored onto the
# real Leaderboard AgentWrapper/SensorInterface contract instead of a hand-rolled camera+dict.
# Default: our own bench2drive_agent.py, config "<arch>:<ckpt>" - unchanged behaviour for every
# existing caller. Point EVAL_AGENT at carla_garage/team_code/sensor_agent.py (with
# EVAL_PYTHON=/workspace/venv_tfpp/bin/python, EVAL_AGENT_CONFIG=<pretrained_models dir>) to run
# TF++ on these same cheap routes instead - the whole reason Tier 1 needed this refactor.
EVAL_AGENT="${EVAL_AGENT:-}"
EVAL_AGENT_CONFIG="${EVAL_AGENT_CONFIG:-}"
EVAL_PYTHON="${EVAL_PYTHON:-/workspace/venv_carla/bin/python}"
PY="$EVAL_PYTHON"
if [ ! -x "$PY" ]; then
  echo "FATAL: EVAL_PYTHON=$PY is not executable."
  exit 1
fi

# The frozen-backbone guard only means something for our own default agent - WorB2DAgent's
# loader finds those weights as a sibling of the checkpoint (wor_loader.py:150), and a
# checkpoint copied out of its directory silently runs different vision weights. TF++ (or any
# other --agent) has no such file and no such loader, so the check would be a false FATAL for
# every non-default agent - skip it whenever EVAL_AGENT overrides the default.
if [ -z "$EVAL_AGENT" ] && [ ! -f "$(dirname "$CKPT")/frozen_backbone.pth" ]; then
  echo "FATAL: no frozen_backbone.pth beside $CKPT"; exit 1
fi

# route_seed is FIXED, not derived from the label or the clock. Every trial in a hyperparameter
# search has to drive the identical routes with the identical traffic, or the search is ranking
# route luck instead of checkpoints. Override deliberately (FAST_EVAL_SEED) to measure seed
# variance; never override it per-trial.
SEED="${FAST_EVAL_SEED:-0}"

case "$PRESET" in
  # "does this checkpoint drive at all" - use after a training run, before anything else.
  smoke) ROUTES=3;  MAX_STEPS=600;  VEHICLES=20 ;;
  # the hyperparameter-search fitness signal: enough routes that the mean is not dominated by a
  # single unlucky spawn, still short enough to run many trials.
  hp)    ROUTES=12; MAX_STEPS=1500; VEHICLES=30 ;;
  # pre-final confirmation with more routes; still far cheaper than the official run.
  wide)  ROUTES=25; MAX_STEPS=2000; VEHICLES=40 ;;
  *) echo "FATAL: unknown preset '$PRESET' (want smoke|hp|wide)"; exit 1 ;;
esac

export WOR_BACKBONE="${WOR_BACKBONE:-regnety_032}"
# leaderboard/scenario_runner roots are new as of the AgentWrapper refactor - eval_wor_closed_
# loop.py now imports leaderboard.autoagents.agent_wrapper and srunner.scenariomanager.*
# directly, which it never needed under the old hand-rolled sensor path. TF++'s own flat
# imports (model, config, data, ...) need team_code importable too, and go FIRST: those names
# are generic enough ("model", "config") that whichever entry wins first decides the import,
# and $MTHESIS must not shadow them for a run that actually wants TF++'s.
AGENT_PYTHONPATH=""
[ -n "$EVAL_AGENT" ] && AGENT_PYTHONPATH="$GARAGE/team_code"
export PYTHONPATH="${AGENT_PYTHONPATH:+$AGENT_PYTHONPATH:}$MTHESIS:$REAL_CARLA/PythonAPI:$REAL_CARLA/PythonAPI/carla:$GARAGE/leaderboard:$GARAGE/scenario_runner${PYTHONPATH:+:$PYTHONPATH}"
cd "$MTHESIS"

# "Ready" means the RPC server answers, not that the TCP port is open. CarlaUE4 binds the port
# well before it will serve requests, so a /dev/tcp probe returns success against a server that
# is still loading - which is how a "reusing CARLA already serving" message was followed by a
# 60 s client time-out. Ask the simulator something only a live server can answer.
carla_ready() {
  "$PY" -c "
import carla, sys
try:
    c = carla.Client('127.0.0.1', $PORT); c.set_timeout(5.0); c.get_server_version()
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
}

# Kill ONLY the CarlaUE4 server bound to this port. Two rules here, both learned the hard way on
# 2026-09-15:
#   1. `pgrep -x CarlaUE4-Linux-Shipping` matches the process *name*, so it can never match the
#      shell running this script. A plain `pkill -9 -f "carla-rpc-port=$PORT"` matches any
#      command line containing that string - including the ssh/bash invocation that launched the
#      cleanup - and kills the caller. That happened.
#   2. The server must be killed at all. run_leaderboard_official.sh traps EXIT to kill its
#      results-watchdog but never its CarlaUE4 server, so when an arm OOM'd, the orphaned server
#      kept running and held 5.7 GB of VRAM - which then blocked the very arm the watchdog was
#      waiting to relaunch. A launcher that starts a server owns killing it on every exit path.
#   3. `pgrep -x CarlaUE4-Linux-Shipping` looks correct and never matches anything: the kernel
#      truncates /proc/PID/comm to 15 characters, so the real process name is "CarlaUE4-Linux-".
#      An -x cleanup like that is a silent no-op that leaks the server on every run. Match on the
#      comm *prefix* instead, walking /proc directly - and because a shell's comm is "bash"/"sh",
#      never "CarlaUE4*", this still cannot match the caller. Both the CarlaUE4.sh launcher and
#      the Shipping binary carry the port in their cmdline, so this kills the pair.
cleanup_carla() {
  local pid cmd
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ -r "/proc/$pid/comm" ] || continue
    case "$(cat "/proc/$pid/comm" 2>/dev/null)" in CarlaUE4*) ;; *) continue ;; esac
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) "
    case "$cmd" in
      *"carla-rpc-port=$PORT "*) kill -9 "$pid" 2>/dev/null ;;
    esac
  done
}

# Reuse an already-serving port instead of paying CARLA's 60-100 s cold start on every trial.
# A hyperparameter sweep of 40 trials would otherwise spend over an hour just booting simulators.
# If we did not start it, we do not kill it - the sweep driver owns that server's lifetime.
STARTED_CARLA=0
if carla_ready; then
  echo "=== reusing CARLA already serving on port $PORT (not ours to kill) ==="
else
  echo "=== starting CARLA on port $PORT ==="
  export XDG_RUNTIME_DIR="/tmp/runtime-carlauser-fast-${LABEL}"
  mkdir -p "$XDG_RUNTIME_DIR"
  chown carlauser:carlauser "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
  setsid su carlauser -c "ulimit -n 65536 2>/dev/null; export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR; \
    $REAL_CARLA/CarlaUE4.sh -carla-rpc-port=$PORT -carla-streaming-port=0 \
    -RenderOffScreen -nosound -vulkan -quality-level=Low" \
    </dev/null >"$OUT/carla_${LABEL}.log" 2>&1 &
  STARTED_CARLA=1
  trap cleanup_carla EXIT INT TERM

  for i in $(seq 1 30); do carla_ready && { echo "CARLA answering RPCs after ~$((i*10))s"; break; }; sleep 10; done
  if ! carla_ready; then
    echo "FATAL: CARLA never answered RPCs on port $PORT - see $OUT/carla_${LABEL}.log"; exit 1
  fi
fi

EXTRA_ARGS=()
[ -n "$EVAL_AGENT" ] && EXTRA_ARGS+=(--agent="$EVAL_AGENT")
[ -n "$EVAL_AGENT_CONFIG" ] && EXTRA_ARGS+=(--agent-config="$EVAL_AGENT_CONFIG")

RESULT="$OUT/${LABEL}.json"
echo "=== fast eval: $LABEL ($ARCH) preset=$PRESET town=$TOWN routes=$ROUTES seed=$SEED ==="
[ -n "$EVAL_AGENT" ] && echo "=== reference agent: $EVAL_AGENT (config $EVAL_AGENT_CONFIG) ==="
START=$(date +%s)
"$PY" "$MTHESIS/eval_wor_closed_loop.py" \
  --checkpoint="$CKPT" \
  --policy_arch="$ARCH" \
  --backbone="$WOR_BACKBONE" \
  --town="$TOWN" \
  --routes="$ROUTES" \
  --route_seed="$SEED" \
  --max_steps="$MAX_STEPS" \
  --num_vehicles="$VEHICLES" \
  --num_walkers=0 \
  --port="$PORT" \
  --tm_port=$((PORT + 8000)) \
  --out="$RESULT" \
  "${EXTRA_ARGS[@]}"
STATUS=$?
ELAPSED=$(( $(date +%s) - START ))

echo "=== $LABEL finished in ${ELAPSED}s (status $STATUS) ==="
[ "$STARTED_CARLA" = "1" ] && cleanup_carla
exit $STATUS
