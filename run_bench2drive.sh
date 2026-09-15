#!/usr/bin/env bash
# Runs ONE Bench2Drive evaluation arm and lets Bench2Drive do the scoring.
#
# usage: run_bench2drive.sh <label> <arch> <ckpt> <routes_xml> <port> <tm_port> [gpu_rank] [routes_subset]
#
# Unlike run_closed_loop_arms.sh, nothing here starts a CARLA server or computes a driving
# score: Bench2Drive's leaderboard_evaluator.py launches the simulator itself and writes the
# statistics, which is the entire point - the resulting number is in the same units as its
# published table (TCP, UniAD, VAD, DriveTransformer, ...) rather than in ours.
#
# routes_subset (optional, comma/dash route-id list, e.g. "25378" or "1-5,12"): passed
# straight to leaderboard_evaluator.py's own --routes-subset. As of 2026-09-09 (challenges_03
# 3.9), three of Bench2Drive's towns (Town11/12/13) crashed or hung the server on load - but that
# was on the box destroyed 2026-09-14; Town12 has since loaded cleanly via a different launcher
# (run_fast_eval.sh, 2026-09-15) on the current box, so this caveat is unverified here and may be
# stale. Re-test before assuming it still applies; leave empty to run every route in the file, or
# pass the subset of route ids whose town is confirmed safe on *this* box.
set -x
LABEL="$1"; ARCH="$2"; CKPT="$3"; ROUTES="$4"; PORT="$5"; TM_PORT="$6"; GPU_RANK="${7:-0}"
ROUTES_SUBSET="${8:-}"

# 2026-09-15: was /workspace/Bench2Drive, which does not exist on this box - Bench2Drive lives
# as a subdirectory of the carla_garage checkout (/workspace/carla_garage/Bench2Drive). That path
# predates this box; never actually exercised here until now.
GARAGE=/workspace/carla_garage
B2D="$GARAGE/Bench2Drive"
REAL_CARLA=/workspace/carla
SHIM=/workspace/carla_shim
OUT=/workspace/bench2drive_out
MTHESIS_ROOT=/workspace/MThesis

# Same pluggable-agent mechanism as run_fast_eval.sh / run_leaderboard_official.sh: default is
# our own bench2drive_agent.py (unchanged behaviour for every existing caller). Point EVAL_AGENT
# at carla_garage/team_code/sensor_agent.py (TF++) or .../team_code/autopilot.py (PDM-Lite, the
# privileged rule-based expert used to generate this project's training data) to run either
# through Bench2Drive's own published 220-route benchmark instead - see eval_tiers_design.md,
# "Stage 1, revised". EVAL_AGENT_CONFIG for TF++ is a pretrained_models directory
# (.../all_towns, since Bench2Drive is scored with the all-towns checkpoint per carla_garage's
# README); for PDM-Lite it is unused (the privileged expert reads simulator ground truth, not a
# checkpoint) but AGENT_CONFIG below still needs SOME string - autopilot.py's setup() ignores it.
EVAL_AGENT="${EVAL_AGENT:-}"
EVAL_AGENT_CONFIG="${EVAL_AGENT_CONFIG:-}"
EVAL_PYTHON="${EVAL_PYTHON:-/workspace/venv_carla/bin/python}"
if [ ! -x "$EVAL_PYTHON" ]; then
  echo "FATAL: EVAL_PYTHON=$EVAL_PYTHON is not executable."
  exit 1
fi
# 2026-09-15: was $B2D/leaderboard/team_code/bench2drive_agent.py, which has never existed -
# bench2drive_agent.py is this project's own adapter, at $MTHESIS_ROOT, not part of the vendored
# Bench2Drive checkout. This script had never actually been run to completion before (no prior
# output under /workspace/bench2drive_out). The evaluator itself resolves the module via
# `sys.path.insert(0, os.path.dirname(args.agent))` (leaderboard_evaluator.py:126), so pointing
# --agent at the right file is sufficient - no separate PYTHONPATH entry needed for it.
AGENT="${EVAL_AGENT:-$MTHESIS_ROOT/bench2drive_agent.py}"
AGENT_CONFIG="${EVAL_AGENT_CONFIG:-${ARCH}:${CKPT}}"
# team_code holds TF++'s/PDM-Lite's flat imports (model, config, data, autopilot, ...) as
# top-level modules. Bench2Drive ships its OWN team_code/ with different files under the same
# names (config.py, data_agent.py, autopilot.py all exist in both) - carla_garage's own reference
# launcher (Bench2Drive/leaderboard/scripts/run_evaluation_tf++.sh) points TEAM_AGENT at the
# *top-level* carla_garage/team_code, not Bench2Drive's copy, and puts that directory on
# PYTHONPATH ahead of everything else for exactly this reason. Mirrored here: only prepended when
# EVAL_AGENT overrides the default, so bench2drive_agent.py's own imports (which expect
# Bench2Drive's team_code to win) are unaffected.
AGENT_PYTHONPATH=""
[ -n "$EVAL_AGENT" ] && AGENT_PYTHONPATH="$GARAGE/team_code"

# leaderboard_evaluator.py:203 launches CarlaUE4.sh as the *current* user. CARLA aborts as
# root in this container - the reason every other script in this project goes through
# `su carlauser` (challenges_01 1.1/1.2). CARLA_ROOT is therefore pointed at a shim whose
# CarlaUE4.sh drops privileges and adds the -vulkan/-quality-level flags used everywhere
# else here. Safe because CARLA_ROOT is read in exactly one place in the evaluator (that
# launch path); the PythonAPI paths below still point at the real install.
#
# 2026-09-15: both the shim path and its XDG_RUNTIME_DIR are now keyed by LABEL. They used to be
# one shared /workspace/carla_shim/CarlaUE4.sh with no XDG_RUNTIME_DIR override at all, which is
# fine for one arm at a time but not for two: run_leaderboard_official.sh's own history records
# two CARLA processes sharing one XDG_RUNTIME_DIR corrupting each other's Vulkan ICD/session
# state. This script never needed the fix before because it had never actually been run to
# completion (see the other 2026-09-15 fixes above) - now that it works, running TF++/cnn/qwen
# concurrently to cut the Bench2Drive comparison's wall time is the whole point, so isolation
# has to hold from the first concurrent run, not be bolted on after a corrupted run explains why.
SHIM="$SHIM/$LABEL"
XDG_RUNTIME_DIR="/tmp/runtime-carlauser-b2d-${LABEL}"
mkdir -p "$SHIM" "$XDG_RUNTIME_DIR"
chown carlauser:carlauser "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
cat > "$SHIM/CarlaUE4.sh" <<SHIMEOF
#!/usr/bin/env bash
exec su carlauser -c "export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR; /workspace/carla/CarlaUE4.sh \$* -vulkan -quality-level=Low"
SHIMEOF
chmod +x "$SHIM/CarlaUE4.sh"

export CARLA_ROOT="$SHIM"
export MTHESIS_ROOT
export SCENARIO_RUNNER_ROOT="$B2D/scenario_runner"
export LEADERBOARD_ROOT="$B2D/leaderboard"

# Deliberately NOT appending carla-0.9.15-py3.7-linux-x86_64.egg the way
# leaderboard/scripts/run_evaluation.sh does: venv_carla is Python 3.10 with `carla`
# installed from pip, and putting a py3.7 egg ahead of it on sys.path shadows a working
# client with an unloadable one - the same trap documented for eval_wor.py in 11.7.
# AGENT_PYTHONPATH goes FIRST so a non-default agent's top-level team_code wins the naming
# collision with Bench2Drive's own team_code (see the comment above AGENT_PYTHONPATH).
export PYTHONPATH="${AGENT_PYTHONPATH:+$AGENT_PYTHONPATH:}$PYTHONPATH:$REAL_CARLA/PythonAPI:$REAL_CARLA/PythonAPI/carla:$B2D/leaderboard:$B2D/leaderboard/team_code:$B2D/scenario_runner"

export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=True
export DEBUG_CHALLENGE=0
export IS_BENCH2DRIVE=True
export SAVE_PATH="${OUT}/${LABEL}/"
mkdir -p "$SAVE_PATH"

# 2026-09-15: unlike run_leaderboard_official.sh/run_fast_eval.sh, this script never starts
# CarlaUE4 itself - the evaluator does, via the CARLA_ROOT shim above - so it was never given a
# matching reaper either. Confirmed leaking: a probe run that crashed on a bad --agent path
# (ModuleNotFoundError, before the sim loop even started) left CarlaUE4-Linux-Shipping running
# and holding 5 GB of VRAM with nothing left to kill it. Same fix as the other two scripts -
# match on the comm *prefix* via /proc (pgrep -x truncates to 15 chars and never matches;
# pkill -f on the port string can match the caller's own command line) - just triggered by the
# evaluator's own exit instead of by a server this script launched directly.
cleanup() {
  local pid cmd
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ -r "/proc/$pid/comm" ] || continue
    case "$(cat "/proc/$pid/comm" 2>/dev/null)" in CarlaUE4*) ;; *) continue ;; esac
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) "
    case "$cmd" in *"carla-rpc-port=$PORT "*) kill -9 "$pid" 2>/dev/null ;; esac
  done
}
trap cleanup EXIT INT TERM

cd "$B2D"
EXTRA_ARGS=()
if [ -n "$ROUTES_SUBSET" ]; then
  EXTRA_ARGS+=(--routes-subset="$ROUTES_SUBSET")
fi
[ -n "$EVAL_AGENT" ] && echo "=== reference agent: $AGENT (config $AGENT_CONFIG) ==="
"$EVAL_PYTHON" leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="$ROUTES" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="${OUT}/${LABEL}.json" \
  --agent="$AGENT" \
  --agent-config="$AGENT_CONFIG" \
  --debug=0 \
  --resume=True \
  --port="$PORT" \
  --traffic-manager-port="$TM_PORT" \
  --gpu-rank="$GPU_RANK" \
  "${EXTRA_ARGS[@]}"
