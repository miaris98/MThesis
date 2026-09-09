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
# straight to leaderboard_evaluator.py's own --routes-subset. Needed on this instance
# because three of Bench2Drive's towns (Town11/12/13, challenges_03 3.9) crash or hang the
# server on load here - leave empty to run every route in the file, or pass the subset of
# route ids whose town is confirmed safe.
set -x
LABEL="$1"; ARCH="$2"; CKPT="$3"; ROUTES="$4"; PORT="$5"; TM_PORT="$6"; GPU_RANK="${7:-0}"
ROUTES_SUBSET="${8:-}"

B2D=/workspace/Bench2Drive
REAL_CARLA=/workspace/carla
SHIM=/workspace/carla_shim
OUT=/workspace/bench2drive_out

# leaderboard_evaluator.py:203 launches CarlaUE4.sh as the *current* user. CARLA aborts as
# root in this container - the reason every other script in this project goes through
# `su carlauser` (challenges_01 1.1/1.2). CARLA_ROOT is therefore pointed at a shim whose
# CarlaUE4.sh drops privileges and adds the -vulkan/-quality-level flags used everywhere
# else here. Safe because CARLA_ROOT is read in exactly one place in the evaluator (that
# launch path); the PythonAPI paths below still point at the real install.
mkdir -p "$SHIM"
cat > "$SHIM/CarlaUE4.sh" <<'SHIMEOF'
#!/usr/bin/env bash
exec su carlauser -c "/workspace/carla/CarlaUE4.sh $* -vulkan -quality-level=Low"
SHIMEOF
chmod +x "$SHIM/CarlaUE4.sh"

export CARLA_ROOT="$SHIM"
export MTHESIS_ROOT=/workspace/MThesis
export SCENARIO_RUNNER_ROOT="$B2D/scenario_runner"
export LEADERBOARD_ROOT="$B2D/leaderboard"

# Deliberately NOT appending carla-0.9.15-py3.7-linux-x86_64.egg the way
# leaderboard/scripts/run_evaluation.sh does: venv_carla is Python 3.10 with `carla`
# installed from pip, and putting a py3.7 egg ahead of it on sys.path shadows a working
# client with an unloadable one - the same trap documented for eval_wor.py in 11.7.
export PYTHONPATH="$PYTHONPATH:$REAL_CARLA/PythonAPI:$REAL_CARLA/PythonAPI/carla:$B2D/leaderboard:$B2D/leaderboard/team_code:$B2D/scenario_runner"

export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=True
export DEBUG_CHALLENGE=0
export IS_BENCH2DRIVE=True
export SAVE_PATH="${OUT}/${LABEL}/"
mkdir -p "$SAVE_PATH"

cd "$B2D"
EXTRA_ARGS=()
if [ -n "$ROUTES_SUBSET" ]; then
  EXTRA_ARGS+=(--routes-subset="$ROUTES_SUBSET")
fi
/workspace/venv_carla/bin/python leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="$ROUTES" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="${OUT}/${LABEL}.json" \
  --agent="$B2D/leaderboard/team_code/bench2drive_agent.py" \
  --agent-config="${ARCH}:${CKPT}" \
  --debug=0 \
  --resume=True \
  --port="$PORT" \
  --traffic-manager-port="$TM_PORT" \
  --gpu-rank="$GPU_RANK" \
  "${EXTRA_ARGS[@]}"
