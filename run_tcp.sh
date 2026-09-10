#!/usr/bin/env bash
# Runs the TCP baseline (Bench2DriveZoo tcp/admlp branch, tcp_b2d.ckpt) through Bench2Drive,
# for external comparability against our own WoR-style arms.
# usage: run_tcp.sh <label> <ckpt> <routes_xml> <port> <tm_port> [gpu_rank] [routes_subset]
set -x
LABEL="$1"; CKPT="$2"; ROUTES="$3"; PORT="$4"; TM_PORT="$5"; GPU_RANK="${6:-0}"
ROUTES_SUBSET="${7:-}"

B2D=/workspace/Bench2Drive
REAL_CARLA=/workspace/carla
SHIM=/workspace/carla_shim
OUT=/workspace/bench2drive_out

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
export PYTHONPATH="$PYTHONPATH:$REAL_CARLA/PythonAPI:$REAL_CARLA/PythonAPI/carla:$B2D/leaderboard:$B2D/leaderboard/team_code:$B2D/scenario_runner"

export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=True
export DEBUG_CHALLENGE=0
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=merge_ctrl_traj
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
  --agent="$B2D/leaderboard/team_code/tcp_b2d_agent.py" \
  --agent-config="$CKPT" \
  --debug=0 \
  --resume=True \
  --port="$PORT" \
  --traffic-manager-port="$TM_PORT" \
  --gpu-rank="$GPU_RANK" \
  "${EXTRA_ARGS[@]}"
