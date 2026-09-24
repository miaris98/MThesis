#!/usr/bin/env bash
# Provision a fresh box for Bench2Drive and run the b2d20 closed-loop suite on one WoR checkpoint.
#
# usage:
#   scripts/eval/launch_b2d20.sh setup            # CARLA + maps + venv_carla + carla_garage + deps
#   scripts/eval/launch_b2d20.sh preflight        # load the checkpoint through the eval agent
#   scripts/eval/launch_b2d20.sh run              # 2 parallel arms x 10 routes, under guardians
#
# env: LABEL (default b2d20_8towns_e20), ARCH (qwen30m), CKPT (8-town epoch-20 best_model.pth)
#
# b2d20 = the same 20-route subset of bench2drive220.xml used for the 2026-09-16 baseline run
# (E:\MThesis_EXP\bench2drive_eval\run_20260916_final: TF++ 80.33 DS, cnn 26.23, qwen 27.05), so
# results are directly comparable. The 20 routes are split into two 10-route arms on separate
# CARLA ports so both halves run concurrently; merge afterwards with merge_leaderboard_results.py.
set -uo pipefail
PHASE="${1:?usage: $0 setup|preflight|run}"
LABEL="${LABEL:-b2d20_8towns_e20}"
ARCH="${ARCH:-qwen30m}"
CKPT="${CKPT:-/workspace/checkpoints/wor_qwen30m_8towns_fast/best_model.pth}"
MTHESIS_ROOT=/workspace/MThesis
GARAGE=/workspace/carla_garage
PY=/workspace/venv_carla/bin/python
ROUTES="$GARAGE/Bench2Drive/leaderboard/data/bench2drive220.xml"
ARM_A_ROUTES="2050,2143,2286,2509,2664,3373,3457,3717,3936,23687"
ARM_B_ROUTES="24340,24784,24795,25318,25975,26401,27515,27532,28035,28198"

case "$PHASE" in
setup)
  set -e
  if [ ! -f /workspace/carla/CarlaUE4.sh ] || [ ! -x "$PY" ]; then
    bash "$MTHESIS_ROOT/scripts/setup/setup_carla_eval.sh"
  fi
  [ -d "$GARAGE/Bench2Drive/leaderboard" ] || \
    git clone --depth 1 -b leaderboard_2 https://github.com/autonomousvision/carla_garage.git "$GARAGE"
  # Vanilla leaderboard/scenario_runner runtime set (py-trees MUST be 0.8.3 - a bare install
  # pulls 2.x, S-011) plus timm for the regnety_032 backbone.
  VIRTUAL_ENV=/workspace/venv_carla uv pip install six "py-trees==0.8.3" Shapely xmlschema ephem \
    tabulate opencv-python matplotlib psutil pygame pexpect dictor transforms3d \
    simple-watchdog-timer requests timm
  "$PY" -c "import carla, torch, timm, py_trees; print('carla ok | torch', torch.__version__, torch.cuda.is_available(), '| py_trees', py_trees.__version__ if hasattr(py_trees,'__version__') else '?')"
  echo "LAUNCH_B2D20_SETUP_DONE"
  ;;
preflight)
  # Builds the agent exactly as the evaluator will and fails loudly if the checkpoint does not
  # load - a silent fallback to an untrained model is how the pre-2026-09-17 numbers went wrong.
  cd "$MTHESIS_ROOT"
  PYTHONPATH="$MTHESIS_ROOT:/workspace/carla/PythonAPI:/workspace/carla/PythonAPI/carla:$GARAGE/Bench2Drive/leaderboard:$GARAGE/Bench2Drive/scenario_runner" \
    "$PY" - "$ARCH" "$CKPT" <<'EOF'
import sys
from src.agents.wor_agent import WorldOnRailsAgent
arch, ckpt = sys.argv[1], sys.argv[2]
a = WorldOnRailsAgent(checkpoint_path=ckpt, policy_arch=arch)
print("PREFLIGHT_OK backbone=", a.backbone_name, "route_points=", a.route_points)
EOF
  ;;
run)
  # ARMS: space-separated "suffix:rpc_port:tm_port:route,ids" entries; default = the two 10-route
  # halves. GPU: device for both CARLA (-graphicsadapter) and the agent (CUDA_VISIBLE_DEVICES),
  # so a 2-GPU box can keep CARLA off the GPU that trains.
  GPU="${GPU:-0}"
  ARMS="${ARMS:-a:2000:8000:$ARM_A_ROUTES b:2100:8100:$ARM_B_ROUTES}"
  mkdir -p /workspace/bench2drive_out
  for spec in $ARMS; do
    IFS=: read -r arm PORT TM SUBSET <<<"$spec"
    N=$(awk -F, '{print NF}' <<<"$SUBSET")
    ARM_LABEL="${LABEL}_${arm}"
    CUDA_VISIBLE_DEVICES="$GPU" nohup bash "$MTHESIS_ROOT/scripts/eval/b2d_guardian.sh" "$ARM_LABEL" \
      "/workspace/bench2drive_out/${ARM_LABEL}.json" "$N" \
      bash "$MTHESIS_ROOT/scripts/eval/run_bench2drive.sh" "$ARM_LABEL" "$ARCH" "$CKPT" \
        "$ROUTES" "$PORT" "$TM" "$GPU" "$SUBSET" \
      > "/workspace/guardian_${ARM_LABEL}.out" 2>&1 &
    echo "launched guardian for $ARM_LABEL ($N routes, port $PORT, GPU $GPU) pid $!"
    sleep 20  # stagger CARLA boots so the servers do not race for the same Vulkan init
  done
  ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
