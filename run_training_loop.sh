#!/usr/bin/env bash
# ==============================================================================
#  🚀 CARLA Multi-Camera PPO Continuous Auto-Restart Training Supervisor
#  Automatically restarts CARLA server & resumes training on any crash/timeout
# ==============================================================================

TOTAL_STEPS=${1:-20000}
BACKBONE=${2:-lav}
TOWN=${3:-Town10HD_Opt}
NUM_VEHICLES=${4:-3}
NUM_WALKERS=${5:-10}

# Locate Python binary in carla_py38 conda environment
PYTHON_BIN="python"
for p in "/workspace/miniconda/envs/carla_py38/bin/python" \
         "/opt/conda/envs/carla_py38/bin/python" \
         "$HOME/miniconda3/envs/carla_py38/bin/python" \
         "$HOME/anaconda3/envs/carla_py38/bin/python" \
         "/root/miniconda3/envs/carla_py38/bin/python" \
         "/usr/local/miniconda3/envs/carla_py38/bin/python"; do
    if [ -f "$p" ]; then
        PYTHON_BIN="$p"
        break
    fi
done

# Auto-activate carla_py38 environment if available
for p in "/workspace/miniconda" "/opt/conda" "$HOME/miniconda3" "$HOME/anaconda3" "/root/miniconda3" "/usr/local/miniconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
        source "$p/etc/profile.d/conda.sh"
        conda activate carla_py38 2>/dev/null || true
        break
    fi
done

# Ensure required tracking packages exist in target environment
"$PYTHON_BIN" -c "import tensorboard, mlflow" 2>/dev/null || "$PYTHON_BIN" -m pip install tensorboard mlflow 2>/dev/null || true

echo "=============================================================="
echo "   🔄 Starting Autonomous Auto-Restart Training Supervisor    "
echo "=============================================================="
echo "Target Steps: $TOTAL_STEPS | Backbone: $BACKBONE | Town: $TOWN"
echo "NPC Vehicles: $NUM_VEHICLES | Pedestrians: $NUM_WALKERS"
echo "Python Executable: $PYTHON_BIN"
echo "=============================================================="

attempt=1

while true; do
    echo ""
    echo "--------------------------------------------------------------"
    echo " 🚀 Launching Training Session (Run #$attempt)..."
    echo "--------------------------------------------------------------"

    # 1. Clean up any stale or frozen processes
    pkill -9 -f CarlaUE4 2>/dev/null || true
    pkill -9 -f train_rl_agent 2>/dev/null || true
    fuser -k 2000/tcp 2001/tcp 2002/tcp 8000/tcp 2>/dev/null || true
    tmux kill-session -t carla_server 2>/dev/null || true
    sleep 2

    # 2. Start clean CARLA Server instance
    if [ -f "/workspace/carla/CarlaUE4.sh" ]; then
        echo "--> Starting fresh CARLA server (Town: $TOWN)..."
        tmux new-session -d -s carla_server "su carlauser -c '/workspace/carla/CarlaUE4.sh /Game/Carla/Maps/$TOWN -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low -benchmark -fps=20' > /workspace/carla_server.log 2>&1"
        sleep 18
    fi

    WEIGHTS_ARG=""
    if [ -f "./papers_and_code/LAV/lav_pretrained.pth" ]; then
        WEIGHTS_ARG="--weights-path ./papers_and_code/LAV/lav_pretrained.pth"
    elif [ -f "/workspace/pretrained_carla/model_0030_0.pth" ]; then
        WEIGHTS_ARG="--weights-path /workspace/pretrained_carla/model_0030_0.pth"
    fi

    # 3. Launch / Resume Training
    "$PYTHON_BIN" train_rl_agent.py \
        --env-type camera_easycarla \
        --backbone "$BACKBONE" \
        --town "$TOWN" \
        --num-vehicles "$NUM_VEHICLES" \
        --num-walkers "$NUM_WALKERS" \
        --ent-coef 0.02 \
        --minibatch-size 128 \
        --use-mlflow \
        --mlflow-port 5055 \
        $WEIGHTS_ARG \
        --resume \
        --total-steps "$TOTAL_STEPS"

    exit_code=$?

    # 4. Check Exit Status
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=============================================================="
        echo "   🎉 Training Completed Successfully to $TOTAL_STEPS Steps!  "
        echo "=============================================================="
        break
    else
        echo ""
        echo "⚠️  Training process terminated with exit code $exit_code (timeout/crash)."
        echo "🔄  Auto-restarting in 5 seconds and resuming from last saved checkpoint..."
        sleep 5
        attempt=$((attempt + 1))
    fi
done
