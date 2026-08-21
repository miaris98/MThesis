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

echo "=============================================================="
echo "   🔄 Starting Autonomous Auto-Restart Training Supervisor    "
echo "=============================================================="
echo "Target Steps: $TOTAL_STEPS | Backbone: $BACKBONE | Town: $TOWN"
echo "NPC Vehicles: $NUM_VEHICLES | Pedestrians: $NUM_WALKERS"
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
        sleep 8
    fi

    # 3. Launch / Resume Training
    python train_rl_agent.py \
        --env-type camera_easycarla \
        --backbone "$BACKBONE" \
        --town "$TOWN" \
        --num-vehicles "$NUM_VEHICLES" \
        --num-walkers "$NUM_WALKERS" \
        --ent-coef 0.02 \
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
