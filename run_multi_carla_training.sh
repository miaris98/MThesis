#!/usr/bin/env bash
# ==============================================================================
#  🚀 Multi-CARLA Server Parallel PPO Continuous Auto-Restart Training Supervisor
#  Runs N CARLA simulators concurrently and feeds 100M Qwen Decision Policy
# ==============================================================================

# Default configuration
NUM_ENVS=2
START_PORT=2000
TOTAL_STEPS=50000
POLICY_ARCH=qwen100m
BACKBONE=lav
TOWN=Town10HD_Opt
NUM_VEHICLES=3
NUM_WALKERS=10
START_FRESH=false

# Argument parsing
POS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --fresh|--from-scratch|fresh)
            START_FRESH=true
            ;;
        --num-envs=*)
            NUM_ENVS="${arg#*=}"
            ;;
        --policy=*)
            POLICY_ARCH="${arg#*=}"
            ;;
        *)
            POS_ARGS+=("$arg")
            ;;
    esac
done

[ ${#POS_ARGS[@]} -ge 1 ] && [ -n "${POS_ARGS[0]}" ] && NUM_ENVS=${POS_ARGS[0]}
[ ${#POS_ARGS[@]} -ge 2 ] && [ -n "${POS_ARGS[1]}" ] && TOTAL_STEPS=${POS_ARGS[1]}
[ ${#POS_ARGS[@]} -ge 3 ] && [ -n "${POS_ARGS[2]}" ] && POLICY_ARCH=${POS_ARGS[2]}
[ ${#POS_ARGS[@]} -ge 4 ] && [ -n "${POS_ARGS[3]}" ] && BACKBONE=${POS_ARGS[3]}
[ ${#POS_ARGS[@]} -ge 5 ] && [ -n "${POS_ARGS[4]}" ] && TOWN=${POS_ARGS[4]}

# Compute non-overlapping CARLA port list (stride of 4 per instance)
PORTS=()
PORTS_CSV=""
for ((i=0; i<NUM_ENVS; i++)); do
    P=$((START_PORT + i * 4))
    PORTS+=("$P")
    if [ -z "$PORTS_CSV" ]; then
        PORTS_CSV="$P"
    else
        PORTS_CSV="${PORTS_CSV},$P"
    fi
done

if [ "$START_FRESH" = true ]; then
    echo "=============================================================="
    echo "🧹 [START FRESH] Requested fresh training run from scratch."
    echo "   Clearing previous checkpoints and telemetry logs..."
    echo "=============================================================="
    rm -rf /workspace/checkpoints/* /workspace/runs/* /workspace/telemetry.csv 2>/dev/null || true
    rm -rf ./runs/* ./checkpoints/* ./telemetry.csv 2>/dev/null || true
    echo "✓ Previous checkpoints and run logs cleared!"
fi

# Locate Python 3.8 binary in carla_py38 environment
PYTHON_BIN=""
for p in "/venv/carla_py38/bin/python" \
         "/opt/conda/envs/carla_py38/bin/python" \
         "/workspace/miniconda/envs/carla_py38/bin/python" \
         "$HOME/miniconda3/envs/carla_py38/bin/python" \
         "$HOME/anaconda3/envs/carla_py38/bin/python" \
         "/root/miniconda3/envs/carla_py38/bin/python" \
         "/root/.conda/envs/carla_py38/bin/python" \
         "/usr/local/miniconda3/envs/carla_py38/bin/python"; do
    if [ -f "$p" ]; then
        PYTHON_BIN="$p"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(which python3.8 2>/dev/null || echo "python")
fi

# Auto-activate carla_py38 environment if available
for p in "/workspace/miniconda" "/opt/conda" "$HOME/miniconda3" "$HOME/anaconda3" "/root/miniconda3" "/usr/local/miniconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
        source "$p/etc/profile.d/conda.sh"
        conda activate carla_py38 2>/dev/null || true
        break
    fi
done

export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Ensure required tracking packages exist in target environment
export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=============================================================="
echo "   🔄 Starting Multi-CARLA Auto-Restart Training Supervisor   "
echo "=============================================================="
echo "Parallel Environments: $NUM_ENVS Instances (Ports: $PORTS_CSV)"
echo "Policy Architecture:   ${POLICY_ARCH^^} (~100M Params)"
echo "Vision Backbone:       $BACKBONE"
echo "Target Steps:          $TOTAL_STEPS"
echo "CARLA Map:             $TOWN"
echo "Mode:                  $([ "$START_FRESH" = true ] && echo 'FRESH (From Scratch)' || echo 'RESUME (From Checkpoint)')"
echo "Python Executable:     $PYTHON_BIN"
echo "=============================================================="

# Auto-detect an available port for MLflow UI
find_mlflow_port() {
    SKIP_PORTS="22 1111 2000 2001 2002 2004 2005 2006 2008 2009 2010 6006 8080 8384"
    MAPPED_PORTS=$(iptables -t nat -L -n 2>/dev/null | grep DNAT | grep -oP 'to:[^:]+:\K\d+' | sort -un)

    if [ -n "$MAPPED_PORTS" ]; then
        for port in $MAPPED_PORTS; do
            if echo "$SKIP_PORTS" | grep -qw "$port"; then continue; fi
            if ! ss -tlnp 2>/dev/null | grep -q ":${port} "; then
                echo "$port"
                return 0
            fi
        done
    fi
    for port in 10100 10200 9090 7070 4040; do
        if ! ss -tlnp 2>/dev/null | grep -q ":${port} "; then
            echo "$port"
            return 0
        fi
    done
    echo "10100"
    return 0
}

# Start or recover MLflow server in isolated tmux session
if tmux has-session -t mlflow_server 2>/dev/null; then
    MLFLOW_PORT=$(ss -tlnp 2>/dev/null | grep "mlflow\|gunicorn" | grep -oP ':(\d+)' | head -1 | tr -d ':')
    MLFLOW_PORT=${MLFLOW_PORT:-$(find_mlflow_port)}
    echo "✓ MLflow UI server running in protected tmux session 'mlflow_server' (port ${MLFLOW_PORT})."
else
    MLFLOW_PORT=$(find_mlflow_port)
    fuser -k ${MLFLOW_PORT}/tcp 2>/dev/null || true
    sleep 1
    echo "--> 📊 Launching persistent MLflow UI server on port ${MLFLOW_PORT}..."
    tmux new-session -d -s mlflow_server \
        "$PYTHON_BIN -m mlflow ui --host 0.0.0.0 --port ${MLFLOW_PORT} --backend-store-uri /workspace/MThesis/mlruns > /workspace/mlflow_server.log 2>&1"
    
    # Wait until MLflow server is responding
    for i in $(seq 1 15); do
        if curl -s -I "http://127.0.0.1:${MLFLOW_PORT}" 2>/dev/null | grep -q -E "HTTP/|200|302|mlflow"; then
            break
        fi
        sleep 1
    done
    echo "✓ MLflow UI server active on port ${MLFLOW_PORT}"
fi

# Ensure cloudflared is installed for public HTTPS dashboard access
if ! command -v cloudflared &>/dev/null; then
    echo "--> Installing cloudflared for direct public HTTPS dashboard access..."
    (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O /tmp/cloudflared.deb 2>/dev/null && \
     dpkg -i /tmp/cloudflared.deb >/dev/null 2>&1 && rm -f /tmp/cloudflared.deb) || true
fi

CLOUDFLARE_URL=""
if command -v cloudflared &>/dev/null; then
    if tmux has-session -t mlflow_tunnel 2>/dev/null; then
        if ! grep -q "127.0.0.1:${MLFLOW_PORT}" /tmp/mlflow_tunnel.log 2>/dev/null; then
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            rm -f /tmp/mlflow_tunnel.log
        fi
    fi

    if ! tmux has-session -t mlflow_tunnel 2>/dev/null; then
        rm -f /tmp/mlflow_tunnel.log
        echo "--> 🌐 Launching public Cloudflare HTTPS tunnel for MLflow (port ${MLFLOW_PORT})..."
        tmux new-session -d -s mlflow_tunnel \
            "cloudflared tunnel --url http://127.0.0.1:${MLFLOW_PORT} 2>&1 | tee /tmp/mlflow_tunnel.log"
    fi

    echo "--> Waiting for Cloudflare public tunnel URL to initialize..."
    for i in $(seq 1 12); do
        if [ -f /tmp/mlflow_tunnel.log ]; then
            CLOUDFLARE_URL=$(grep -o 'https://[-a-zA-Z0-9@:%._\+~#=]*\.trycloudflare\.com' /tmp/mlflow_tunnel.log | head -n 1)
            if [ -n "$CLOUDFLARE_URL" ]; then
                break
            fi
        fi
        sleep 1
    done
fi

echo "=============================================================="
echo "   📊 MLFLOW DASHBOARD ONLINE (PORT ${MLFLOW_PORT})           "
if [ -n "$CLOUDFLARE_URL" ]; then
    echo -e "   👉 \033[1;32mPublic HTTPS URL:  $CLOUDFLARE_URL\033[0m"
else
    echo "   👉 Public HTTPS URL:  (Check: tail -n 20 /tmp/mlflow_tunnel.log)"
fi
echo "   👉 Vast.ai Tunnel:    Open Port ${MLFLOW_PORT} in Vast.ai Tunnels UI"
echo "   👉 Localhost URL:     http://127.0.0.1:${MLFLOW_PORT}"
echo "=============================================================="

attempt=1

while true; do
    echo ""
    echo "--------------------------------------------------------------"
    echo " 🚀 Launching Multi-CARLA Training Session (Run #$attempt)..."
    echo "--------------------------------------------------------------"

    # 1. Clean up stale processes and release GPU memory
    echo "--> Cleaning up stale training and CARLA processes..."
    pkill -9 -f train_rl_agent 2>/dev/null || true
    pkill -9 -f CarlaUE4 2>/dev/null || true
    pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null || true
    for ((i=0; i<NUM_ENVS; i++)); do
        tmux kill-session -t "carla_server_${i}" 2>/dev/null || true
        P=${PORTS[$i]}
        fuser -k ${P}/tcp $((P+1))/tcp $((P+2))/tcp 2>/dev/null || true
    done
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 5  # Allow driver to fully release GPU Vulkan contexts

    # 2. Launch CARLA servers (staggered by 3s to prevent Vulkan driver race conditions)
    for ((i=0; i<NUM_ENVS; i++)); do
        PORT=${PORTS[$i]}
        SESSION_NAME="carla_server_${i}"
        LOG_FILE="/workspace/carla_server_${PORT}.log"
        echo "--> Launching CARLA Server #$((i+1))/$NUM_ENVS on port $PORT (tmux: $SESSION_NAME)..."
        
        tmux new-session -d -s "$SESSION_NAME" \
            "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -vulkan -quality-level=Low -benchmark -fps=20' > $LOG_FILE 2>&1"
        sleep 3
    done
    sleep 4

    # 3. Health check all CARLA server instances
    echo "--> Probing all $NUM_ENVS CARLA server instances..."
    all_ready=true
    for ((i=0; i<NUM_ENVS; i++)); do
        PORT=${PORTS[$i]}
        SESSION_NAME="carla_server_${i}"
        LOG_FILE="/workspace/carla_server_${PORT}.log"
        echo -n "   [CARLA #$((i+1))/$NUM_ENVS | Port $PORT] Waiting for server initialization"
        ready=false
        for attempt_check in $(seq 1 30); do
            echo -n "."
            if "$PYTHON_BIN" -c "
import sys, glob
for e in glob.glob('/workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg'): sys.path.insert(0, e)
import carla
c = carla.Client('127.0.0.1', $PORT)
c.set_timeout(2.0)
v = c.get_server_version()
" 2>/dev/null; then
                ready=true
                echo ""
                echo "✓ CARLA Server on port $PORT is online and verified!"
                break
            fi
            sleep 2
        done

        # If Vulkan didn't respond in 60s, fallback to OpenGL for this instance
        if [ "$ready" = false ]; then
            echo ""
            echo "⚠️  CARLA on port $PORT not responding with Vulkan. Retrying with OpenGL mode..."
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
            fuser -k ${PORT}/tcp $((PORT+1))/tcp $((PORT+2))/tcp 2>/dev/null || true
            sleep 2
            tmux new-session -d -s "$SESSION_NAME" \
                "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -opengl -quality-level=Low -benchmark -fps=20' > $LOG_FILE 2>&1"
            echo -n "   [CARLA #$((i+1))/$NUM_ENVS | Port $PORT (OpenGL)] Waiting for initialization"
            for attempt_check in $(seq 1 25); do
                echo -n "."
                if "$PYTHON_BIN" -c "
import sys, glob
for e in glob.glob('/workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg'): sys.path.insert(0, e)
import carla
c = carla.Client('127.0.0.1', $PORT)
c.set_timeout(2.0)
v = c.get_server_version()
" 2>/dev/null; then
                    ready=true
                    echo ""
                    echo "✓ CARLA Server on port $PORT (OpenGL) is online and verified!"
                    break
                fi
                sleep 2
            done
        fi

        if [ "$ready" = false ]; then
            echo ""
            echo "⚠️  CARLA Server on port $PORT failed to respond. Check log: tail -n 20 $LOG_FILE"
            all_ready=false
            break
        fi
    done

    if [ "$all_ready" = false ]; then
        echo "⚠️  One or more CARLA servers failed startup. Retrying full restart in 10s..."
        sleep 10
        continue
    fi

    # 4. Configure weights and resume/fresh flags
    WEIGHTS_ARG=""
    if [ -f "./papers_and_code/LAV/lav_pretrained.pth" ]; then
        WEIGHTS_ARG="--weights-path ./papers_and_code/LAV/lav_pretrained.pth"
    elif [ -f "/workspace/pretrained_carla/model_0030_0.pth" ]; then
        WEIGHTS_ARG="--weights-path /workspace/pretrained_carla/model_0030_0.pth"
    fi

    RESUME_ARG="--resume"
    if [ "$START_FRESH" = true ] && [ "$attempt" -eq 1 ]; then
        RESUME_ARG="--fresh"
        echo "🌱 Launching FRESH multi-server training from step 0..."
    else
        echo "🔄 Resuming training from latest saved checkpoint..."
    fi

    # 5. Launch Training Pipeline with capped thread allocations and silenced warnings
    export PYTHONWARNINGS="ignore"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    "$PYTHON_BIN" train_rl_agent.py \
        --env-type camera_easycarla \
        --num-envs "$NUM_ENVS" \
        --carla-ports "$PORTS_CSV" \
        --backbone "$BACKBONE" \
        --policy-arch "$POLICY_ARCH" \
        --town "$TOWN" \
        --num-vehicles "$NUM_VEHICLES" \
        --num-walkers "$NUM_WALKERS" \
        --frame-skip 2 \
        --rollout-steps 250 \
        --minibatch-size 128 \
        --ent-coef 0.05 \
        --use-mlflow \
        --mlflow-port "$MLFLOW_PORT" \
        --log-dir /workspace/runs \
        --checkpoint-dir /workspace/checkpoints \
        $WEIGHTS_ARG \
        $RESUME_ARG \
        --total-steps "$TOTAL_STEPS"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=============================================================="
        echo "   🎉 Multi-CARLA Training Completed to $TOTAL_STEPS Steps!   "
        echo "=============================================================="
        break
    else
        echo ""
        echo "⚠️  Training process terminated with exit code $exit_code."
        echo "🔄  Auto-restarting multi-CARLA servers and training in 10 seconds..."
        echo "    (Latest model checkpoint preserved — zero loss of progress)"
        sleep 10
        attempt=$((attempt + 1))
    fi
done
