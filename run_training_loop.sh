#!/usr/bin/env bash
# ==============================================================================
#  🚀 CARLA Multi-Camera PPO Continuous Auto-Restart Training Supervisor
#  Automatically restarts CARLA server & resumes training on any crash/timeout
# ==============================================================================

# Default configuration
TOTAL_STEPS=50000
BACKBONE=lav
POLICY_ARCH=qwen100m
TOWN=Town10HD_Opt
NUM_VEHICLES=3
NUM_WALKERS=10
START_FRESH=false

# Robust argument parsing supporting --fresh in any position
POS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --fresh|--from-scratch|fresh)
            START_FRESH=true
            ;;
        --policy=*)
            POLICY_ARCH="${arg#*=}"
            ;;
        *)
            POS_ARGS+=("$arg")
            ;;
    esac
done

[ ${#POS_ARGS[@]} -ge 1 ] && [ -n "${POS_ARGS[0]}" ] && TOTAL_STEPS=${POS_ARGS[0]}
[ ${#POS_ARGS[@]} -ge 2 ] && [ -n "${POS_ARGS[1]}" ] && BACKBONE=${POS_ARGS[1]}
[ ${#POS_ARGS[@]} -ge 3 ] && [ -n "${POS_ARGS[2]}" ] && POLICY_ARCH=${POS_ARGS[2]}
[ ${#POS_ARGS[@]} -ge 4 ] && [ -n "${POS_ARGS[3]}" ] && TOWN=${POS_ARGS[3]}
[ ${#POS_ARGS[@]} -ge 5 ] && [ -n "${POS_ARGS[4]}" ] && NUM_VEHICLES=${POS_ARGS[4]}
[ ${#POS_ARGS[@]} -ge 6 ] && [ -n "${POS_ARGS[5]}" ] && NUM_WALKERS=${POS_ARGS[5]}

# Normalize system and tmux socket permissions at startup
chmod 1777 /tmp 2>/dev/null || true
chmod 700 /tmp/tmux-* 2>/dev/null || true
chown root:root /tmp/tmux-0 2>/dev/null || true

# Ensure dedicated non-root carlauser exists with home folder and groups
if ! id "carlauser" &>/dev/null; then
    useradd -m -s /bin/bash carlauser 2>/dev/null || true
fi
usermod -aG video,render,sudo carlauser 2>/dev/null || true
mkdir -p /home/carlauser/.config /home/carlauser/.local/share /home/carlauser/Documents /home/carlauser/Desktop /workspace/carla/CarlaUE4/Saved /tmp/runtime-carlauser
chmod 700 /tmp/runtime-carlauser 2>/dev/null || true
chown -R carlauser:carlauser /home/carlauser /workspace/carla /tmp/runtime-carlauser 2>/dev/null || true
chmod -R 777 /workspace/carla 2>/dev/null || true

# Ensure xdg-user-dir binary is available for Unreal Engine
if ! command -v xdg-user-dir &>/dev/null; then
    cat << 'EOF' > /usr/local/bin/xdg-user-dir
#!/bin/sh
case "$1" in
    DESKTOP) echo "$HOME/Desktop" ;;
    DOCUMENTS) echo "$HOME/Documents" ;;
    DOWNLOAD) echo "$HOME/Downloads" ;;
    *) echo "$HOME" ;;
esac
EOF
    chmod +x /usr/local/bin/xdg-user-dir 2>/dev/null || true
fi

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

# GPU and PyTorch CUDA optimizations
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Verify PyTorch CUDA kernel compatibility with current GPU
echo "--> Verifying PyTorch CUDA compatibility with $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')..."
if ! "$PYTHON_BIN" -c "import torch; x = torch.ones(2, device='cuda'); y = x + 1" >/dev/null 2>&1; then
    echo "⚠️  Current PyTorch build does not have active CUDA kernels for this GPU architecture."
    echo "🔄 Installing official PyTorch CUDA 12.1/12.4 build (Ampere / Ada / RTX A4000 support)..."
    "$PYTHON_BIN" -m pip install --force-reinstall torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121 || \
    "$PYTHON_BIN" -m pip install --force-reinstall torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124 || true
fi

# Ensure required tracking packages exist in target environment
"$PYTHON_BIN" -c "import tensorboard, mlflow" 2>/dev/null || "$PYTHON_BIN" -m pip install tensorboard mlflow 2>/dev/null || true

echo "=============================================================="
echo "   🔄 Starting Autonomous Auto-Restart Training Supervisor    "
echo "=============================================================="
echo "Target Steps: $TOTAL_STEPS | Vision Backbone: $BACKBONE"
echo "Policy Architecture: QWEN-500M (Trainable Attention Skip Connections)"
echo "CARLA Map: $TOWN | Mode: $([ "$START_FRESH" = true ] && echo 'FRESH (From Scratch)' || echo 'RESUME (From Checkpoint)')"
echo "NPC Traffic: $NUM_VEHICLES Vehicles | $NUM_WALKERS Walkers"
echo "Python Executable: $PYTHON_BIN"
echo "=============================================================="

# ==========================================================================
#  📊 Start Persistent MLflow UI Server (survives training crash/restarts)
#  This keeps the MLflow port alive so the Vast.ai tunnel URL never changes.
#  MLflow MUST run in its own tmux session — NOT as a subprocess of
#  train_rl_agent.py — so it is completely isolated from CARLA crashes.
#  The port is auto-detected from Vast.ai's Open Ports configuration.
# ==========================================================================

# Auto-detect an available port from Vast.ai's Open Ports (externally mapped)
# This ensures MLflow uses a port with a stable Cloudflare tunnel.
find_mlflow_port() {
    # Well-known service ports to SKIP (already used by other services)
    SKIP_PORTS="22 1111 2000 2001 2002 6006 8080 8384"

    # Method 1: Parse iptables DNAT rules to find externally-mapped internal ports
    MAPPED_PORTS=$(iptables -t nat -L -n 2>/dev/null | grep DNAT | grep -oP 'to:[^:]+:\K\d+' | sort -un)

    if [ -n "$MAPPED_PORTS" ]; then
        for port in $MAPPED_PORTS; do
            # Skip well-known service ports
            if echo "$SKIP_PORTS" | grep -qw "$port"; then
                continue
            fi
            # Check if port is NOT already in use
            if ! ss -tlnp 2>/dev/null | grep -q ":${port} "; then
                echo "$port"
                return 0
            fi
        done
    fi

    # Method 2: Fallback — try common Vast.ai secondary ports
    for port in 10100 10200 9090 7070 4040; do
        if ! ss -tlnp 2>/dev/null | grep -q ":${port} "; then
            echo "$port"
            return 0
        fi
    done

    # Last resort
    echo "10100"
    return 0
}

# Check if MLflow is already running inside our dedicated tmux session
if tmux has-session -t mlflow_server 2>/dev/null; then
    # Recover the port from the running MLflow process
    MLFLOW_PORT=$(ss -tlnp 2>/dev/null | grep "mlflow\|gunicorn" | grep -oP ':(\d+)' | head -1 | tr -d ':')
    MLFLOW_PORT=${MLFLOW_PORT:-$(find_mlflow_port)}
    echo "✓ MLflow UI server running in protected tmux session 'mlflow_server' (port ${MLFLOW_PORT})."
else
    MLFLOW_PORT=$(find_mlflow_port)

    # Kill any orphaned/unmanaged MLflow processes on this port
    fuser -k ${MLFLOW_PORT}/tcp 2>/dev/null || true
    sleep 1

    echo "--> 📊 Launching persistent MLflow UI server on port ${MLFLOW_PORT} (tmux: mlflow_server)..."
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

# Ensure cloudflared is installed for 1-click public HTTPS dashboard access
if ! command -v cloudflared &>/dev/null; then
    echo "--> Installing cloudflared for direct public HTTPS dashboard access..."
    (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O /tmp/cloudflared.deb 2>/dev/null && \
     dpkg -i /tmp/cloudflared.deb >/dev/null 2>&1 && rm -f /tmp/cloudflared.deb) || true
fi

# Launch or recover Cloudflare public tunnel for MLflow
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
    echo " 🚀 Launching Training Session (Run #$attempt)..."
    echo "--------------------------------------------------------------"

    # 1. Kill ALL stale CARLA and training processes, release GPU memory
    echo "--> Cleaning up stale processes and GPU memory..."
    pkill -9 -f train_rl_agent 2>/dev/null || true
    pkill -u carlauser -9 2>/dev/null || true
    pkill -9 -f CarlaUE4 2>/dev/null || true
    pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null || true
    killall -9 CarlaUE4-Linux-Shipping CarlaUE4 CarlaUE4.sh 2>/dev/null || true
    tmux kill-session -t carla_server 2>/dev/null || true
    fuser -k -9 2000/tcp 2001/tcp 2002/tcp 8000/tcp 2>/dev/null || true
    sleep 2

    # Fix GPU, workspace, and tmux socket permissions
    chmod 1777 /tmp 2>/dev/null || true
    chmod 700 /tmp/tmux-* 2>/dev/null || true
    chown root:root /tmp/tmux-0 2>/dev/null || true
    chmod -R 666 /dev/nvidia* 2>/dev/null || true
    chmod -R 777 /dev/dri 2>/dev/null || true
    chmod -R 777 /workspace/carla 2>/dev/null || true
    if id "carlauser" &>/dev/null; then
        usermod -aG video,render,sudo carlauser 2>/dev/null || true
        chown -R carlauser:carlauser /workspace/carla 2>/dev/null || true
    fi

    # 2. Start clean CARLA Server instance
    if [ -f "/workspace/carla/CarlaUE4.sh" ]; then
        CARLA_USER_ENV="export HOME=/home/carlauser; export USER=carlauser; export XDG_CONFIG_HOME=/home/carlauser/.config; export XDG_DATA_HOME=/home/carlauser/.local/share; export XDG_RUNTIME_DIR=/tmp/runtime-carlauser; export SDL_VIDEODRIVER=offscreen; export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json"

        LOG_FILE="/workspace/carla_server.log"
        touch "$LOG_FILE" && chmod 666 "$LOG_FILE" 2>/dev/null || true
        > "$LOG_FILE" 2>/dev/null || true
        echo "--> Launching CARLA server on port 2000 (tmux: carla_server)..."
        LAUNCH_CMD="/workspace/carla/CarlaUE4.sh -carla-rpc-port=2000 -port=2000 -RenderOffScreen -nosound -quality-level=Low -benchmark -fps=20"

        tmux new-session -d -s carla_server \
            "su -s /bin/bash carlauser -c '$CARLA_USER_ENV; $LAUNCH_CMD' > $LOG_FILE 2>&1"
        sleep 2

        # 3. Verify CARLA is actually responding on port 2000 before launching training
        echo -n "--> Waiting for CARLA RPC server to accept connections on port 2000"
        carla_ready=false
        for i in $(seq 1 45); do
            echo -n "."
            if [ "$i" -ge 4 ]; then
                if ! tmux has-session -t carla_server 2>/dev/null; then
                    echo ""
                    echo "⚠️  CARLA process terminated unexpectedly!"
                    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                        echo "--- Last 20 lines of $LOG_FILE ---"
                        tail -n 20 "$LOG_FILE"
                        echo "-----------------------------------"
                    elif [ -f "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log" ]; then
                        echo "--- Last 20 lines of CarlaUE4.log ---"
                        tail -n 20 "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log"
                        echo "-------------------------------------"
                    fi
                    break
                fi
            fi

            if "$PYTHON_BIN" -W ignore -c "
import sys, glob, socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.4)
res = s.connect_ex(('127.0.0.1', 2000))
s.close()
if res != 0:
    sys.exit(1)
for e in glob.glob('/workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg'):
    if e not in sys.path: sys.path.insert(0, e)
import carla
c = carla.Client('127.0.0.1', 2000)
c.set_timeout(2.0)
v = c.get_server_version()
" 2>/dev/null; then
                carla_ready=true
                echo ""
                echo "✓ CARLA server verified and responding on port 2000!"
                break
            fi
            sleep 1.5
        done

        # If carlauser failed on first attempt, retry once cleanly
        if [ "$carla_ready" = false ]; then
            echo ""
            echo "⚠️  CARLA on port 2000 not responding. Retrying clean launch..."
            tmux kill-session -t carla_server 2>/dev/null || true
            fuser -k -9 2000/tcp 2001/tcp 2002/tcp 2>/dev/null || true
            sleep 2
            touch "$LOG_FILE" && chmod 666 "$LOG_FILE" 2>/dev/null || true
            > "$LOG_FILE" 2>/dev/null || true
            tmux new-session -d -s carla_server \
                "su -s /bin/bash carlauser -c '$CARLA_USER_ENV; $LAUNCH_CMD' > $LOG_FILE 2>&1"
            echo -n "--> Waiting for CARLA server on port 2000 (Retry)"
            for i in $(seq 1 40); do
                echo -n "."
                if [ "$i" -ge 4 ]; then
                    if ! tmux has-session -t carla_server 2>/dev/null; then
                        echo ""
                        echo "⚠️  Retry CARLA process died!"
                        if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                            echo "--- Last 20 lines of $LOG_FILE ---"
                            tail -n 20 "$LOG_FILE"
                            echo "-----------------------------------"
                        elif [ -f "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log" ]; then
                            echo "--- Last 20 lines of CarlaUE4.log ---"
                            tail -n 20 "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log"
                            echo "-------------------------------------"
                        fi
                        break
                    fi
                fi
                if "$PYTHON_BIN" -W ignore -c "
import sys, glob, socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.4)
res = s.connect_ex(('127.0.0.1', 2000))
s.close()
if res != 0:
    sys.exit(1)
for e in glob.glob('/workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg'):
    if e not in sys.path: sys.path.insert(0, e)
import carla
c = carla.Client('127.0.0.1', 2000)
c.set_timeout(2.0)
v = c.get_server_version()
" 2>/dev/null; then
                    carla_ready=true
                    echo ""
                    echo "✓ CARLA server (Retry) verified and responding on port 2000!"
                    break
                fi
                sleep 1.5
            done
        fi

        if [ "$carla_ready" = false ]; then
            echo ""
            echo "⚠️  CARLA server failed to respond after retry. Retrying full restart in 10s..."
            if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                echo "--- Last 25 lines of $LOG_FILE ---"
                tail -n 25 "$LOG_FILE"
                echo "-----------------------------------"
            elif [ -f "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log" ]; then
                echo "--- Last 25 lines of CarlaUE4.log ---"
                tail -n 25 "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log"
                echo "-------------------------------------"
            fi
            sleep 10
            continue
        fi
    fi

    WEIGHTS_ARG=""
    if [ -f "./papers_and_code/LAV/lav_pretrained.pth" ]; then
        WEIGHTS_ARG="--weights-path ./papers_and_code/LAV/lav_pretrained.pth"
    elif [ -f "/workspace/pretrained_carla/model_0030_0.pth" ]; then
        WEIGHTS_ARG="--weights-path /workspace/pretrained_carla/model_0030_0.pth"
    fi

    # Configure resume vs fresh start
    RESUME_ARG="--resume"
    if [ "$START_FRESH" = true ] && [ "$attempt" -eq 1 ]; then
        RESUME_ARG="--fresh"
        echo "🌱 Launching FRESH training run from step 0 (Qwen-500M Decision Policy + LAV Vision Backbone)..."
    else
        echo "🔄 Resuming training from latest saved checkpoint..."
    fi

    # 3. Launch / Resume Training with capped thread allocations and silenced warnings
    export PYTHONWARNINGS="ignore"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    "$PYTHON_BIN" train_rl_agent.py \
        --env-type camera_easycarla \
        --backbone "$BACKBONE" \
        --policy-arch "$POLICY_ARCH" \
        --town "$TOWN" \
        --num-vehicles "$NUM_VEHICLES" \
        --num-walkers "$NUM_WALKERS" \
        --frame-skip 2 \
        --rollout-steps 500 \
        --minibatch-size 256 \
        --ent-coef 0.05 \
        --use-mlflow \
        --mlflow-port $MLFLOW_PORT \
        --log-dir /workspace/runs \
        --checkpoint-dir /workspace/checkpoints \
        $WEIGHTS_ARG \
        $RESUME_ARG \
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
        if [ $exit_code -eq 134 ]; then
            echo "⚠️  Training process terminated with exit code 134 (CARLA C++ abort / UE4 deadlock)."
        elif [ $exit_code -eq 1 ]; then
            echo "⚠️  Training process terminated with exit code 1 (CARLA watchdog triggered clean exit)."
        else
            echo "⚠️  Training process terminated with exit code $exit_code."
        fi
        echo "🔄  Auto-restarting with full CARLA server restart in 10 seconds..."
        echo "    (Checkpoint saved — no training progress lost)"
        sleep 10
        attempt=$((attempt + 1))
    fi
done
