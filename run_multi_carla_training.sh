#!/usr/bin/env bash
# ==============================================================================
#  🚀 Multi-CARLA Server Parallel PPO Continuous Auto-Restart Training Supervisor
#  Runs N CARLA simulators concurrently and feeds 100M Qwen Decision Policy
# ==============================================================================

# Default configuration
NUM_ENVS=2
START_PORT=2000
TOTAL_STEPS=70000
POLICY_ARCH=qwen100m
BACKBONE=lav
TOWN=Town10HD_Opt
NUM_VEHICLES=3
NUM_WALKERS=10
START_FRESH=false
REWARD_FN=custom_1
EARLY_STOPPING=true
PATIENCE=20
MIN_DELTA=1.0

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
        --reward-fn=*)
            REWARD_FN="${arg#*=}"
            ;;
        --policy=*)
            POLICY_ARCH="${arg#*=}"
            ;;
        --no-early-stopping)
            EARLY_STOPPING=false
            ;;
        --patience=*)
            PATIENCE="${arg#*=}"
            ;;
        --min-delta=*)
            MIN_DELTA="${arg#*=}"
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

# Detect available GPUs so CARLA servers can be spread across adapters instead
# of every instance defaulting to adapter 0 (Unreal Engine's default).
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
# PyTorch training is pinned to the last GPU so it doesn't silently pile onto
# whichever adapter CARLA server #0 already renders on.
TRAIN_GPU=$((NUM_GPUS - 1))
echo "Detected GPUs: $NUM_GPUS (CARLA servers round-robin across adapters, training pinned to GPU $TRAIN_GPU)"

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
echo "Reward Function:       $REWARD_FN"
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

# Function to test if a cloudflared executable is genuinely valid and working
is_valid_cloudflared() {
    local bin="$1"
    [ -n "$bin" ] && [ -x "$bin" ] && "$bin" --version &>/dev/null
}

# Extract Cloudflare tunnel URL from log file
extract_cf_url() {
    local logfile="$1"
    [ -f "$logfile" ] || return 1
    local url=""
    url=$("$PYTHON_BIN" -c "
import re
try:
    with open('$logfile', 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    m = re.search(r'https://[-a-zA-Z0-9]+\.trycloudflare\.com', txt)
    if m:
        print(m.group(0))
except Exception:
    pass
" 2>/dev/null || true)
    # grep fallback
    if [ -z "$url" ]; then
        url=$(grep -oE 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "$logfile" 2>/dev/null | head -1 || true)
    fi
    [ -n "$url" ] && echo "$url" && return 0
    return 1
}

# Locate or install working cloudflared binary
CLOUDFLARED_BIN=""
for candidate in "$(command -v cloudflared 2>/dev/null)" "/usr/local/bin/cloudflared" "/usr/bin/cloudflared"; do
    if is_valid_cloudflared "$candidate"; then
        CLOUDFLARED_BIN="$candidate"
        break
    fi
done

if [ -z "$CLOUDFLARED_BIN" ]; then
    echo "--> Installing cloudflared binary for public HTTPS dashboard access..."
    rm -f /usr/local/bin/cloudflared /tmp/cloudflared*
    # Try direct binary download first, then .deb package
    if curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o /usr/local/bin/cloudflared 2>/dev/null || \
       wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -O /usr/local/bin/cloudflared 2>/dev/null; then
        chmod +x /usr/local/bin/cloudflared 2>/dev/null || true
    fi
    if is_valid_cloudflared "/usr/local/bin/cloudflared"; then
        CLOUDFLARED_BIN="/usr/local/bin/cloudflared"
    else
        if wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb" -O /tmp/cloudflared.deb 2>/dev/null; then
            dpkg -i /tmp/cloudflared.deb >/dev/null 2>&1 || true
            rm -f /tmp/cloudflared.deb
        fi
        for candidate in "$(command -v cloudflared 2>/dev/null)" "/usr/local/bin/cloudflared"; do
            if is_valid_cloudflared "$candidate"; then
                CLOUDFLARED_BIN="$candidate"
                break
            fi
        done
    fi
fi

# --- Tunnel launch with tmux persistence, reuse + retry on 429 rate-limit ---
CLOUDFLARE_URL=""

# 1) Reuse: check if tmux 'mlflow_tunnel' session is already running with a valid URL
if tmux has-session -t mlflow_tunnel 2>/dev/null && [ -f /tmp/mlflow_tunnel.log ]; then
    CLOUDFLARE_URL=$(extract_cf_url /tmp/mlflow_tunnel.log) || true
    if [ -n "$CLOUDFLARE_URL" ]; then
        echo "✓ Reusing active Cloudflare HTTPS tunnel from tmux session 'mlflow_tunnel': $CLOUDFLARE_URL"
    fi
fi

# Also check cached URL file from a previous successful tunnel
if [ -z "$CLOUDFLARE_URL" ] && [ -f /tmp/mlflow_cf_url ]; then
    CACHED_URL=$(cat /tmp/mlflow_cf_url 2>/dev/null)
    # Verify the cached tunnel is still responding
    if [ -n "$CACHED_URL" ] && curl -s -k --max-time 3 -I "$CACHED_URL" >/dev/null 2>&1; then
        CLOUDFLARE_URL="$CACHED_URL"
        echo "✓ Reusing cached Cloudflare tunnel: $CLOUDFLARE_URL"
    fi
fi

# 2) Launch new tunnel in protected tmux session if none exists
if [ -z "$CLOUDFLARE_URL" ] && [ -n "$CLOUDFLARED_BIN" ]; then
    tmux kill-session -t mlflow_tunnel 2>/dev/null || true
    pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
    sleep 1

    MAX_RETRIES=3
    RETRY_DELAYS=(5 15 30)
    
    for attempt_num in $(seq 1 $MAX_RETRIES); do
        rm -f /tmp/mlflow_tunnel.log
        echo "--> 🌐 Launching Cloudflare HTTPS tunnel in protected tmux 'mlflow_tunnel' (port ${MLFLOW_PORT}) [attempt ${attempt_num}/${MAX_RETRIES}]..."
        tmux new-session -d -s mlflow_tunnel \
            "$CLOUDFLARED_BIN tunnel --url http://127.0.0.1:${MLFLOW_PORT} --no-autoupdate > /tmp/mlflow_tunnel.log 2>&1"

        # Wait up to 15 seconds for URL to appear in log
        for i in $(seq 1 15); do
            if [ -f /tmp/mlflow_tunnel.log ]; then
                CLOUDFLARE_URL=$(extract_cf_url /tmp/mlflow_tunnel.log) || true
                [ -n "$CLOUDFLARE_URL" ] && break 2  # break both loops
            fi
            if ! tmux has-session -t mlflow_tunnel 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # If we get here, this attempt failed — check why
        if [ -f /tmp/mlflow_tunnel.log ] && grep -q "429\|Too Many Requests\|rate" /tmp/mlflow_tunnel.log 2>/dev/null; then
            DELAY=${RETRY_DELAYS[$((attempt_num - 1))]}
            echo "--> ⚠️  Cloudflare rate-limited (429). Waiting ${DELAY}s before retry..."
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
            sleep "$DELAY"
        elif [ -f /tmp/mlflow_tunnel.log ] && ! tmux has-session -t mlflow_tunnel 2>/dev/null; then
            echo "--> [Note] cloudflared exited unexpectedly:"
            head -n 3 /tmp/mlflow_tunnel.log 2>/dev/null | sed 's/^/    /' || true
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
            if [ "$attempt_num" -lt "$MAX_RETRIES" ]; then
                DELAY=${RETRY_DELAYS[$((attempt_num - 1))]}
                echo "--> Retrying in ${DELAY}s..."
                sleep "$DELAY"
            fi
        else
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
            if [ "$attempt_num" -lt "$MAX_RETRIES" ]; then
                echo "--> Timed out. Retrying..."
                sleep 2
            fi
        fi
    done
fi

# 3) Wait for public DNS propagation & verify reachability
if [ -n "$CLOUDFLARE_URL" ]; then
    echo "$CLOUDFLARE_URL" > /tmp/mlflow_cf_url
    echo "--> Verifying Cloudflare public DNS propagation..."
    for i in $(seq 1 10); do
        if curl -s -k -m 3 -I "$CLOUDFLARE_URL" 2>/dev/null | grep -q -E "HTTP/|200|302|301|404|403|502|503"; then
            break
        fi
        sleep 1
    done
fi

echo "=============================================================="
echo "   📊 MLFLOW DASHBOARD ONLINE (PORT ${MLFLOW_PORT})           "
if [ -n "$CLOUDFLARE_URL" ]; then
    echo -e "   👉 \033[1;32mlink to mlflow :     $CLOUDFLARE_URL\033[0m"
else
    echo "   👉 Vast.ai Tunnel:    Open Port ${MLFLOW_PORT} in Vast.ai Tunnels UI"
fi
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
    pkill -u carlauser -9 2>/dev/null || true
    pkill -9 -f CarlaUE4 2>/dev/null || true
    pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null || true
    killall -9 CarlaUE4-Linux-Shipping CarlaUE4 CarlaUE4.sh 2>/dev/null || true
    for ((i=0; i<NUM_ENVS; i++)); do
        tmux kill-session -t "carla_server_${i}" 2>/dev/null || true
        P=${PORTS[$i]}
        fuser -k -9 ${P}/tcp $((P+1))/tcp $((P+2))/tcp 2>/dev/null || true
    done
    sleep 2  # Allow OS & GPU driver to release sockets and Vulkan contexts

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

    # 2. Launch CARLA servers (staggered by 2s to prevent GPU driver race conditions)
    for ((i=0; i<NUM_ENVS; i++)); do
        PORT=${PORTS[$i]}
        SESSION_NAME="carla_server_${i}"
        LOG_FILE="/workspace/carla_server_${PORT}.log"
        GPU_IDX=$((i % NUM_GPUS))
        echo "--> Launching CARLA Server #$((i+1))/$NUM_ENVS on port $PORT (tmux: $SESSION_NAME, GPU $GPU_IDX)..."

        > "$LOG_FILE" 2>/dev/null || true
        tmux new-session -d -s "$SESSION_NAME" \
            "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -vulkan -graphicsadapter=${GPU_IDX} -quality-level=Low -benchmark -fps=20' > ${LOG_FILE} 2>&1"
        sleep 3
    done
    sleep 3

    # 3. Health check all CARLA server instances
    echo "--> Probing all $NUM_ENVS CARLA server instances..."
    all_ready=true
    for ((i=0; i<NUM_ENVS; i++)); do
        PORT=${PORTS[$i]}
        SESSION_NAME="carla_server_${i}"
        LOG_FILE="/workspace/carla_server_${PORT}.log"
        echo -n "   [CARLA #$((i+1))/$NUM_ENVS | Port $PORT] Waiting for server initialization"
        ready=false
        for attempt_check in $(seq 1 45); do
            echo -n "."
            
            # Check if CARLA tmux session closed (command exited/crashed)
            if [ "$attempt_check" -ge 4 ]; then
                if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                    echo ""
                    echo "⚠️  CARLA process on port $PORT terminated unexpectedly!"
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
s.settimeout(0.5)
res = s.connect_ex(('127.0.0.1', $PORT))
s.close()
if res != 0:
    sys.exit(1)
for e in glob.glob('/workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg'):
    if e not in sys.path: sys.path.insert(0, e)
import carla
c = carla.Client('127.0.0.1', $PORT)
c.set_timeout(4.0)
w = c.get_world()
bp = w.get_blueprint_library()
if bp is None:
    sys.exit(1)
" 2>/dev/null; then
                ready=true
                echo ""
                echo "✓ CARLA Server on port $PORT is online and verified!"
                break
            fi
            sleep 1.5
        done

        # If carlauser failed on first attempt, retry once cleanly
        if [ "$ready" = false ]; then
            echo ""
            echo "⚠️  CARLA on port $PORT not responding. Retrying clean launch..."
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
            fuser -k -9 ${PORT}/tcp $((PORT+1))/tcp $((PORT+2))/tcp 2>/dev/null || true
            sleep 2
            tmux new-session -d -s "$SESSION_NAME" \
                "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=${PORT} -RenderOffScreen -nosound -vulkan -graphicsadapter=${GPU_IDX} -quality-level=Low -benchmark -fps=20' > ${LOG_FILE} 2>&1"
            echo -n "   [CARLA #$((i+1))/$NUM_ENVS | Port $PORT (Retry)] Waiting for initialization"
            for attempt_check in $(seq 1 40); do
                echo -n "."
                if [ "$attempt_check" -ge 4 ]; then
                    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                        echo ""
                        echo "⚠️  Retry CARLA process on port $PORT died!"
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
res = s.connect_ex(('127.0.0.1', $PORT))
s.close()
if res != 0:
    sys.exit(1)
for e in glob.glob('/workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg'):
    if e not in sys.path: sys.path.insert(0, e)
import carla
c = carla.Client('127.0.0.1', $PORT)
c.set_timeout(2.0)
v = c.get_server_version()
" 2>/dev/null; then
                    ready=true
                    echo ""
                    echo "✓ CARLA Server on port $PORT (Retry) is online and verified!"
                    break
                fi
                sleep 1.5
            done
        fi

        if [ "$ready" = false ]; then
            echo ""
            echo "⚠️  CARLA Server on port $PORT failed to start."
            if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
                echo "--- Last 25 lines of $LOG_FILE ---"
                tail -n 25 "$LOG_FILE"
                echo "-----------------------------------"
            elif [ -f "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log" ]; then
                echo "--- Last 25 lines of CarlaUE4.log ---"
                tail -n 25 "/workspace/carla/CarlaUE4/Saved/Logs/CarlaUE4.log"
                echo "-------------------------------------"
            fi
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
    # Pin PyTorch to its own GPU so it doesn't default to cuda:0 and pile onto
    # whichever adapter CARLA server #0 is already rendering on.
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"

    "$PYTHON_BIN" -W ignore train_rl_agent.py \
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
        --ent-coef 0.005 \
        --reward-fn "$REWARD_FN" \
        --use-mlflow \
        --mlflow-port "$MLFLOW_PORT" \
        --log-dir /workspace/runs \
        --checkpoint-dir /workspace/checkpoints \
        $WEIGHTS_ARG \
        $RESUME_ARG \
        $([ "$EARLY_STOPPING" = true ] && echo "--early-stopping --early-stopping-patience $PATIENCE --early-stopping-min-delta $MIN_DELTA" || echo "--no-early-stopping") \
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
