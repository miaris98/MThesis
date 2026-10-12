#!/bin/bash
# Script to launch dedicated Hardware Diagnostics Dashboard (nvitop GPU + btop CPU/RAM)

SESSION_NAME="dashboard"
MLFLOW_PORT=10100

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Attaching to existing '$SESSION_NAME' diagnostics session..."
    if [ -n "$TMUX" ]; then
        tmux switch-client -t $SESSION_NAME
    else
        tmux attach -t $SESSION_NAME
    fi
    exit 0
fi

echo "Creating dedicated System Diagnostics Dashboard (nvitop GPU | btop CPU/RAM)..."

# 1. Create main session (mouse off allows normal browser text selection & copying)
tmux new-session -d -s $SESSION_NAME
tmux set-option -t $SESSION_NAME mouse off
tmux set-option -g mouse off

# 2. Split vertically into two monitoring panes (Top: nvitop | Bottom: btop)
tmux split-window -v -t $SESSION_NAME

# Find Conda activation path dynamically
CONDA_ACTIVATE="conda activate carla_py38"
for p in "/workspace/miniconda" "/opt/conda" "$HOME/miniconda3" "$HOME/anaconda3" "/root/miniconda3" "/usr/local/miniconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
        CONDA_ACTIVATE="source $p/etc/profile.d/conda.sh && conda activate carla_py38"
        break
    fi
done

# Detect Public IP
PUBLIC_IP=$(curl -s https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')

# 3. Pane 0 (Top): Launch nvitop for GPU VRAM & CUDA process monitoring
tmux send-keys -t $SESSION_NAME.0 "$CONDA_ACTIVATE && nvitop || watch -n 1 nvidia-smi" C-m

# 4. Pane 1 (Bottom): Launch btop or htop for CPU/RAM system monitoring
if command -v btop &> /dev/null; then
    tmux send-keys -t $SESSION_NAME.1 "btop" C-m
else
    tmux send-keys -t $SESSION_NAME.1 "htop" C-m
fi

echo "Diagnostics Dashboard setup complete!"
echo "----------------------------------------------------------------"
echo "  📊 MLFLOW WEB DASHBOARD URL:"
if [ -f /tmp/mlflow_tunnel.log ]; then
    CLOUDFLARE_URL=$(python -c "import re; txt=open('/tmp/mlflow_tunnel.log','r').read(); m=re.search(r'https://[a-zA-Z0-9.-]+\.trycloudflare\.com', txt); print(m.group(0) if m else '')" 2>/dev/null || true)
    if [ -n "$CLOUDFLARE_URL" ]; then
        echo -e "  👉 \033[1;32mlink to mlflow :     $CLOUDFLARE_URL\033[0m"
    fi
fi
echo "  👉 http://$PUBLIC_IP:$MLFLOW_PORT"
echo "  👉 http://127.0.0.1:$MLFLOW_PORT"
echo "----------------------------------------------------------------"
if [ -n "$TMUX" ]; then
    tmux switch-client -t $SESSION_NAME
else
    tmux attach -t $SESSION_NAME
fi
