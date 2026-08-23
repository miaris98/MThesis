#!/bin/bash
# Script to launch automated 3-pane monitoring & training dashboard in tmux with MLflow (Port 5055)

SESSION_NAME="dashboard"
MLFLOW_PORT=5055

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Attaching to existing '$SESSION_NAME' session..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

echo "Creating new 3-pane monitoring & MLflow dashboard (Port $MLFLOW_PORT)..."

# 1. Create main session with mouse support enabled
tmux new-session -d -s $SESSION_NAME
tmux set-option -t $SESSION_NAME mouse on
tmux set-option -g mouse on

# 2. Split vertically into two main sections (Top & Bottom)
tmux split-window -v -t $SESSION_NAME

# 3. Split bottom pane horizontally into two sub-panes (Left & Right)
tmux split-window -h -t $SESSION_NAME.1

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

# 4. Pane 0 (Top): Set up Python 3.8 environment & display MLflow URL
tmux send-keys -t $SESSION_NAME.0 "$CONDA_ACTIVATE && cd /workspace/MThesis && echo '====================================================' && echo '   📊 MLFLOW DASHBOARD PORT: $MLFLOW_PORT' && echo '   👉 Public URL: http://$PUBLIC_IP:$MLFLOW_PORT' && echo '===================================================='" C-m

# 5. Pane 1 (Bottom-Left): Launch MLflow UI background server on port 5055 or nvitop
tmux send-keys -t $SESSION_NAME.1 "$CONDA_ACTIVATE && mlflow ui --host 0.0.0.0 --port $MLFLOW_PORT" C-m

# 6. Pane 2 (Bottom-Right): Launch nvitop for GPU/PyTorch monitoring
tmux send-keys -t $SESSION_NAME.2 "$CONDA_ACTIVATE && nvitop || watch -n 1 nvidia-smi" C-m

echo "Dashboard setup complete!"
echo "----------------------------------------------------------------"
echo "  📊 MLFLOW WEB DASHBOARD URL:"
echo "  👉 http://$PUBLIC_IP:$MLFLOW_PORT"
echo "  👉 http://127.0.0.1:$MLFLOW_PORT"
echo "----------------------------------------------------------------"
tmux attach-session -t $SESSION_NAME
