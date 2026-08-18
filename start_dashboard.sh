#!/bin/bash
# Script to launch automated 3-pane monitoring & training dashboard in tmux

SESSION_NAME="dashboard"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Attaching to existing '$SESSION_NAME' session..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

echo "Creating new 3-pane monitoring dashboard..."

# 1. Create main session
tmux new-session -d -s $SESSION_NAME

# 2. Split vertically into two main sections (Top & Bottom)
tmux split-window -v -t $SESSION_NAME

# 3. Split bottom pane horizontally into two sub-panes (Left & Right)
tmux split-window -h -t $SESSION_NAME.1

# 4. Pane 0 (Top): Set up Python 3.8 environment for RL Training
tmux send-keys -t $SESSION_NAME.0 "source /opt/conda/bin/activate carla_py38 && cd /workspace/MThesis" C-m

# 5. Pane 1 (Bottom-Left): Launch nvitop for GPU/PyTorch monitoring
tmux send-keys -t $SESSION_NAME.1 "nvitop" C-m

# 6. Pane 2 (Bottom-Right): Launch btop or htop for CPU/RAM monitoring
if command -v btop &> /dev/null; then
    tmux send-keys -t $SESSION_NAME.2 "btop" C-m
else
    tmux send-keys -t $SESSION_NAME.2 "htop" C-m
fi

echo "Dashboard setup complete! Attaching now..."
tmux attach-session -t $SESSION_NAME
