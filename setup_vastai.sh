#!/usr/bin/env bash
# ==============================================================================
# CARLA 0.9.15 & Deep RL Environment Automated Setup Script for Vast.ai
# ==============================================================================
# Sets up system packages, CARLA 0.9.15, Python 3.8 Conda environment, Jupyter kernel,
# monitoring tools (nvitop, btop, nvtop, htop), environment variables, and launches
# CARLA server in a background tmux session.
# ==============================================================================

set -e

# --- Color Formatting ---
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}==============================================================${NC}"
echo -e "${CYAN}   🚀 Starting Automated CARLA & RL Setup on Vast.ai        ${NC}"
echo -e "${CYAN}==============================================================${NC}"

# --- Check Root Permissions ---
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[WARNING] Not running as root. Some apt-get commands may require sudo.${NC}"
fi

# --- 1. System Packages & Monitoring Utilities ---
echo -e "\n${CYAN}[1/7] Installing system libraries and monitoring tools...${NC}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y

# Install standard packages (using modern libgl1 and libglx-mesa0 for Ubuntu 22.04 / 24.04 Noble)
apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libvulkan1 \
    ffmpeg \
    tmux \
    wget \
    git \
    curl \
    htop \
    btop 2>/dev/null || apt-get install -y ffmpeg tmux wget git curl htop

# Try installing libtiff versions safely (libtiff6 on Ubuntu 24.04, libtiff5 on 20.04/22.04, or libtiff-dev)
apt-get install -y libtiff6 2>/dev/null || apt-get install -y libtiff5 2>/dev/null || apt-get install -y libtiff-dev 2>/dev/null || true

# Try installing nvtop if available in repository
apt-get install -y nvtop 2>/dev/null || true

echo -e "${GREEN}✓ System dependencies installed successfully.${NC}"

# --- 2. Download & Extract CARLA 0.9.15 ---
echo -e "\n${CYAN}[2/7] Checking CARLA 0.9.15 Simulator installation...${NC}"
CARLA_DIR="/workspace/carla"

if [ -f "$CARLA_DIR/CarlaUE4.sh" ]; then
    echo -e "${GREEN}✓ CARLA 0.9.15 already exists at $CARLA_DIR. Skipping download.${NC}"
else
    echo -e "${YELLOW}--> Downloading CARLA 0.9.15 tarball (~16 GB uncompressed)...${NC}"
    mkdir -p "$CARLA_DIR"
    cd /workspace
    
    wget -c https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz -O /workspace/CARLA_0.9.15.tar.gz
    
    echo -e "${YELLOW}--> Extracting CARLA package to $CARLA_DIR...${NC}"
    tar -xvf /workspace/CARLA_0.9.15.tar.gz -C "$CARLA_DIR"
    
    echo -e "${YELLOW}--> Cleaning up tarball to save disk space...${NC}"
    rm -f /workspace/CARLA_0.9.15.tar.gz
    echo -e "${GREEN}✓ CARLA 0.9.15 extracted successfully.${NC}"
fi

# --- 3. Non-Root User Setup for Unreal Engine ---
echo -e "\n${CYAN}[3/7] Setting up dedicated 'carlauser' for Unreal Engine execution...${NC}"
if id "carlauser" &>/dev/null; then
    echo -e "${GREEN}✓ User 'carlauser' already exists.${NC}"
else
    useradd -m -s /bin/bash carlauser
    echo -e "${GREEN}✓ Created user 'carlauser'.${NC}"
fi
chown -R carlauser:carlauser "$CARLA_DIR"

# --- 4. Python 3.8 Conda Environment & Dependencies ---
echo -e "\n${CYAN}[4/7] Setting up Python 3.8 Conda environment ('carla_py38')...${NC}"

# Locate or auto-install Conda
CONDA_PROFILE=""
for p in "/opt/conda" "/workspace/miniconda" "$HOME/miniconda3" "$HOME/anaconda3" "/root/miniconda3" "/usr/local/miniconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
        CONDA_PROFILE="$p/etc/profile.d/conda.sh"
        break
    fi
done

if [ -z "$CONDA_PROFILE" ] && ! command -v conda &>/dev/null; then
    echo -e "${YELLOW}--> Conda not detected. Installing Miniconda to /workspace/miniconda...${NC}"
    mkdir -p /workspace/miniconda
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda.sh
    bash /workspace/miniconda.sh -b -u -p /workspace/miniconda
    rm -f /workspace/miniconda.sh
    CONDA_PROFILE="/workspace/miniconda/etc/profile.d/conda.sh"
    /workspace/miniconda/bin/conda init bash
fi

if [ -n "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    if ! grep -q "$CONDA_PROFILE" ~/.bashrc; then
        echo "source $CONDA_PROFILE" >> ~/.bashrc
    fi
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
fi

# Install nvitop globally if pip exists
pip install nvitop 2>/dev/null || true

if conda env list 2>/dev/null | grep -q "carla_py38"; then
    echo -e "${GREEN}✓ Conda environment 'carla_py38' already exists.${NC}"
else
    echo -e "${YELLOW}--> Creating conda environment 'carla_py38' with Python 3.8...${NC}"
    conda create -n carla_py38 python=3.8 -y
fi

conda activate carla_py38 2>/dev/null || source /opt/conda/bin/activate carla_py38 2>/dev/null || source activate carla_py38 2>/dev/null

echo -e "${YELLOW}--> Installing Python libraries & PyTorch...${NC}"
conda install -y ipykernel
python -m ipykernel install --user --name=carla_py38 --display-name "Python 3.8 (CARLA RL)"

pip install --upgrade pip
pip install "setuptools<80" gymnasium numpy pillow opencv-python tensorboard torch torchvision jupyterlab ipywidgets nvitop

echo -e "${GREEN}✓ Python environment and Jupyter kernel configured.${NC}"

# --- 5. Configure Environment Variables ---
echo -e "\n${CYAN}[5/7] Configuring environment variables (~/.bashrc)...${NC}"
EGG_PATH=$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg 2>/dev/null | tail -n 1)

if ! grep -q "CARLA_ROOT=/workspace/carla" ~/.bashrc; then
    echo 'export CARLA_ROOT=/workspace/carla' >> ~/.bashrc
fi

if ! grep -q "PythonAPI/carla/dist" ~/.bashrc; then
    echo 'export PYTHONPATH=$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg 2>/dev/null | tail -n 1):/workspace/carla/PythonAPI/carla:$PYTHONPATH' >> ~/.bashrc
fi

export CARLA_ROOT=/workspace/carla
if [ -n "$EGG_PATH" ]; then
    export PYTHONPATH="$EGG_PATH:/workspace/carla/PythonAPI/carla:$PYTHONPATH"
fi

echo -e "${GREEN}✓ Environment variables configured.${NC}"

# --- 6. Launch CARLA Server in Background (tmux) ---
echo -e "\n${CYAN}[6/7] Checking and launching CARLA Server...${NC}"
if tmux has-session -t carla_server 2>/dev/null; then
    echo -e "${GREEN}✓ CARLA server tmux session ('carla_server') is already running.${NC}"
else
    echo -e "${YELLOW}--> Starting headless CARLA server in tmux session 'carla_server'...${NC}"
    tmux new-session -d -s carla_server "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low' > /workspace/carla_server.log 2>&1"
    echo -e "${YELLOW}--> Waiting 10 seconds for CARLA engine initialization...${NC}"
    sleep 10
fi

# --- 7. Verify Connection ---
echo -e "\n${CYAN}[7/7] Verifying connection to CARLA Server...${NC}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -f "$SCRIPT_DIR/test_connection.py" ]; then
    python "$SCRIPT_DIR/test_connection.py" --host 127.0.0.1 --port 2000 || {
        echo -e "${YELLOW}[NOTE] Server is still starting up. Check logs with: tail -n 20 /workspace/carla_server.log${NC}"
    }
fi

echo -e "\n${GREEN}==============================================================${NC}"
echo -e "${GREEN}   🎉 SETUP COMPLETE! You are ready to train RL agents!      ${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo -e "Quick Commands:"
echo -e "  1. Launch 3-Pane Dashboard:  ${CYAN}bash $SCRIPT_DIR/start_dashboard.sh${NC}"
echo -e "  2. Train PPO Agent:          ${CYAN}python $SCRIPT_DIR/train_rl_agent.py --total-steps 2000${NC}"
echo -e "  3. Start TensorBoard:        ${CYAN}tensorboard --logdir=/workspace/runs --port=6006 --host=0.0.0.0 &${NC}"
echo -e "  4. View CARLA Server Logs:   ${CYAN}tail -f /workspace/carla_server.log${NC}"
echo -e "==============================================================\n"
