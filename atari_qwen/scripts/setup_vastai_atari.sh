#!/usr/bin/env bash
# Setup script for Atari Qwen Reinforcement Learning on Vast.ai
set -e

echo "================================================================"
echo "    Atari Qwen Reinforcement Learning - Vast.ai Setup Script   "
echo "================================================================"

# 1. Detect Python / venv
if [ -d "/venv/main" ]; then
    echo "--> Activating Vast.ai default environment: /venv/main"
    source /venv/main/bin/activate
    PYTHON_BIN="/venv/main/bin/python"
    PIP_BIN="/venv/main/bin/pip"
else
    PYTHON_BIN="$(which python3)"
    PIP_BIN="$(which pip3)"
fi

echo "--> Python binary: $PYTHON_BIN"
$PYTHON_BIN --version

# 2. System apt dependencies (OpenGL, FFmpeg, etc.)
echo "--> Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    ffmpeg \
    libgl1 \
    libglx-mesa0 \
    libvulkan1 \
    tmux \
    htop \
    git \
    wget > /dev/null

# 3. Python package dependencies
echo "--> Installing Gymnasium, ALE-Py, AutoROM, OpenCV, Tensorboard, MLflow..."
$PIP_BIN install --quiet \
    "gymnasium[atari,accept-rom-license]" \
    "ale-py>=0.8.1" \
    "AutoROM>=0.6.1" \
    "opencv-python-headless" \
    "tensorboard" \
    "mlflow" \
    "pytest"

# 4. Accept ROM license and install Atari ROMs
echo "--> Downloading & installing Atari ROMs via AutoROM..."
AutoROM --accept-license || true

# 5. Validate environment and CUDA
echo "--> Verifying PyTorch and GPU..."
$PYTHON_BIN -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'BF16 Supported: {torch.cuda.is_bf16_supported()}')
"

# 6. Validate Atari Environment
echo "--> Verifying Gymnasium Atari environment..."
$PYTHON_BIN -c "
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('BreakoutNoFrameskip-v4')
obs, _ = env.reset()
print(f'Breakout observation shape: {obs.shape}, action space: {env.action_space}')
env.close()
print('✓ Atari Environment verified!')
"

echo "================================================================"
echo "✓ Vast.ai Setup Complete! You are ready to train."
echo "  Example run:"
echo "    python atari_qwen/training/train_ppo.py --env-id BreakoutNoFrameskip-v4 --preset tiny --num-envs 16"
echo "================================================================"
