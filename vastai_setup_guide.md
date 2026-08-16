# CARLA Simulator & RL Training Guide (Vast.ai Cloud Deployment)

This guide provides step-by-step instructions for running the **CARLA Simulator Engine** natively inside a **Vast.ai PyTorch Development Instance** and training Reinforcement Learning (RL) agents.

---

## 1. Cloud Architecture & Strategy Overview

| Parameter | Configuration / Specification |
| :--- | :--- |
| **Provider & Template** | **Vast.ai** — PyTorch Development Environment (SSH terminal + root access) |
| **Instance Hardware** | Instance `#47104842` — **RTX 3080 (10 GB VRAM)** @ ~$0.088/hr |
| **CARLA Package** | Linux Native Tarball (`CARLA_0.9.15.tar.gz`) stored in `/workspace/carla` |
| **Execution Mode** | Headless native execution (`./CarlaUE4.sh -RenderOffScreen -nosound -vulkan`) |
| **Why Avoid Desktop UI?** | Avoids ~1.5–2 GB unnecessary X11/GUI VRAM overhead |
| **Why Avoid Docker-in-Docker?** | Prevents nested container GPU driver pass-through issues and storage bottlenecks inside Vast.ai |

---

## 2. Step-by-Step Vast.ai Instance Setup

### Step A: Connect to Instance via SSH
Once your Vast.ai instance is started, connect using the provided SSH string:
```bash
ssh -p <PORT> root@<VAST_IP_ADDRESS>
```

### Step B: Download & Extract CARLA 0.9.15 to `/workspace`
Vast.ai mounts persistent disk storage at `/workspace`. Always install CARLA inside `/workspace` so data persists across instance restarts.

```bash
# Create target directory
mkdir -p /workspace/carla
cd /workspace

# Download CARLA 0.9.15 Linux package
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# Extract package to /workspace/carla
tar -xvf CARLA_0.9.15.tar.gz -C /workspace/carla

# Clean up tarball to save disk space
rm CARLA_0.9.15.tar.gz
```

### Step C: Install CARLA Client & RL Dependencies
The PyTorch Development template already has `torch`, `torchvision`, and CUDA pre-configured. Install the remaining requirements:

```bash
pip install carla==0.9.15 gymnasium numpy pillow opencv-python tensorboard
```

### Step D: Set Environment Variable
Add `CARLA_ROOT` to your environment so scripts auto-detect the simulator path:
```bash
export CARLA_ROOT=/workspace/carla
echo "export CARLA_ROOT=/workspace/carla" >> ~/.bashrc
```

---

## 3. Running CARLA Engine Headless

### Option 1: Using `carla_runner.py` (Automated)
Run the included runner script to start CARLA in headless mode with graphics rendering enabled and audio disabled:

```bash
python carla_runner.py --headless --nosound --graphics vulkan --port 2000
```

### Option 2: Direct Command Line (Manual / Background `tmux`)
You can launch CARLA directly inside a `tmux` or `screen` session:

```bash
cd /workspace/carla
./CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low
```

---

## 4. VRAM Budget & Optimization for RTX 3080 (10 GB)

Running both CARLA and PyTorch deep RL training loops (e.g., SAC / PPO) on a single 10 GB VRAM GPU requires careful memory budgeting:

```text
+-------------------------------------------------------------+
|               RTX 3080 VRAM (10 GB Total)                   |
+------------------------------------+------------------------+
|  CARLA 0.9.15 Offscreen Vulkan     |  PyTorch RL Training   |
|  Low Quality Shaders & Physics     |  SAC/PPO Replay Buffer |
|  (~3.8 GB - 4.5 GB)                |  (~4.0 GB - 4.8 GB)    |
+------------------------------------+------------------------+
|             Remaining VRAM Safety Headroom (~1.0 GB)       |
+-------------------------------------------------------------+
```

### Recommended Optimization Guidelines:
1. **Camera Observation Resolution**: Use `160x120` or `256x256` for RGB camera sensors instead of full HD (e.g. `python carla_rl_client_demo.py --img-width 160 --img-height 120`).
2. **Quality Level**: Keep `-quality-level=Low` on the CARLA server to reduce shadow map and texture VRAM allocations.
3. **PyTorch Batch Size**: Use batch sizes of `128` or `256` during policy updates.
4. **Upgrade Path**: If your agent uses multiple cameras/LiDAR or larger CNN backbones (e.g., ResNet-18), upgrade to an **RTX 3090 (24 GB VRAM)** on Vast.ai.

---

## 5. Running the RL Client Demo & Verification

Test your deployment by executing the client script:

```bash
# 1. Verify CARLA Connection
python test_connection.py --host 127.0.0.1 --port 2000

# 2. Run RL Client Demo (50 Steps + Screenshot Capture)
python carla_rl_client_demo.py --host 127.0.0.1 --port 2000 --steps 50 --img-width 256 --img-height 256 --output-dir /workspace/output_screenshots
```

---
