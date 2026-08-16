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

### Step C: Create Python 3.8 Environment & Install Dependencies
> **Crucial Note**: CARLA 0.9.15 C++ Boost bindings were natively compiled for **Python 3.7 and 3.8**. Set up a dedicated Python 3.8 environment:

```bash
# 1. Install missing C++ shared libraries (libtiff5)
apt-get update && (apt-get install -y libtiff5 2>/dev/null || apt-get install -y libtiff-dev 2>/dev/null)

# 2. Create and activate Python 3.8 Conda environment
conda create -n carla_py38 python=3.8 -y
conda activate carla_py38  # (or source /opt/conda/bin/activate carla_py38)

# 3. Install RL dependencies & PyTorch (pin setuptools<80 for CARLA's pkg_resources dependency)
pip install "setuptools<80" gymnasium numpy pillow opencv-python tensorboard torch torchvision
```

### Step D: Set Environment Variables
Add `CARLA_ROOT` and `PYTHONPATH` to your environment so Python auto-loads the CARLA 0.9.15 API:

```bash
export CARLA_ROOT=/workspace/carla
export PYTHONPATH=$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg | tail -n 1):/workspace/carla/PythonAPI/carla:$PYTHONPATH

echo "export CARLA_ROOT=/workspace/carla" >> ~/.bashrc
echo "export PYTHONPATH=\$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg | tail -n 1):/workspace/carla/PythonAPI/carla:\$PYTHONPATH" >> ~/.bashrc
```

### Step E: Clone & Sync your GitHub Repository
Your repository (`miaris98/MThesis`) is **Public**, which means you can clone it directly without setting up SSH keys or authentication tokens:

```bash
cd /workspace
git clone https://github.com/miaris98/MThesis.git
cd /workspace/MThesis
```

> **Development Workflow**:
> 1. Edit code on your local PC, then commit & push to GitHub:
>    ```bash
>    git add . && git commit -m "Updated RL agent" && git push
>    ```
> 2. On your Vast.ai SSH terminal, pull updates instantly:
>    ```bash
>    cd /workspace/MThesis && git pull
>    ```
> *(Note: If you ever change the repo visibility to Private in the future, clone using a Personal Access Token: `git clone https://<PAT_TOKEN>@github.com/miaris98/MThesis.git`)*

---

## 3. Running CARLA Engine Headless

> **Important Note for Vast.ai (Root Execution)**:
> Unreal Engine security rules refuse to run directly as `root` (`Stderr: Refusing to run with the root privileges.`), and Docker containers on Vast.ai restrict user namespaces (`unshare`).
> - `carla_runner.py` automatically detects `root`, creates a `carlauser` account, and launches CARLA via `su` (no manual configuration required).
> - For manual binary execution as `root`, create a non-root user and run via `su carlauser`.

### Option 1: Using `carla_runner.py` (Automated - Recommended)
Run the included runner script to start CARLA in headless mode with graphics rendering enabled and audio disabled (auto-creates `carlauser` & bypasses root restrictions):

```bash
python carla_runner.py --headless --nosound --graphics vulkan --port 2000
```

### Option 2: Direct Command Line (Manual / Background `tmux`)
You can launch CARLA directly inside a `tmux` or `screen` session:

```bash
# 1. One-time setup: Create non-root user & fix folder ownership
useradd -m -s /bin/bash carlauser
chown -R carlauser:carlauser /workspace/carla

# 2. Launch CARLA as carlauser
su carlauser -c "/workspace/carla/CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low"
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

## 6. Video Rendering & Interactive JupyterLab Visualization

### Step A: Generate MP4 Driving Video from Screenshots
Convert captured simulation frames into an MP4 video using `ffmpeg`:

```bash
apt-get install -y ffmpeg
ffmpeg -framerate 10 -i /workspace/output_screenshots/step_%03d.png -c:v libx264 -pix_fmt yuv420p /workspace/output_screenshots/driving_demo.mp4
```

### Step B: Launch JupyterLab on Vast.ai
Start JupyterLab inside the `carla_py38` environment:

```bash
conda activate carla_py38
pip install jupyterlab ipywidgets
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```

### Step C: Access JupyterLab & View Media Inline
1. **Local SSH Tunnel** (run in local Windows PowerShell):
   ```powershell
   ssh -p <PORT> -L 8888:localhost:8888 root@<VAST_IP>
   ```
2. Open `http://localhost:8888` in your local web browser.
3. Display screenshots and play videos directly inside notebook cells:

```python
# Display Screenshot
from IPython.display import Image, display
display(Image(filename='/workspace/output_screenshots/step_025.png'))

# Play MP4 Video Inline
from IPython.display import Video
Video('/workspace/output_screenshots/driving_demo.mp4', embed=True, width=640)
```

---

## 7. Deep RL Agent Training & Multi-Sensor Traffic Simulation

### Step A: Train PPO Deep RL Agent with PyTorch & TensorBoard
Run full Actor-Critic PPO policy updates with NatureCNN feature extraction:

```bash
conda activate carla_py38
python train_rl_agent.py --host 127.0.0.1 --port 2000 --total-steps 1000 --rollout-steps 250 --log-dir /workspace/runs --checkpoint-dir /workspace/checkpoints
```

### Step B: Monitor Training Metrics via TensorBoard
1. **Launch TensorBoard on Vast.ai**:
   ```bash
   tensorboard --logdir=/workspace/runs --port=6006 --host=0.0.0.0
   ```
2. **Local SSH Tunnel** (run in local Windows PowerShell):
   ```powershell
   ssh -p <PORT> -L 6006:localhost:6006 root@<VAST_IP>
   ```
3. Open `http://localhost:6006` to monitor Policy Loss, Value Loss, Episode Rewards, and Driving Speeds in real time.

### Step C: Multi-Sensor Traffic Evaluation & Split-Screen Video
Spawn 20+ NPC AI vehicles via `TrafficManager` and record synchronized RGB + Depth + Semantic Segmentation multi-view video:

```bash
python record_eval_video.py --host 127.0.0.1 --port 2000 --steps 150 --output-video /workspace/output_screenshots/driving_multiview.mp4
```
