# CARLA Simulator & PPO Training Guide (Vast.ai Jupyter Template End-to-End)

This guide provides an **end-to-end, copy-pasteable workflow** for setting up a brand new **Vast.ai instance** (using the default Jupyter Notebook / PyTorch template), launching the CARLA 0.9.15 simulator headlessly, training a PPO Deep RL Agent, monitoring training metrics via TensorBoard & Jupyter Notebooks, and tracking system resources (GPU/CPU/RAM) with interactive tools like **`nvitop`**, **`btop`**, and **`tmux`**.

---

## 1. Instance Creation Requirements (Vast.ai Console)

When renting an instance on Vast.ai:
* **Template**: Select **PyTorch** or **Jupyter Notebook** (Default Docker image: `pytorch/pytorch` or Vast PyTorch image).
* **GPU Selection**: Choose an **RTX 3080, RTX 3090, or RTX 4090** (At least **10 GB VRAM**).
* **Disk Space Allocation**: Set container disk allocation to **at least 40 GB** (CARLA 0.9.15 uncompressed is ~16 GB + PyTorch/dependencies).

---

## 2. How to Access Your New Instance

You have **two ways** to access your instance:

### Option A: Via JupyterLab Web Interface (Easiest)
1. In the Vast.ai **Instances** dashboard, click **"Open"** or **"Jupyter"**.
2. Once JupyterLab opens in your browser, go to **File** $\to$ **New** $\to$ **Terminal** (or click the **Terminal** tile on the Launcher page).

### Option B: Via SSH Command Line
Copy the SSH connection command provided in your Vast.ai instance dashboard and run it in Windows PowerShell or Terminal:
```bash
ssh -p <PORT> root@<VAST_IP_ADDRESS>
```

---

## 3. One-Time End-to-End Setup (Run inside Terminal)

### Option A: 1-Command Automated Setup (Recommended)
Clone the repository and run the automated setup script to configure packages, CARLA, Python 3.8, Jupyter kernel, monitoring tools, and launch the CARLA server:

```bash
cd /workspace
git clone https://github.com/miaris98/MThesis.git
cd /workspace/MThesis
bash setup_vastai.sh
```

---

### Option B: Step-by-Step Manual Setup

If you prefer executing the steps manually, run the following commands sequentially:

#### Step 3.1: Install System Shared Libraries, Monitoring Tools & Utilities
```bash
apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libvulkan1 \
    ffmpeg \
    tmux \
    wget \
    git \
    curl \
    htop \
    btop

# Install libtiff (libtiff6 on Ubuntu 24.04 Noble, libtiff5 on older releases)
apt-get install -y libtiff6 2>/dev/null || apt-get install -y libtiff5 2>/dev/null || apt-get install -y libtiff-dev 2>/dev/null || true
[ -f /usr/lib/x86_64-linux-gnu/libtiff.so.6 ] && ln -sf /usr/lib/x86_64-linux-gnu/libtiff.so.6 /usr/lib/x86_64-linux-gnu/libtiff.so.5 && ldconfig
apt-get install -y nvtop 2>/dev/null || true
```


#### Step 3.2: Download & Extract CARLA 0.9.15 to `/workspace`
> `/workspace` is the persistent storage volume on Vast.ai.
```bash
mkdir -p /workspace/carla
cd /workspace

# Download CARLA 0.9.15 Linux package
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# Extract package to /workspace/carla (takes ~1-2 mins)
tar -xvf CARLA_0.9.15.tar.gz -C /workspace/carla

# Remove tarball to save disk space
rm CARLA_0.9.15.tar.gz
```

#### Step 3.3: Create Python 3.8 Environment & Install Dependencies
CARLA 0.9.15 requires **Python 3.8**.
```bash
# 1. Create Python 3.8 conda environment
conda create -n carla_py38 python=3.8 -y

# 2. Activate environment
source /opt/conda/bin/activate carla_py38

# 3. Register environment as a Jupyter Kernel
conda install -y ipykernel
python -m ipykernel install --user --name=carla_py38 --display-name "Python 3.8 (CARLA RL)"

# 4. Install PyTorch & RL packages (pin setuptools<80 for CARLA compatibility)
pip install "setuptools<80" gymnasium gym numpy pillow opencv-python tensorboard torch torchvision jupyterlab ipywidgets scipy matplotlib

# 5. Install nvitop & EasyCarla-RL package directly from GitHub
pip install nvitop
pip install git+https://github.com/silverwingsbot/EasyCarla-RL.git
```

#### Step 3.4: Configure Environment Variables
```bash
export CARLA_ROOT=/workspace/carla
export PYTHONPATH=$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg | tail -n 1):/workspace/carla/PythonAPI/carla:$PYTHONPATH

echo 'export CARLA_ROOT=/workspace/carla' >> ~/.bashrc
echo 'export PYTHONPATH=$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg | tail -n 1):/workspace/carla/PythonAPI/carla:$PYTHONPATH' >> ~/.bashrc
```


---

## 4. Launch CARLA Engine Headless in Background & Save Logs

Unreal Engine security rules refuse to launch directly as `root`. We create a dedicated non-root user `carlauser` and launch CARLA in a background `tmux` window while saving log outputs to `/workspace/carla_server.log`.

### Step 4.1: Create Non-Root User & Set Permissions
```bash
useradd -m -s /bin/bash carlauser
chown -R carlauser:carlauser /workspace/carla
```

### Step 4.2: Start CARLA in a Background `tmux` Session with Logging
```bash
tmux new-session -d -s carla_server "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low' > /workspace/carla_server.log 2>&1"
```

### Step 4.3: Verify CARLA Connection & View Logs
Wait ~10 seconds for the server to load, then test the connection:
```bash
source /opt/conda/bin/activate carla_py38
cd /workspace/MThesis
python test_connection.py --host 127.0.0.1 --port 2000
```
*Expected Output*: `Successfully connected to CARLA Server! Map: Town10HD_Opt`

* **To check CARLA server logs live**:
  ```bash
  tail -f /workspace/carla_server.log
  ```

---

## 5. GPU, CPU, RAM & Process Monitoring Tools

To monitor your system resources while running CARLA and PyTorch side-by-side:

| Tool | Focus | Command | When to Use |
| :--- | :--- | :--- | :--- |
| **`nvitop`** | **PyTorch & CUDA** | `nvitop` | **Best for RL**: Shows GPU memory footprint per PyTorch script, active CUDA kernels, and process IDs. |
| **`btop`** | **System Dashboard** | `btop` | **Best All-In-One**: Beautiful interactive UI for CPU core loads, System RAM, GPU, and Disk I/O. |
| **`nvtop`** | **GPU Charts** | `nvtop` | Visual real-time graphs of GPU compute usage, VRAM consumption, and power draw over time. |
| **`htop`** | **CPU / RAM** | `htop` | Standard lightweight monitor for CPU threads, memory usage, and background processes. |

### Background GPU Usage Logging to CSV
If you want to log GPU memory and compute usage automatically over time for post-training analysis:
```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv -l 5 > /workspace/gpu_usage.csv &
```

---

## 6. Recommended `tmux` Multi-Pane Monitoring Dashboard

Instead of switching between terminal tabs, set up a 3-pane `tmux` dashboard session to view **RL Training**, **GPU Memory (`nvitop`)**, and **System Resources (`btop`)** simultaneously.

### Option A: One-Command Automated Script (Recommended)
You can run the included helper script directly inside your repository:
```bash
bash /workspace/MThesis/start_dashboard.sh
```

### Option B: Manual Command Sequence
```bash
# Create a new tmux session named 'dashboard'
tmux new-session -d -s dashboard

# Split session vertically into two main panes
tmux split-window -v -t dashboard

# Split bottom pane horizontally into two sub-panes
tmux split-window -h -t dashboard.1

# Pane 0 (Top): Prepare for RL Training
tmux send-keys -t dashboard.0 "source /opt/conda/bin/activate carla_py38 && cd /workspace/MThesis" C-m

# Pane 1 (Bottom-Left): Launch nvitop for GPU monitoring
tmux send-keys -t dashboard.1 "nvitop" C-m

# Pane 2 (Bottom-Right): Launch btop for CPU/RAM monitoring
tmux send-keys -t dashboard.2 "btop" C-m

# Attach to the multi-pane dashboard
tmux attach -t dashboard
```

> **`tmux` Navigation Cheat-Sheet**:
> * Switch between panes: Press `Ctrl + B`, then use **Arrow Keys**.
> * Detach from dashboard (keep everything running): Press `Ctrl + B`, then `D`.
> * Re-attach to dashboard later: `tmux attach -t dashboard`.

---

## 7. Train Camera-Only PPO Deep RL Agent

Run the camera-only PPO continuous control policy training loop in CARLA (inside Pane 0 or a standard terminal):

### Option A: ImageNet Pretrained ResNet Vision Backbone (Default)
```bash
source /opt/conda/bin/activate carla_py38
cd /workspace/MThesis

# Run Camera-Only PPO RL Training with Pretrained ResNet-18 Vision Backbone
python train_rl_agent.py \
    --env-type camera_easycarla \
    --backbone resnet18 \
    --freeze-backbone \
    --total-steps 2000 \
    --rollout-steps 250 \
    --log-dir /workspace/runs \
    --checkpoint-dir /workspace/checkpoints
```

### Option B: CARLA Domain Pretrained Vision Weights (TransFuser++ / TCP)
To use a vision encoder specifically pretrained on millions of CARLA driving frames:
```bash
# Download CARLA-pretrained model weights (TransFuser++ / Leaderboard 2.0)
mkdir -p /workspace/pretrained_carla
wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip -O /workspace/pretrained_carla/models.zip
unzip /workspace/pretrained_carla/models.zip -d /workspace/pretrained_carla/

# Run PPO training using CARLA pretrained vision weights
python train_rl_agent.py \
    --env-type camera_easycarla \
    --backbone resnet34 \
    --weights-path /workspace/pretrained_carla/model_0030_0.pth \
    --freeze-backbone \
    --total-steps 2000
```

---

## 8. Monitor Training Metrics with TensorBoard

### Step 8.1: Start TensorBoard Server
Inside your terminal, launch TensorBoard in the background:
```bash
tensorboard --logdir=/workspace/runs --port=6006 --host=0.0.0.0 &
```

### Step 8.2: Open TensorBoard in Local Browser
* **Option A (Vast.ai Open Ports)**: In the Vast.ai dashboard, click **"Open Port"** for `6006`.
* **Option B (SSH Tunnel)**: Run on local Windows PowerShell:
  ```powershell
  ssh -p <PORT> -L 6006:localhost:6006 root@<VAST_IP>
  ```
  Then navigate to `http://localhost:6006` in your local web browser to view real-time Policy Loss, Value Loss, and Episode Rewards.

---

## 9. Record & Play Evaluation Video in Jupyter Notebook

### Step 9.1: Record Evaluation Video
Generate multi-view video (RGB + Depth + Semantic Segmentation) with traffic:
```bash
source /opt/conda/bin/activate carla_py38
cd /workspace/MThesis
python record_eval_video.py --host 127.0.0.1 --port 2000 --steps 150 --output-video /workspace/output_screenshots/driving_multiview.mp4
```

### Step 9.2: Play Video inside Jupyter Notebook Cell
1. In JupyterLab, open a new `.ipynb` notebook.
2. Select Kernel: **`Python 3.8 (CARLA RL)`** (top right dropdown).
3. Paste and run the following code cell:

```python
from IPython.display import Video

# Play recorded driving video inline
Video('/workspace/output_screenshots/driving_multiview.mp4', embed=True, width=720)
```

---

## 10. Quick Cheat-Sheet for Subsequent Server Restarts

If your Vast.ai instance is paused/restarted, the environment is preserved in `/workspace`. Run these commands to resume:

```bash
# 1. Activate environment & Navigate to repo
source /opt/conda/bin/activate carla_py38
cd /workspace/MThesis
git pull

# 2. Start CARLA Server in background with logging
tmux new-session -d -s carla_server "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low' > /workspace/carla_server.log 2>&1"

# 3. Check connection
python test_connection.py

# 4. Launch nvitop or btop in background monitor window
nvitop

# 5. Start PPO Training
python train_rl_agent.py --total-steps 5000
```
