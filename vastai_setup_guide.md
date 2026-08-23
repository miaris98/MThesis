# CARLA Simulator & PPO Training Guide (Vast.ai Jupyter Template End-to-End)

This guide provides an **end-to-end, copy-pasteable workflow** for setting up a brand new **Vast.ai instance** (using the default Jupyter Notebook / PyTorch template), launching CARLA 0.9.15 headlessly, training a PPO Deep RL Agent, tracking experiment metrics with **MLflow (Port 10100)** & **TensorBoard**, logging step-by-step CSV telemetry (`training_telemetry.csv`), and tracking GPU/CPU system resources with interactive tools like **`nvitop`**, **`btop`**, and **`tmux`**.

---

## 1. Instance Creation Requirements (Vast.ai Console)

When renting an instance on Vast.ai:
* **Template**: Select **PyTorch** or **Jupyter Notebook** (Default Docker image: `pytorch/pytorch` or Vast PyTorch image).
* **GPU Selection**: Choose an **RTX 3060 (12GB), RTX 3080, RTX 3090, or RTX 4090** (At least **10 GB VRAM**).
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
Clone the repository and run the automated setup script to configure system packages, CARLA, Python 3.8, Jupyter kernel, MLflow, and launch the CARLA server:

```bash
cd /workspace
git clone https://github.com/miaris98/MThesis.git
cd /workspace/MThesis
bash setup_vastai.sh
```

---

### Option B: Step-by-Step Manual Setup

If you prefer executing the steps manually, run the following commands sequentially:

#### Step 3.1: Install System Shared Libraries & Monitoring Utilities
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

# 4. Install PyTorch, RL packages & MLflow tracking (pin setuptools<80)
pip install --upgrade pip
pip install "setuptools<80" gymnasium gym numpy pillow opencv-python tensorboard mlflow torch torchvision jupyterlab ipywidgets nvitop scipy matplotlib

# 5. Install EasyCarla-RL package directly from GitHub
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

## 4. Launch CARLA Engine Headless in Background

### Step 4.1: Create Non-Root User & Set Permissions
```bash
useradd -m -s /bin/bash carlauser
chown -R carlauser:carlauser /workspace/carla
```

### Step 4.2: Start CARLA in a Background `tmux` Session
```bash
tmux new-session -d -s carla_server "su carlauser -c '/workspace/carla/CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low' > /workspace/carla_server.log 2>&1"
```

### Step 4.3: Verify Connection
```bash
source /opt/conda/bin/activate carla_py38
cd /workspace/MThesis
python test_connection.py --host 127.0.0.1 --port 2000
```
*Expected Output*: `Successfully connected to CARLA Server! Map: Town10HD_Opt`

---

## 5. MLflow Tracking Dashboard (Port 10100)

When you launch `train_rl_agent.py`, the `ExperimentLogger` automatically starts an MLflow tracking server on **port 10100** and outputs clickable URLs to stdout:

```text
======================================================================
   📊 MLFLOW DASHBOARD ONLINE (PORT 10100)
   👉 Clickable Public URL:  http://<YOUR_VAST_IP>:10100
   👉 Localhost URL:         http://127.0.0.1:10100
   ✓ Experiment: 'CARLA_PPO_RL' | Run ID: 4f8b9d...
======================================================================
```

To start the MLflow server manually at any time:
```bash
source /opt/conda/bin/activate carla_py38
mlflow ui --host 0.0.0.0 --port 10100
```

---

## 6. Multi-Pane `tmux` Monitoring Dashboard

Launch the automated 3-pane monitoring & MLflow dashboard:
```bash
bash /workspace/MThesis/start_dashboard.sh
```

---

## 7. Train PPO Agent with LAV Pretrained Weights & VRAM Acceleration

Run high-throughput PPO RL training utilizing your GPU's VRAM:

```bash
source /opt/conda/bin/activate carla_py38
cd /workspace/MThesis

# Train 1,000,000 steps with LAV Pretrained Encoder, GPU Tensor Core batching (128), and MLflow
python train_rl_agent.py \
    --env-type camera_easycarla \
    --backbone lav \
    --weights-path ./papers_and_code/LAV/lav_pretrained.pth \
    --freeze-backbone \
    --minibatch-size 128 \
    --use-mlflow \
    --mlflow-port 10100 \
    --total-steps 1000000 \
    --log-dir /workspace/runs \
    --checkpoint-dir /workspace/checkpoints
```

### Auto-Restart Supervisor (Resilient to Crash / Timeouts)
```bash
bash run_training_loop.sh 1000000 lav Town10HD_Opt 3 10
```

---

## 8. CSV Telemetry Export & Model Artifacts

* **Step-by-Step CSV Log**: All step inputs, outputs, actions, speed, raw rewards, sub-rewards, and curriculum parameters are written to `/workspace/runs/training_telemetry.csv`.
* **MLflow Artifact Sync**: `training_telemetry.csv` and best model checkpoints (`ppo_carla_best.pth`) are automatically uploaded as MLflow artifacts and downloadable from the MLflow UI (`http://<VAST_IP>:10100`).

---

## 9. Record Evaluation Video

Generate evaluation videos with real-time driving telemetry overlays:
```bash
python record_eval_video.py \
    --checkpoint /workspace/checkpoints/ppo_carla_best.pth \
    --backbone lav \
    --steps 300 \
    --output-video /workspace/output_screenshots/driving_multiview.mp4
```

In JupyterLab, play the recorded video cell:
```python
from IPython.display import Video
Video('/workspace/output_screenshots/driving_multiview.mp4', embed=True, width=720)
```
