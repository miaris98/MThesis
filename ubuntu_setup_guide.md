# CARLA Simulator & RL Training Guide (Ubuntu Linux)

This guide provides complete, end-to-end instructions for running the **CARLA Simulator Engine** (via Docker with GPU Vulkan pass-through or native Linux execution) and training Reinforcement Learning (RL) agents with **MLflow Tracking (Port 5055)** and **Step-Level CSV Telemetry Logging** on **Ubuntu Linux**.

---

## 1. System Prerequisites & NVIDIA GPU Drivers

Ensure your Ubuntu system has updated NVIDIA drivers and GPU Container tools installed.

### A. Install NVIDIA Drivers & Vulkan Utilities
```bash
sudo apt update
sudo apt install -y build-essential vulkan-tools libvulkan1 nvidia-driver-535
```
*Verify GPU installation by running `nvidia-smi`.*

### B. Install Docker Engine & NVIDIA Container Toolkit (For Docker Mode)
```bash
# 1. Install Docker Engine
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# 2. Configure NVIDIA Container Toolkit
distribution=$(. /os-release 2>/dev/null || . /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```
*Verify Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`*

---

## 2. Setting Up the Conda Environment (`carla_rl`)

Set up a dedicated Python Conda environment for your RL client dependencies:

```bash
# 1. Create Conda environment
conda create -n carla_rl python=3.8 -y

# 2. Activate environment
conda activate carla_rl

# 3. Install matching CARLA 0.9.15 client, RL & MLflow tracking dependencies
pip install --upgrade pip
pip install "setuptools<80" carla==0.9.15 numpy pillow opencv-python torch torchvision gymnasium tensorboard mlflow nvitop
```

---

## 3. Running CARLA Server on Ubuntu

Choose either **Docker Execution (Containerized)** or **Native Execution**.

### Option A: Running CARLA via Docker with GPU Vulkan (Recommended)

1. **Pull the official CARLA 0.9.15 image:**
   ```bash
   docker pull carlasim/carla:0.9.15
   ```

2. **Launch CARLA in offscreen Vulkan mode with GPU acceleration:**
   ```bash
   docker run -d \
     --name carla_server_2000 \
     --privileged \
     --gpus all \
     -p 2000-2002:2000-2002 \
     carlasim/carla:0.9.15 \
     ./CarlaUE4.sh -carla-port=2000 -RenderOffScreen -vulkan -quality-level=Low
   ```

3. **Verify running container status:**
   ```bash
   docker ps
   ```

---

### Option B: Running CARLA Natively on Ubuntu

1. **Download & Extract CARLA 0.9.15 Linux Package:**
   ```bash
   mkdir -p ~/carla_simulator
   cd ~/carla_simulator
   wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
   tar -xvf CARLA_0.9.15.tar.gz
   ```

2. **Set Environment Variable & Launch:**
   ```bash
   export CARLA_ROOT=~/carla_simulator
   python carla_runner.py --headless --graphics vulkan --port 2000
   ```

---

## 4. Train PPO Agent with MLflow (Port 5055) & Step Telemetry Logging

Launch multi-camera vision PPO training with **MLflow (Port 5055)** and **Step-Level CSV Telemetry**:

```bash
conda activate carla_rl
cd ~/MThesis

# Run PPO training with LAV Pretrained Weights, MLflow (Port 5055) and CSV Telemetry
python train_rl_agent.py \
    --env-type camera_easycarla \
    --backbone lav \
    --weights-path ./papers_and_code/LAV/lav_pretrained.pth \
    --minibatch-size 128 \
    --use-mlflow \
    --mlflow-port 5055 \
    --total-steps 500000
```

### MLflow Web UI Dashboard
Open your web browser at `http://<UBUNTU_IP>:5055` to view real-time reward curves, policy losses, CARLA driving score estimates ($DS_{\text{est}}$), and download `training_telemetry.csv`.

---

## 5. Artifact Output & Verification

- **Step Telemetry CSV**: Saved automatically to `runs/training_telemetry.csv` and synced as an MLflow artifact.
- **Evaluation Video**: Run `python record_eval_video.py --checkpoint checkpoints/ppo_carla_best.pth --steps 300` to generate driving videos with telemetry overlays.
- **Connection Test**: Run `python test_connection.py --port 2000` anytime to verify server connection.
