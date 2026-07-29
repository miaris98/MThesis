# CARLA Simulator & RL Training Guide (Ubuntu Linux)

This guide provides complete, end-to-end instructions for running the **CARLA Simulator Engine** (via Docker with GPU Vulkan pass-through or native Linux execution) and training Reinforcement Learning (RL) agents on **Ubuntu Linux**.

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
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```
*Verify Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`*

---

## 2. Setting Up the Conda Environment (`carla_rl`)

Set up a dedicated Python 3.10 Conda environment for your RL client dependencies:

```bash
# 1. Create Conda environment
conda create -n carla_rl python=3.10 -y

# 2. Activate environment
conda activate carla_rl

# 3. Install matching CARLA 0.9.15 client and RL dependencies
pip install carla==0.9.15 numpy pillow opencv-python torch gymnasium
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

## 4. Running the RL Agent Demo & Capturing Screenshots

With the CARLA server running on port 2000 and the `carla_rl` Conda environment activated:

```bash
conda activate carla_rl
python carla_rl_client_demo.py --host 127.0.0.1 --port 2000 --steps 50 --output-dir output_screenshots
```

### Expected Output Console:
```text
Connecting to CARLA server at 127.0.0.1:2000...
Successfully connected to Carla Simulator!
Client Version: 0.9.15
Server Version: 0.9.15
Spawned Vehicle 'vehicle.tesla.model3' (ID: 104) at Location(x=120.5, y=-45.2, z=0.5)
Screenshots will be saved to: /home/user/MThesis/output_screenshots

--- Starting RL Control Loop ---
[Step 01/50] Action(T=0.6, S=+0.02) | Speed:  12.4 km/h | Reward: +1.20 | Total Reward: +1.20 | Collision: False | Frame Saved: step_001.png
[Step 02/50] Action(T=0.6, S=+0.04) | Speed:  21.8 km/h | Reward: +2.10 | Total Reward: +3.30 | Collision: False | Frame Saved: step_002.png
...
[Step 50/50] Action(T=0.6, S=-0.05) | Speed:  45.1 km/h | Reward: +4.41 | Total Reward: +185.30 | Collision: False | Frame Saved: step_050.png

--- Simulation Episode Finished ---
Total Cumulative Reward: 185.30
Cleanup completed.
```

---

## 5. Artifact Output & Verification

- **Camera Screenshots**: Saved as PNG files in [output_screenshots/](file:///c:/Users/miari/Desktop/MThesis/output_screenshots).
- **RL Reward Tracking**: Logs per-step speed incentives, steering smoothness penalties, and collision event penalties.
- **Connection Test**: Run `python test_connection.py --port 2000` anytime to verify connection status.
