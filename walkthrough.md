# Carla Simulator Headless/Vulkan Running Utility - Walkthrough

This document outlines the files created and explains how to set up the Conda environment, run the CARLA simulator server in headless/Vulkan mode via Docker (or natively), send vehicle control actions, capture camera screenshots, and compute step rewards.

## Created Files
1. **[carla_runner.py](file:///c:/Users/miari/Desktop/MThesis/carla_runner.py)**: Python utility script to start, configure, and stop the CARLA simulator (supports both native execution and Docker containers).
2. **[carla_rl_client_demo.py](file:///c:/Users/miari/Desktop/MThesis/carla_rl_client_demo.py)**: Client script to connect to CARLA, spawn a vehicle with RGB camera and collision sensors, step through throttle/steer control actions, compute RL rewards, and save camera screenshots.
3. **[environment.yml](file:///c:/Users/miari/Desktop/MThesis/environment.yml)** & **[requirements.txt](file:///c:/Users/miari/Desktop/MThesis/requirements.txt)**: Conda environment definition (Python 3.10) and pip package dependencies.
4. **[test_connection.py](file:///c:/Users/miari/Desktop/MThesis/test_connection.py)**: Client script to test basic connection and fetch map/server metadata.

---

## 1. Setting Up the Conda Environment (`carla_rl`)

Create and activate a dedicated Conda environment named `carla_rl` with **Python 3.10**:

### Option A: Using Conda CLI
```powershell
conda create -n carla_rl python=3.10 -y
conda activate carla_rl
pip install -r requirements.txt
```

### Option B: Using environment.yml
```powershell
conda env create -f environment.yml
conda activate carla_rl
```

---

## 2. Running CARLA Server in Docker (Headless + Vulkan)

To launch the official CARLA server inside a Docker container with GPU hardware acceleration (Vulkan) and offscreen rendering:

```powershell
python carla_runner.py --docker --headless --graphics vulkan --port 2000
```

This automatically runs:
`docker run --name carla_server_2000 --rm --gpus all -p 2000-2002:2000-2002 carlasim/carla:0.9.15 ./CarlaUE4.sh -carla-port=2000 -RenderOffScreen -vulkan -quality-level=Low`

---

## 3. Running CARLA Server Natively (Optional Alternative)

If running directly on Windows external drive `E:\`:

```powershell
$env:CARLA_ROOT = "E:\Carla\CarlaSimulator"
python carla_runner.py --headless --graphics vulkan --port 2000
```

---

## 4. RL Client Demo: Actions, Screenshots & Reward Output

With the `carla_rl` Conda environment activated, run the client demo:

```powershell
python carla_rl_client_demo.py --host 127.0.0.1 --port 2000 --steps 50 --output-dir output_screenshots
```

### Example Console Output:
```text
Connecting to CARLA server at 127.0.0.1:2000...
Spawned Vehicle 'vehicle.tesla.model3' (ID: 104) at Location(x=120.5, y=-45.2, z=0.5)
Screenshots will be saved to: C:\Users\miari\Desktop\MThesis\output_screenshots

--- Starting RL Control Loop ---
[Step 01/50] Action(T=0.6, S=+0.02) | Speed:  12.4 km/h | Reward: +1.20 | Total Reward: +1.20 | Collision: False | Frame Saved: step_001.png
[Step 02/50] Action(T=0.6, S=+0.04) | Speed:  21.8 km/h | Reward: +2.10 | Total Reward: +3.30 | Collision: False | Frame Saved: step_002.png
...
[Step 50/50] Action(T=0.6, S=-0.05) | Speed:  45.1 km/h | Reward: +4.41 | Total Reward: +185.30 | Collision: False | Frame Saved: step_050.png

--- Simulation Episode Finished ---
Total Cumulative Reward: 185.30
Cleanup completed.
```

### Inspecting Captured Output:
- All step camera images will be saved in `output_screenshots/step_001.png`, `step_002.png`, etc.
- Step rewards reflect speed incentives, steering smooth penalties, and collision penalties.
