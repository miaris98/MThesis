# Running Carla Simulator in Headless or Vulkan Mode via Docker & Native Setup

This plan outlines how to execute and manage the CARLA simulator engine using Docker (headless with `-nullrhi` or WSL2 GPU pass-through) or native execution on Windows 11 for reinforcement learning (RL) training.

## User Review Required & Key Findings

> [!WARNING]
> **Windows Docker GPU Pass-through Technical Finding**:
> 1. Running standard Docker commands from Windows PowerShell directly can hit GPU driver isolation issues in Unreal Engine.
> 2. **Docker Headless Mode (`-nullrhi`)**: Runs lightweight CARLA Docker containers for physics, actions, state, and step rewards without requiring GPU 3D rendering drivers inside the container.
> 3. **Docker 3D GPU Mode (`WSL2 Ubuntu`)**: Mounts host WSL GPU drivers (`-v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib`) to enable full 3D Vulkan/RGB camera rendering inside Docker.

## Proposed Changes

We provide Python utilities and documentation for both Docker modes and native execution.

### 1. CARLA Runner Utility & RL Client

#### [MODIFY] [carla_runner.py](file:///c:/Users/miari/Desktop/MThesis/carla_runner.py)
Utility to manage CARLA simulator execution via Docker or native execution, setting ports, headless mode, and graphics API flags.

#### [NEW] [carla_rl_client_demo.py](file:///c:/Users/miari/Desktop/MThesis/carla_rl_client_demo.py)
Client script that connects to CARLA, spawns an actor with RGB camera and collision sensors, executes steering/throttle actions, computes step rewards, and logs camera screenshots.

### 2. Documentation & Setup Guides

#### [MODIFY] [storage_and_installation_guide.md](file:///c:/Users/miari/Desktop/MThesis/storage_and_installation_guide.md)
Document external drive (`E:\`) storage configuration, `carla_rl` Conda environment setup (Python 3.10), and Docker storage relocation.

#### [MODIFY] [walkthrough.md](file:///c:/Users/miari/Desktop/MThesis/walkthrough.md)
Complete guide for starting CARLA Docker containers, native execution, Conda setup, and running the RL step-reward demo.

## Verification Plan

### Manual Verification
1. Verify CARLA Docker container startup in headless (`-nullrhi`) or WSL2 GPU mode.
2. Activate `carla_rl` Conda environment.
3. Run `python carla_rl_client_demo.py --host 127.0.0.1 --port 2000 --steps 50` to confirm connection, action execution, step reward calculations, and screenshot generation.
