# Running Carla Simulator in Headless or Vulkan Mode via Docker

This plan outlines how we will run and manage the Carla simulator engine inside a Docker container with GPU/Vulkan pass-through on Windows 11 using WSL2, which is ideal for isolating training environments for reinforcement learning.

## User Review Required

> [!WARNING]
> **Windows + Docker GPU Pass-through Limitations**: 
> Running Vulkan inside WSL2 Docker requires:
> 1. Docker Desktop with the WSL2 backend enabled.
> 2. Latest Nvidia Graphics Drivers installed on the Windows host.
> 3. NVIDIA Container Toolkit installed inside the WSL2 kernel.
> 
> If Vulkan drivers fail to pass through cleanly, we may need to fall back to OpenGL or native Windows execution.

## Proposed Changes

We will create a Docker integration within the python runner script to start/stop the simulator using Docker commands.

### 1. Update Carla Runner Utility

#### [MODIFY] [carla_runner.py](file:///c:/Users/miari/Desktop/MThesis/carla_runner.py)
Extend `CarlaRunner` to support running Carla via Docker. It will construct and execute the `docker run` command, mapping the appropriate ports and sharing GPU resources.

Example Docker Command to be generated:
```powershell
docker run --gpus all --privileged --net=host -d carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -vulkan -carla-port=2000 -quality-level=Low
```

### 2. Update Documentation & Guide

#### [NEW] [docker_setup_guide.md](file:///c:/Users/miari/Desktop/MThesis/docker_setup_guide.md)
Document the steps to configure Windows 11, WSL2, Docker Desktop, and NVIDIA Container Toolkit to ensure GPU and Vulkan pass-through work correctly.

## Verification Plan

### Manual Verification
1. Run `docker pull carlasim/carla:0.9.15` to verify access to the image.
2. Launch the simulator using the updated script:
   ```powershell
   python carla_runner.py --use-docker --headless --graphics vulkan --port 2000
   ```
3. Run `python test_connection.py --port 2000` to confirm connection.
