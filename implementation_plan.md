# Running Carla Simulator in Headless or Vulkan Mode

This plan outlines the steps to build a utility or script to start/manage the Carla simulator engine in headless (off-screen rendering) or Vulkan modes, which is essential for training reinforcement learning (RL) models efficiently.

## User Review Required

> [!IMPORTANT]
> - **Carla Installation Path (External Storage)**: Due to disk space limitations on `C:`, Carla Simulator must be installed on the external storage drive (e.g., `E:\Carla\CarlaSimulator`).
> - **Carla Version Compatibility**: Although the RL wrapper codebase (`RL_CARLA-main.zip`) was originally targeted for Carla 0.9.6, we will target Carla 0.9.15 (or 0.9.16) for Windows 11 compatibility and modern Python support.
> - **GPU/Driver Requirements**: Headless running via Vulkan/Off-Screen rendering requires correct display driver configurations (especially on Windows vs Linux, though we are on Windows here).

## Proposed Changes

We will create a python runner/utility script to manage the Carla simulator process.

### Carla Runner Utility

#### [NEW] [carla_runner.py](file:///c:/Users/miari/Desktop/MThesis/carla_runner.py)
Create a helper script that allows starting, monitoring, and stopping the Carla simulator with custom command-line options.

Key flags to support:
- `-RenderOffScreen`: Run offscreen (headless)
- `-vulkan` / `-dx12`: Select graphics API
- `-carla-port=PORT`: Custom port configuration
- `-quality-level=Low`: Lower quality settings to speed up simulator ticks during RL training

## Verification Plan

### Manual Verification
- Execute `python carla_runner.py --headless --vulkan --port 2000` and check if the Carla process starts without opening a window and responds on port 2000.
- Verify using a simple client connection script (e.g., `carla.Client('localhost', 2000)`) to check simulator connectivity.
