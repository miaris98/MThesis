# External Storage & Carla Installation Guide

Due to limited space on the primary partition (`C:`), all major environments, dependencies, Docker data, and the Carla Simulator are configured on your external storage drive (`E:`).

## IMPORTANT: Carla Version & RL Codebase Clarification

- **`E:\RL_CARLA-main.zip`**: This is **not** the Carla Simulator itself. It is a repository containing the Reinforcement Learning agent (SAC) and the Gym environment wrapper.
- **Carla Simulator Engine**: You must download or run the actual simulator image/package (`carlasim/carla:0.9.15`).
- **Version Compatibility**: While the legacy `RL_CARLA` codebase was originally written for **Carla 0.9.6** (legacy Python 3.5/3.6), we use **Carla 0.9.15** (or **0.9.16**) for Windows 11. Modern versions provide native Windows 11 stability, better Vulkan graphics driver performance, and compatibility with **Python 3.10**.

## 1. Directory Structure on External Storage

Unified directory structure on external drive `E:\`:
```text
E:/
├── DockerWSLData/              # Relocated Docker Desktop WSL2 virtual disk
└── Carla/
    ├── CarlaSimulator/         # Native Carla release extraction (if running natively)
    └── RL_Environments/        # Training checkpoints and large datasets
```

## 2. Docker Storage Relocation (`E:\DockerWSLData`)
Docker Desktop's virtual disk is imported and stored on `E:\DockerWSLData` so pulled Docker images (such as `carlasim/carla:0.9.15`) do not consume `C:` drive space.

## 3. Conda Environment Setup (`carla_rl`)

Create and activate the `carla_rl` environment with Python 3.10:

```powershell
# Create environment with Python 3.10
conda create -n carla_rl python=3.10 -y

# Activate environment
conda activate carla_rl

# Install dependencies
pip install -r requirements.txt
```

## 4. Native Execution Setup (`CARLA_ROOT`)

If running CARLA natively on `E:\`:

### Temporary Setup (Current Terminal Session)
In PowerShell:
```powershell
$env:CARLA_ROOT = "E:\Carla\CarlaSimulator"
```

### Permanent Setup (System-wide)
1. Search for **"Edit the system environment variables"** in Windows.
2. Click **Environment Variables...**
3. Under **User variables**, add `CARLA_ROOT` with value `E:\Carla\CarlaSimulator`.

## 5. Automatic Detection in `carla_runner.py`
`carla_runner.py` automatically scans drives `C:`, `D:`, `E:`, `F:`, and `G:` under `<Drive>:\Carla` or `<Drive>:\carla`.
