# External Storage & Carla Installation Guide

Due to limited space on the primary partition (`C:`), all major environments, dependencies, and the Carla Simulator must be installed on your external storage drive (e.g., `D:`, `E:`, `F:`, or `G:`).

## IMPORTANT: Carla Version & RL Codebase Clarification

- **`E:\RL_CARLA-main.zip`**: This is **not** the Carla Simulator itself. It is a repository containing the Reinforcement Learning agent (SAC) and the Gym environment wrapper.
- **Carla Simulator Engine**: You must download the actual simulator separately. 
- **Version Compatibility**: While the `RL_CARLA` codebase was originally written for **Carla 0.9.6** (which is outdated and requires legacy Python 3.5/3.6), we recommend downloading **Carla 0.9.15** (or **0.9.16**) for Windows 11. Modern versions provide native Windows 11 stability, better Vulkan graphics driver performance, and compatibility with modern Python versions (3.7 / 3.8 / 3.10) for your RL libraries.

## 1. Directory Structure on External Storage

We recommend creating a unified directory structure on your external drive:
```text
<ExternalDrive>:/
└── Carla/
    ├── CarlaSimulator/         # Extract the Carla release package here
    └── RL_Environments/        # Place training checkpoints and large datasets here
```

## 2. Setting up the Carla Environment Variable

After extracting Carla on your external storage drive, configure your terminal or user environment variables so the scripts can locate it automatically.

### Temporary Setup (Current Terminal Session)
In PowerShell:
```powershell
$env:CARLA_ROOT = "E:\Carla\CarlaSimulator"  # Replace E with your actual external drive letter
```

In Command Prompt:
```cmd
set CARLA_ROOT=E:\Carla\CarlaSimulator
```

### Permanent Setup (System-wide)
1. Search for **"Edit the system environment variables"** in the Windows Start menu.
2. Click **Environment Variables...**
3. Under **User variables**, click **New...**
4. Set **Variable name** to `CARLA_ROOT` and **Variable value** to your installation directory (e.g., `E:\Carla\CarlaSimulator`).
5. Restart your terminal or IDE for the change to take effect.

## 3. How the Runner Script Detects the Installation
The runner script `carla_runner.py` is configured to automatically check for installations on drives `C:`, `D:`, `E:`, `F:`, and `G:` under `<Drive>:\Carla` or `<Drive>:\carla`. If installed in these directories, it will launch without requiring command-line configuration.
