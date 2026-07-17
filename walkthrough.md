# Carla Simulator Headless/Vulkan Running Utility - Walkthrough

This document outlines the files created and explains how to run them to execute the Carla simulator in headless or Vulkan mode.

## Created Files
1. **[carla_runner.py](file:///c:/Users/miari/Desktop/MThesis/carla_runner.py)**: Python script to start, configure, and gracefully stop the Carla simulator process.
2. **[test_connection.py](file:///c:/Users/miari/Desktop/MThesis/test_connection.py)**: Client script using the `carla` library to test connection to a running simulator instance.

## Usage Instructions

### 1. Configure CARLA_ROOT Environment Variable
Set the environment variable pointing to your Carla installation directory on the external drive (see [storage_and_installation_guide.md](file:///c:/Users/miari/Desktop/MThesis/storage_and_installation_guide.md) for details):
```powershell
$env:CARLA_ROOT = "E:\Carla\CarlaSimulator"
```

### 2. Run Carla Simulator in Headless & Vulkan Mode
Execute the runner script. By default, it runs in **headless** mode with the **Vulkan** API and low quality settings:
```powershell
python carla_runner.py --headless --graphics vulkan --port 2000
```
If your executable is in a custom path not automatically detected, you can supply it via `--path`:
```powershell
python carla_runner.py --path "C:\path\to\CarlaUE4.exe" --headless --graphics vulkan
```

### 3. Test Connection
In another terminal, test the connection using the client script:
```powershell
python test_connection.py --port 2000
```
This should output the client and server version, along with the active map name.
