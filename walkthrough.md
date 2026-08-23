# Walkthrough - Hardware Telemetry & Dashboard Layout Optimization

We have updated [start_dashboard.sh](file:///c:/Users/miari/Desktop/MThesis/start_dashboard.sh) and [train_rl_agent.py](file:///c:/Users/miari/Desktop/MThesis/train_rl_agent.py) to restore the classic 3-pane monitoring layout and record hardware telemetry (GPU memory, CPU load, System RAM) in step CSV logs and MLflow dashboard metrics.

---

## 1. Restored 3-Pane `tmux` Dashboard Layout

In [start_dashboard.sh](file:///c:/Users/miari/Desktop/MThesis/start_dashboard.sh):
* **Pane 0 (Top)**: Main training command terminal. Displays clickable MLflow URL notice (`http://<PUBLIC_IP>:5055`).
* **Pane 1 (Bottom-Left)**: **`nvitop`** interactive monitor (shows GPU VRAM, CUDA compute kernels, and PyTorch memory allocations).
* **Pane 2 (Bottom-Right)**: **`btop` / `htop`** monitor (shows CPU cores, System RAM, Swap memory, and Disk I/O).
* **Single MLflow UI Daemon**: `train_rl_agent.py` manages background MLflow server auto-launch without taking up a tmux pane or running duplicate instances.

---

## 2. Hardware Metrics Added to CSV & MLflow

[train_rl_agent.py](file:///c:/Users/miari/Desktop/MThesis/train_rl_agent.py) now queries real-time hardware telemetry at every step:

* **Recorded Fields in `training_telemetry.csv`**:
  * `gpu_mem_used_mb`: Active VRAM memory allocated by PyTorch.
  * `gpu_mem_pct`: VRAM memory allocation ratio ($0\text{--}100\%$).
  * `sys_cpu_pct`: Total system CPU load percentage.
  * `sys_ram_used_gb`: Active system RAM memory used.
* **MLflow & TensorBoard Dashboard Metrics**:
  * `Hardware/GPU_Memory_MB`
  * `Hardware/CPU_Usage_Pct`
  * `Hardware/RAM_Used_GB`

---

## 3. Verification

- **Syntax & Compilation Test:** `train_rl_agent.py` passed with **0 errors**.
