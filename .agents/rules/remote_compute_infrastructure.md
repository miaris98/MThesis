---
trigger: always_on
description: Mandatory pattern for all remote compute scripts. Never hardcode SSH host/port/paths — always use remote_config.py and the .remote file.
---

## Remote Compute Infrastructure Rules

### 1. Box-Agnostic Script Requirement
All scripts in `scripts/setup/` and `scratch/` that touch a remote compute box **MUST** use the central config loader. **Never hardcode SSH host, port, user, workspace path, venv path, or CPU core ranges** anywhere in script source files.

### 2. The Config Layer

**Config file** (git-ignored, one per box): `.remote` at the repository root.
```ini
[remote]
host      = 1.2.3.4
port      = 50011
user      = root
workspace = /workspace/MThesis
venv      = /venv/main/bin
venv_carla = /venv/carla_py38/bin
cpu_cores_a = 16-20
cpu_cores_b = 21-25
cpu_cores_c = 26-31
```

**Template** (committed): `.remote.example` — copy this to `.remote` on every new box.

**Loader module**: `scripts/setup/remote_config.py`

### 3. How to Use in a Script

```python
from remote_config import load_config   # scripts/setup/ scripts: direct import
# OR for scratch/ scripts:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "setup"))
from remote_config import load_config

cfg = load_config()

# SSH command list (pass to subprocess.run)
subprocess.run(cfg.ssh("nvidia-smi"), capture_output=True)

# SSH with stdin pipe (-T flag, no pseudo-TTY)
subprocess.run(cfg.ssh_T("python -"), input=code.encode())

# SCP upload
subprocess.run(cfg.scp("local/file.py", cfg.remote_path("some/file.py")))

# SCP download
subprocess.run(cfg.scp_download("/remote/file.pth", "local/dest.pth"))

# Remote paths under workspace root
cfg.remote_path("results/100k_benchmark")  # → /workspace/MThesis/results/100k_benchmark

# Remote Python interpreter
cfg.python()        # → /venv/main/bin/python
cfg.python_carla()  # → /venv/carla_py38/bin/python

# CPU cores for taskset (Gate 2 parallel variants)
cfg.cpu_cores_a   # → "16-20"
cfg.cpu_cores_b   # → "21-25"
cfg.cpu_cores_c   # → "26-31"

# Convenience: run_ssh prints the command and returns CompletedProcess
cfg.run_ssh("tail -n 20 /workspace/gate1_poc.log", check=False)
```

### 4. Switching Boxes

Edit **`.remote`** only — all 25+ scripts update automatically:
```ini
host = NEW_IP
port = NEW_PORT
```

For CI or one-off overrides, use environment variables (no `.remote` file needed):
```
REMOTE_HOST=x.x.x.x REMOTE_PORT=22 python scripts/setup/foo.py
```

### 5. Creating New Remote Scripts

When writing any new script that SSHes into a remote box:
1. Import `load_config` from `remote_config`.
2. Use `cfg.ssh()`, `cfg.scp()`, `cfg.remote_path()`, etc. — **no raw strings**.
3. Never import `paramiko` in setup/scratch scripts; use `subprocess` + `cfg.ssh_T()` for stdin piping.
4. Do not add `HOST =` or `PORT =` module-level constants.

### 6. Files Modified in the 2026-09-23 Migration

All 25 box-specific scripts were migrated. The complete list:

**`scripts/setup/`**: `remote_commander.py`, `launch_gate1_poc.py`, `launch_gate2_trio.py`, `launch_atari_38k_resumed.py`, `sync_atari_resume.py`, `upload_atari_checkpoints.py`, `upload_carla_checkpoints.py`, `remote_setup_carla_and_mlflow.py`, `check_atari_live.py`, `check_atari_status.py`, `check_atari_steps.py`, `check_50k_eval.py`, `check_gpu.py`, `check_progress.py`, `check_carla_epoch.py`, `check_auto_launch.py`, `check_dataset_and_atari.py`, `check_remote_env.py`, `verify_gate1_ckpt.py`, `inspect_atari_ckpts.py`, `terminate_atari.py`, `test_gym_remote.py`, `test_ssh_fast.py`, `probe_new_box.py`, `monitor_box.py`, `live_status_report.py`, `backup_epoch50_final.py`, `launch_pdm_lite_download.py`

**`scratch/`**: `profile_step.py`, `eval_check.py`, `deploy_resume.py`, `deploy_launch.py`, `check_stack.py`
