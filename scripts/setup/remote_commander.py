import subprocess
import sys
from remote_config import load_config

cfg = load_config()


def run_ssh(cmd, check=True, timeout=None):
    print(f"--> [SSH] {cmd}")
    res = subprocess.run(cfg.ssh(cmd), capture_output=True, text=True, timeout=timeout)
    if check and res.returncode != 0:
        print(f"[ERROR] returncode {res.returncode}:")
        print(res.stderr)
        raise RuntimeError(f"Command failed: {cmd}\n{res.stderr}")
    return res.stdout.strip()


def check_status():
    print("=== System Status ===")
    out = run_ssh("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
    print("GPU:", out)
    out = run_ssh("lscpu | grep 'Model name' || uname -m")
    print("CPU:", out)
    out = run_ssh("df -h /")
    print("Disk:\n", out)


if __name__ == "__main__":
    check_status()
