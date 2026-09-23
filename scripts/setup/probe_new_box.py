"""probe_new_box.py — Probe SSH connectivity and report system info.

Reads host/port from .remote. To override for a different box, use:
    REMOTE_HOST=1.2.3.4 REMOTE_PORT=22 python probe_new_box.py
"""
import subprocess
import sys
from remote_config import load_config

cfg = load_config()


def probe_ssh():
    print(f"Testing SSH connection to {cfg.host}:{cfg.port}...")
    cmd = [
        "ssh", "-p", cfg.port,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{cfg.user}@{cfg.host}",
        "uname -a && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader && df -h / && ls -la /workspace"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print(f"SUCCESS connecting to {cfg.host}:{cfg.port}!")
            print("--- Output ---")
            print(res.stdout)
            return cfg.host, cfg.port
        else:
            print(f"Failed with returncode {res.returncode}:")
            print(res.stderr)
    except Exception as e:
        print(f"Exception connecting to {cfg.host}:{cfg.port}: {e}")
    return None, None


if __name__ == "__main__":
    host, port = probe_ssh()
    if not host:
        sys.exit(1)
