"""vast_fleet.py — Inspect active Vast.ai instances and configure .remote

Usage:
    python scripts/setup/vast_fleet.py list
    python scripts/setup/vast_fleet.py select <idx_or_id> [--direct]
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path

VAST_CLI = r"C:\Users\miari\anaconda3\envs\graphtools\Scripts\vastai.exe"
if not Path(VAST_CLI).exists():
    VAST_CLI = shutil.which("vastai") or "vastai"

SSH_KEY = r"C:\Users\miari\.ssh\id_ed25519"


def get_instances():
    res = subprocess.run([VAST_CLI, "show", "instances", "--raw"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error calling vastai: {res.stderr}")
        return []
    try:
        return json.loads(res.stdout)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []


def test_ssh(host, port):
    if not host or not port:
        return False, "No host/port"
    cmd = ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=4",
           "-p", str(port), f"root@{host}", "nvidia-smi --query-gpu=name --format=csv,noheader"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
    if res.returncode == 0:
        lines = res.stdout.strip().splitlines()
        return True, f"OK ({len(lines)} GPUs: {lines[0]})"
    return False, (res.stderr.strip() or res.stdout.strip() or "Connection failed")


def list_instances():
    data = get_instances()
    if not data:
        print("No active Vast.ai instances found.")
        return

    print(f"\nActive Vast.ai Fleet ({len(data)} instances):\n")

    for i, inst in enumerate(data):
        iid = inst.get("id")
        status = inst.get("actual_status", "unknown")
        gpu_name = f"{inst.get('num_gpus')}x {inst.get('gpu_name')}"
        cost = inst.get("dph_total", 0.0)
        ssh_host = inst.get("ssh_host")
        ssh_port = inst.get("ssh_port")
        public_ip = inst.get("public_ipaddr")
        direct_port = inst.get("ports", {}).get("22/tcp", [{}])[0].get("HostPort")
        cpu = inst.get("cpu_name", "")
        ram = inst.get("cpu_ram", 0) // 1024
        disk = int(inst.get("disk_space", 0))

        # Test connections
        direct_ok, direct_msg = test_ssh(public_ip, direct_port) if public_ip and direct_port else (False, "N/A")

        print(f"[{i+1}] Instance ID: {iid} | {gpu_name} (${cost:.4f}/hr)")
        print(f"    Status:     {status}")
        print(f"    Hardware:   CPU: {cpu} | RAM: {ram}GB | Disk: {disk}GB")
        print(f"    Direct SSH: ssh -p {direct_port} root@{public_ip} -> {direct_msg}")
        print(f"    Proxy SSH:  ssh -p {ssh_port} root@{ssh_host}")
        print("-" * 75)
    print()


def select_instance(target, prefer_direct=True):
    data = get_instances()
    if not data:
        print("No instances found.")
        return

    chosen = None
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(data):
            chosen = data[idx - 1]
        else:
            for inst in data:
                if str(inst.get("id")) == target:
                    chosen = inst
                    break
    if not chosen:
        print(f"Instance '{target}' not found.")
        return

    iid = chosen.get("id")
    gpu = f"{chosen.get('num_gpus')}x {chosen.get('gpu_name')}"
    public_ip = chosen.get("public_ipaddr")
    direct_port = chosen.get("ports", {}).get("22/tcp", [{}])[0].get("HostPort")
    proxy_host = chosen.get("ssh_host")
    proxy_port = chosen.get("ssh_port")

    # Pick host & port
    if prefer_direct and public_ip and direct_port:
        host, port = public_ip, direct_port
        conn_type = "Direct"
    else:
        host, port = proxy_host, proxy_port
        conn_type = "Proxy"

    remote_file = Path(".remote")
    content = f"""[remote]
host      = {host}
port      = {port}
user      = root
workspace = /workspace/MThesis
venv      = /venv/main/bin
venv_carla = /venv/carla_py38/bin
cpu_cores_a = 16-20
cpu_cores_b = 21-25
cpu_cores_c = 26-31
# Selected from Vast.ai: ID {iid} ({gpu}) via {conn_type}
"""
    remote_file.write_text(content, encoding="utf-8")
    print(f"\n[OK] Updated .remote -> {host}:{port} ({conn_type} SSH for Instance {iid}: {gpu})")
    
    # Test connection
    ok, msg = test_ssh(host, port)
    print(f"[Probe] SSH Test: {msg}\n")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "list"
    if action == "list":
        list_instances()
    elif action == "select" and len(sys.argv) > 2:
        prefer_dir = "--proxy" not in sys.argv
        select_instance(sys.argv[2], prefer_direct=prefer_dir)
    else:
        print("Usage: python scripts/setup/vast_fleet.py list | select <idx_or_id> [--proxy]")
