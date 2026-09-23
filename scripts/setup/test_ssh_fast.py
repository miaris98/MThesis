"""test_ssh_fast.py — Quickly probe connectivity to the configured remote box(es).

By default tests the box in .remote. You can also pass extra host:port pairs
as command-line arguments: python test_ssh_fast.py 1.2.3.4:22
"""
import subprocess
import sys
from remote_config import load_config

sys.stdout.reconfigure(line_buffering=True)

cfg = load_config()

# Primary target from config; additional targets from CLI args
targets = [(cfg.host, cfg.port)]
for arg in sys.argv[1:]:
    h, p = arg.rsplit(":", 1)
    targets.append((h, p))

for host, port in targets:
    print(f"Testing {host}:{port}...")
    cmd = [
        "ssh", "-p", port,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        f"root@{host}",
        "echo OK"
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        print(f"Result for {host}:{port}: returncode={r.returncode}, out={r.stdout.strip()}, err={r.stderr.strip()}")
    except Exception as e:
        print(f"Error {host}:{port}: {e}")
