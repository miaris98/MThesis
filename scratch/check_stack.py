"""check_stack.py — Inspect the kernel wait-channel of a remote process by PID."""
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "setup"))
from remote_config import load_config

cfg = load_config()

# Set the PID of the process you want to inspect
PID = 2371807

cmd = f"""
python3 -c "
with open('/proc/{PID}/wchan') as f:
    print('wchan:', f.read().strip())
with open('/proc/{PID}/stack') as f:
    print('stack:', f.read().strip())
"
"""

proc = subprocess.run(cfg.ssh(cmd), capture_output=True)
print("STDOUT:")
print(proc.stdout.decode("utf-8", errors="replace"))
print("STDERR:")
print(proc.stderr.decode("utf-8", errors="replace"))
