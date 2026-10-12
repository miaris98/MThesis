import subprocess
import time
from remote_config import load_config

cfg = load_config()

time.sleep(5)
r = subprocess.run(
    cfg.ssh(
        "cat /workspace/auto_launch.log | tail -n 20; echo '==='; "
        "ps aux | grep -E 'python.*(train_wor|run_100k|train_mcts)' | grep -v grep || true"
    ),
    capture_output=True, text=True,
)
print(r.stdout)
