"""deploy_launch.py — Launch the Atari 100k benchmark on the remote box."""
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "setup"))
from remote_config import load_config

cfg = load_config()

cmd = (
    f"cd {cfg.workspace} && "
    f"export PYTHONPATH={cfg.workspace} && "
    f"nohup {cfg.python()} atari_qwen/scripts/run_100k_benchmark.py "
    "--variants S049a_mcts_sim20 S049b_mcts_sim35 S049c_mcts_sim50 "
    "--seeds 42 --parallel "
    "</dev/null >/workspace/benchmark_mcts_accelerated.log 2>&1 & "
    "echo $! && sleep 2 && "
    "ps aux | grep -E 'run_100k|train_mcts' | grep -v grep"
)

proc = subprocess.run(cfg.ssh_T(cmd), capture_output=True)
print("STDOUT:")
print(proc.stdout.decode("utf-8", errors="replace"))
print("STDERR:")
print(proc.stderr.decode("utf-8", errors="replace"))
