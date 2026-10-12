import subprocess
from remote_config import load_config

cfg = load_config()


def get_status():
    r = subprocess.run(
        cfg.ssh(f"""
echo "=== [1. GPU & SYSTEM UTILIZATION] ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

echo ""
echo "=== [2. ACTIVE TRAINING PROCESSES] ==="
ps aux | grep -E 'python.*(train_wor|run_100k|train_mcts)' | grep -v grep

echo ""
echo "=== [3. CARLA TRAINING LOG (TAIL)] ==="
tail -n 25 /workspace/train_wor_8towns.log

echo ""
echo "=== [4. ATARI BENCHMARK LOGS (TAILS)] ==="
for f in {cfg.remote_path('results/100k_benchmark/_logs/S049*.log')}; do
    if [ -f "$f" ]; then
        echo "--- File: $f ---"
        tail -n 12 "$f"
    fi
done
        """),
        capture_output=True, text=True,
    )
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr)


if __name__ == "__main__":
    get_status()
