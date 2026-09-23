"""Launch Atari Gate 1 Proof-of-Concept (5k steps, 16 collection simulations).

Runs on remote box with:
- num_simulations = 16
- eval_simulations = 16
- eval_episodes = 10
- num_envs = 8 (per-env contiguous buffer)
- EMA target network
- total_steps = 5000
- MLflow tracking on port 10100
"""
import subprocess
import sys
import time
from remote_config import load_config

cfg = load_config()


def main():
    print("--> 1. Syncing latest trainer code to remote box...")
    subprocess.run(
        cfg.scp(
            "atari_qwen/training/train_mcts_offpolicy.py",
            cfg.remote_path("atari_qwen/training/train_mcts_offpolicy.py"),
        ),
        check=True,
    )
    print("✓ Synced train_mcts_offpolicy.py")

    print("\n--> 2. Ensuring remote log directory exists...")
    cfg.run_ssh(f"mkdir -p {cfg.remote_path('results/100k_benchmark/gate1_poc_16sims')}")

    print("\n--> 3. Dispatching Gate 1 Run (5,000 steps, 16 sims) in background...")
    gate1_cmd = (
        f"cd {cfg.workspace} && "
        f"nohup {cfg.python()} atari_qwen/training/train_mcts_offpolicy.py "
        "--env-id BreakoutNoFrameskip-v4 "
        "--total-steps 5000 "
        "--num-envs 8 "
        "--num-simulations 16 "
        "--eval-simulations 16 "
        "--eval-episodes 10 "
        "--eval-interval 2500 "
        "--min-replay-size 500 "
        "--seed 42 "
        "--log-dir results/100k_benchmark/gate1_poc_16sims "
        "> /workspace/gate1_poc.log 2>&1 &"
    )
    res = cfg.run_ssh(gate1_cmd, check=False)
    if res.returncode == 0:
        print("✓ Gate 1 training process successfully dispatched!")
    else:
        print(f"✗ Failed to dispatch Gate 1: {res.stderr}")
        sys.exit(1)

    print("\n--> 4. Monitoring initial startup (first 10 seconds)...")
    time.sleep(10)
    res = cfg.run_ssh("tail -n 25 /workspace/gate1_poc.log 2>/dev/null || true", check=False)
    print(res.stdout)


if __name__ == "__main__":
    main()
