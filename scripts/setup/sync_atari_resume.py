import subprocess
import time
from remote_config import load_config

cfg = load_config()


def restart_atari_resumed():
    print("\n--> Stopping old Atari processes...")
    cfg.run_ssh(
        "kill -9 $(pgrep -f 'train_mcts_offpolicy') 2>/dev/null || true; "
        "kill -9 $(pgrep -f 'run_100k_benchmark') 2>/dev/null || true",
        check=False,
    )
    time.sleep(2)

    print("--> Launching Atari Benchmark with --resume and --start-step 38000 on cores 16-31...")
    launch_cmd = (
        f"cd {cfg.workspace} && "
        f"taskset -c 16-31 nohup {cfg.python()} atari_qwen/scripts/run_100k_benchmark.py "
        "--variants S049a_mcts_sim20 S049b_mcts_sim35 S049c_mcts_sim50 "
        "--seeds 42 --parallel --resume --start-step 38000 "
        "> /workspace/benchmark_mcts_accelerated.log 2>&1 &"
    )
    subprocess.run(cfg.ssh(launch_cmd), check=True)
    time.sleep(4)

    print("\n--> Verifying output log...")
    res = cfg.run_ssh(
        "cat /workspace/benchmark_mcts_accelerated.log; echo '---'; "
        "ps aux | grep -E 'python.*(train_mcts|run_100k)' | grep -v grep || echo 'NO_PROC'",
        check=False,
    )
    print(res.stdout)


if __name__ == "__main__":
    restart_atari_resumed()
