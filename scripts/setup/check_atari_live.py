import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh(
            "cat /workspace/benchmark_mcts_accelerated.log; echo '---'; "
            "ps aux | grep -E 'python.*(run_100k|train_mcts)' | grep -v grep || echo 'NO_BENCHMARK_PROCESS'"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    check()
