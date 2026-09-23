import subprocess
from remote_config import load_config

cfg = load_config()


def terminate_atari():
    r = subprocess.run(
        cfg.ssh(
            "pkill -9 -f train_mcts_offpolicy 2>/dev/null || true; "
            "pkill -9 -f run_100k_benchmark 2>/dev/null || true; "
            "ps aux | grep -E 'python.*(train_mcts|run_100k)' | grep -v grep || echo 'ALL ATARI PROCESSES TERMINATED'"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    terminate_atari()
