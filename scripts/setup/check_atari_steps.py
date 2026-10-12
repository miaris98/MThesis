import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh(
            f"for f in {cfg.remote_path('results/100k_benchmark/_logs/S049*.log')}; do "
            "echo '--- File:' $(basename $f) '---'; tail -n 3 $f; done"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout.strip())


if __name__ == "__main__":
    check()
