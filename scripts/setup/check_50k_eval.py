import subprocess
from remote_config import load_config

cfg = load_config()


def check_50k():
    r = subprocess.run(
        cfg.ssh(
            f"for f in {cfg.remote_path('results/100k_benchmark/_logs/S049*.log')}; do "
            "  echo '=== ' $(basename $f) ' ==='; "
            "  grep -E 'EVALUATION|Score|FIRE|LEFT|RIGHT|NOOP|Step  50000' $f | tail -n 6; "
            "  tail -n 2 $f; "
            "done"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    check_50k()
