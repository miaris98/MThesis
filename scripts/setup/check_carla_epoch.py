import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh("tail -n 8 /workspace/train_wor_8towns.log"),
        capture_output=True, text=True,
    )
    print(r.stdout.strip())


if __name__ == "__main__":
    check()
