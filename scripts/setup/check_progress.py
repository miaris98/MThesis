import subprocess
from remote_config import load_config

cfg = load_config()


def check():
    r = subprocess.run(
        cfg.ssh(
            "ls -lh /workspace/checkpoints/wor_qwen30m_8towns_fast 2>/dev/null; echo '==='; "
            "ls -lh /workspace/pretrained_carla 2>/dev/null; echo '==='; "
            "ps aux | grep -E 'curl|unzip|mlflow' | grep -v grep || true"
        ),
        capture_output=True, text=True,
    )
    print(r.stdout)


if __name__ == "__main__":
    check()
