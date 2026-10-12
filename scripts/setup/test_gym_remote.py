import subprocess
from remote_config import load_config

cfg = load_config()


def test_gym():
    r = subprocess.run(
        cfg.ssh(
            f"{cfg.python()} -c 'import gymnasium as gym, ale_py; "
            "env = gym.make(\"BreakoutNoFrameskip-v4\"); print(env.reset())'"
        ),
        capture_output=True, text=True,
    )
    print("STDOUT:", r.stdout)
    print("STDERR:", r.stderr)


if __name__ == "__main__":
    test_gym()
