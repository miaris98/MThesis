import subprocess
from remote_config import load_config

cfg = load_config()


def run_remote(cmd_str):
    res = subprocess.run(cfg.ssh(cmd_str), capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


if __name__ == "__main__":
    commands = [
        "which python; python3 --version; which uv",
        "ls -la /venv",
        f"{cfg.python()} -c 'import torch, timm, transformers, gymnasium, ale_py; "
        "print(\"Imports ok!\"); "
        "print(\"Torch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available(), torch.cuda.get_device_name(0))' 2>&1",
        f"cd {cfg.workspace} && git status && git log -1 --oneline"
    ]
    for c in commands:
        print(f"=== RUNNING: {c} ===")
        code, out, err = run_remote(c)
        print(f"Exit code: {code}")
        if out: print(out.strip())
        if err: print("STDERR:", err.strip())
