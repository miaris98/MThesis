import os
import subprocess
import time
from remote_config import load_config

cfg = load_config()


def ssh_exec(cmd, check=True):
    print(f"--> [REMOTE] {cmd}")
    res = subprocess.run(cfg.ssh(cmd), capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[FAILED] code {res.returncode}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout.strip()


def setup_pretrained_carla():
    print("=== [1/2] Setting up CARLA Pretrained Vision Weights ===")
    check_pth = ssh_exec("ls -lh /workspace/pretrained_carla/model_0030_0.pth 2>/dev/null || echo 'MISSING'")
    if "MISSING" in check_pth:
        print("Downloading TransFuser++ model_0030_0.pth from S3...")
        cmd = """
        mkdir -p /workspace/pretrained_carla && cd /workspace/pretrained_carla && \
        curl -sSL "https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip" -o models.zip && \
        unzip -q -o models.zip && \
        rm -f models.zip && \
        FOUND=$(find /workspace/pretrained_carla -name "model_*.pth" | head -n 1) && \
        if [ -n "$FOUND" ] && [ "$FOUND" != "/workspace/pretrained_carla/model_0030_0.pth" ]; then \
            cp "$FOUND" /workspace/pretrained_carla/model_0030_0.pth; \
        fi && \
        ls -lh /workspace/pretrained_carla/
        """
        out = ssh_exec(cmd)
        print(out)
    else:
        print(f"Pretrained weights already present: {check_pth}")


def setup_mlflow_ui():
    print("\n=== [2/2] Starting MLflow UI Tracking Server on Port 8080 ===")
    check_port = ssh_exec("ss -tulpn | grep :8080 || echo 'PORT_FREE'")
    if "PORT_FREE" in check_port:
        print("Starting MLflow UI daemon on 0.0.0.0:8080...")
        cmd = f"""
        mkdir -p {cfg.remote_path('mlruns')}
        nohup {cfg.venv}/mlflow ui --backend-store-uri {cfg.remote_path('mlruns')} --host 0.0.0.0 --port 8080 > /workspace/mlflow_ui.log 2>&1 &
        sleep 2
        ss -tulpn | grep :8080 || cat /workspace/mlflow_ui.log
        """
        out = ssh_exec(cmd)
        print(out)
        print("✓ MLflow UI active on port 8080! Access via http://localhost:8080 through SSH tunnel.")
    else:
        print("Port 8080 is already in use:")
        print(check_port)


if __name__ == "__main__":
    setup_pretrained_carla()
    setup_mlflow_ui()
