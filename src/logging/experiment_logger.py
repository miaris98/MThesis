"""Unified Experiment Logger managing MLflow and TensorBoard tracking."""
import os
import sys
import time
import json
import socket
import subprocess
import re
import numpy as np

# NumPy 2.x backward-compatibility shim for TensorBoard
try:
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
except Exception:
    pass

from typing import Optional, Any
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Unified MLflow + TensorBoard Logger.
    Logs metrics, hyperparameters, and artifacts across experiment runs and restarts.
    """
    def __init__(
        self,
        log_dir: str,
        checkpoint_dir: Optional[str] = None,
        experiment_name: str = "CARLA_PPO_RL",
        use_mlflow: bool = True,
        mlflow_port: int = 10100,
        resume: bool = False
    ):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.tb_writer = SummaryWriter(log_dir)
        self.use_mlflow = False
        self.mlflow_port = mlflow_port
        self.run_id: Optional[str] = None
        
        if use_mlflow:
            try:
                import mlflow

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                port_in_use = (sock.connect_ex(('127.0.0.1', mlflow_port)) == 0)
                sock.close()

                if not port_in_use:
                    print(f"--> Auto-launching MLflow UI tracking server on port {mlflow_port}...")
                    subprocess.Popen(
                        [sys.executable, "-m", "mlflow", "ui", "--host", "0.0.0.0", "--port", str(mlflow_port),
                         "--backend-store-uri", os.path.join(os.path.dirname(os.path.abspath(log_dir)), "mlruns")],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
                    time.sleep(2)
                else:
                    print(f"✓ MLflow UI server is already active on port {mlflow_port} (re-using persistent session).")

                self.mlflow = mlflow
                if port_in_use:
                    self.mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")
                self.mlflow.set_experiment(experiment_name)

                saved_run_id = None
                if resume and checkpoint_dir:
                    state_file = os.path.join(checkpoint_dir, "train_state.json")
                    if os.path.exists(state_file):
                        try:
                            with open(state_file, "r") as f:
                                st = json.load(f)
                            saved_run_id = st.get("mlflow_run_id")
                        except Exception:
                            pass

                if saved_run_id:
                    try:
                        self.mlflow.start_run(run_id=saved_run_id)
                        print(f"✓ [Resume MLflow] Re-connected to active MLflow Run ID: {saved_run_id}")
                    except Exception as run_e:
                        print(f"--> Note: Could not resume MLflow Run ID {saved_run_id} ({run_e}). Starting new run...")
                        self.mlflow.start_run()
                else:
                    self.mlflow.start_run()

                self.use_mlflow = True
                active_run = self.mlflow.active_run()
                if active_run:
                    self.run_id = active_run.info.run_id

                cf_url = None
                if os.path.exists("/tmp/mlflow_tunnel.log"):
                    try:
                        with open("/tmp/mlflow_tunnel.log", "r") as f:
                            log_txt = f.read()
                        m = re.search(r'https://[-a-zA-Z0-9@:%._\+~#=]*\.trycloudflare\.com', log_txt)
                        if m:
                            cf_url = m.group(0)
                    except Exception:
                        pass

                print(f"======================================================================")
                print(f"   📊 MLFLOW DASHBOARD ONLINE (PORT {mlflow_port})                      ")
                if cf_url:
                    print(f"   👉 Public HTTPS URL:      {cf_url}          ")
                print(f"   👉 Vast.ai Tunnel:        Open Port {mlflow_port} in Vast.ai Tunnels UI")
                print(f"   👉 Localhost URL:         http://127.0.0.1:{mlflow_port}             ")
                print(f"   ✓ Experiment: '{experiment_name}' | Run ID: {self.run_id}")
                print(f"======================================================================")
            except Exception as e:
                print(f"--> MLflow import/init note ({e}). Logging to TensorBoard at {log_dir}")
                self.use_mlflow = False

    def log_params(self, args_obj: Any) -> None:
        """Log hyperparameter dictionary or argparse Namespace to MLflow."""
        if self.use_mlflow:
            try:
                params_dict = {k: str(v) for k, v in vars(args_obj).items()} if hasattr(args_obj, '__dict__') else {k: str(v) for k, v in args_obj.items()}
                self.mlflow.log_params(params_dict)
            except Exception:
                pass

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Log scalar metric to TensorBoard and MLflow."""
        self.tb_writer.add_scalar(tag, scalar_value, global_step)
        if self.use_mlflow:
            try:
                clean_tag = tag.replace("/", "_")
                self.mlflow.log_metric(clean_tag, float(scalar_value), step=int(global_step))
            except Exception:
                pass

    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        """Log text payload to TensorBoard and MLflow."""
        self.tb_writer.add_text(tag, text_string, global_step)
        if self.use_mlflow:
            try:
                clean_tag = tag.replace("/", "_")
                self.mlflow.log_param(f"text_{clean_tag}_step_{global_step}", text_string)
            except Exception:
                pass

    def log_artifact(self, file_path: str) -> None:
        """Log file artifact to MLflow storage."""
        if self.use_mlflow and os.path.exists(file_path):
            try:
                self.mlflow.log_artifact(file_path)
            except Exception:
                pass

    def close(self) -> None:
        """Safely close TensorBoard and MLflow run handles."""
        self.tb_writer.close()
        if self.use_mlflow:
            try:
                self.mlflow.end_run()
            except Exception:
                pass
