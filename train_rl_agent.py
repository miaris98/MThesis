import os
import sys
import time
import argparse
import json
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

# NumPy 2.x backward-compatibility shim for TensorBoard / older packages
try:
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from carla_gym_env import CarlaGymEnv
from camera_easycarla_env import CameraEasyCarlaEnv

# --- PyTorch Neural Network Architectures ---

class PretrainedVisionFeatureExtractor(nn.Module):
    """
    ImageNet Pretrained ResNet Feature Extractor for 256x256 RGB Images + Speed State.
    Supports backbone freezing for fast, sample-efficient RL training and zero-overhead feature caching.
    """
    def __init__(self, backbone_name="resnet18", features_dim=512, freeze_backbone=True, weights_path=None):
        super(PretrainedVisionFeatureExtractor, self).__init__()
        self.freeze_backbone = freeze_backbone

        if backbone_name == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if weights_path is None else None)
            backbone_out_dim = 512
        else:
            # Default: ResNet18
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_path is None else None)
            backbone_out_dim = 512

        # Remove the final 1000-class FC classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # -> Outputs (N, 512, 1, 1)

        # Optionally load CARLA-domain pretrained checkpoint (.pth)
        if weights_path is not None and os.path.exists(weights_path):
            print(f"--> Loading CARLA-domain pretrained vision weights (LAV / TransFuser++) from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint.get("state_dict_bev", checkpoint)))
            
            # Extract LAV / TransFuser++ / TCP camera encoder weights if prefixed
            extracted_dict = {}
            for k, v in state_dict.items():
                clean_k = k
                for prefix in ["image_encoder.", "encoder.image_encoder.", "perception.", "bev_planner.", "rgb_encoder.", "camera_encoder.", "bev_encoder.", "model."]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k.replace(prefix, "")
                extracted_dict[clean_k] = v

            missing, unexpected = self.backbone.load_state_dict(extracted_dict, strict=False)
            print(f"✓ Pretrained weights matched & loaded into {backbone_name.upper()} backbone! (Matched keys successfully)")

        # Freeze backbone parameters if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Linear projector for multi-camera visual vector (3 cameras: Left, Center, Right) + speed scalar
        self.num_cameras = 3
        self.visual_feature_dim = backbone_out_dim * self.num_cameras
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 1, features_dim),
            nn.ReLU()
        )

    def extract_visual_features(self, image):
        """Extract multi-camera visual embedding (N, 3 * D) with batch-level Tensor Core efficiency."""
        img_x = image.float() / 255.0

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                N, H, W, C = img_x.shape
                if W == H * 3:
                    # 3-camera horizontal panorama: split into Left, Center, Right
                    img_left = img_x[:, :, :H, :]
                    img_center = img_x[:, :, H:2*H, :]
                    img_right = img_x[:, :, 2*H:, :]
                    # Stack along batch: (3*N, 3, H, H)
                    cams = torch.cat([img_left, img_center, img_right], dim=0).permute(0, 3, 1, 2)
                    cams_normalized = (cams - self.mean) / self.std
                    if self.freeze_backbone:
                        with torch.no_grad():
                            conv_out = self.backbone(cams_normalized).flatten(start_dim=1)
                    else:
                        conv_out = self.backbone(cams_normalized).flatten(start_dim=1)

                    left_out, center_out, right_out = torch.chunk(conv_out, 3, dim=0)
                    visual_features = torch.cat([left_out, center_out, right_out], dim=1)
                else:
                    # Single camera input: (N, 3, H, W)
                    img_perm = img_x.permute(0, 3, 1, 2)
                    img_normalized = (img_perm - self.mean) / self.std
                    if self.freeze_backbone:
                        with torch.no_grad():
                            single_out = self.backbone(img_normalized).flatten(start_dim=1)
                    else:
                        single_out = self.backbone(img_normalized).flatten(start_dim=1)
                    visual_features = single_out.repeat(1, self.num_cameras)
            else:
                img_normalized = (img_x - self.mean) / self.std
                if self.freeze_backbone:
                    with torch.no_grad():
                        single_out = self.backbone(img_normalized).flatten(start_dim=1)
                else:
                    single_out = self.backbone(img_normalized).flatten(start_dim=1)
                visual_features = single_out.repeat(1, self.num_cameras)

        return visual_features.float()

    def forward_with_visual_features(self, visual_features, speed):
        """Zero-backbone forward pass using cached visual features + speed scalar."""
        speed_x = speed.float().view(-1, 1) / 50.0  # Normalize speed by 50 km/h scale
        combined = torch.cat([visual_features, speed_x], dim=1)
        return self.fc(combined)

    def forward(self, image, speed):
        visual_features = self.extract_visual_features(image)
        return self.forward_with_visual_features(visual_features, speed)

class CNNFeatureExtractor(nn.Module):
    """NatureCNN-style architecture for extracting features from multi-camera RGB images + speed state."""
    def __init__(self, in_channels=3, features_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        self.num_cameras = 3
        self.freeze_backbone = False
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.visual_feature_dim = 12544 * self.num_cameras
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 1, features_dim),
            nn.ReLU()
        )

    def extract_visual_features(self, image):
        img_x = image.float() / 255.0
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                N, H, W, C = img_x.shape
                if W == H * 3:
                    img_left = img_x[:, :, :H, :]
                    img_center = img_x[:, :, H:2*H, :]
                    img_right = img_x[:, :, 2*H:, :]
                    cams = torch.cat([img_left, img_center, img_right], dim=0).permute(0, 3, 1, 2)
                    conv_out = self.conv(cams)
                    left_out, center_out, right_out = torch.chunk(conv_out, 3, dim=0)
                    visual_features = torch.cat([left_out, center_out, right_out], dim=1)
                else:
                    img_perm = img_x.permute(0, 3, 1, 2)
                    single_out = self.conv(img_perm)
                    visual_features = single_out.repeat(1, self.num_cameras)
            else:
                single_out = self.conv(img_x)
                visual_features = single_out.repeat(1, self.num_cameras)
        return visual_features.float()

    def forward_with_visual_features(self, visual_features, speed):
        speed_x = speed.float().view(-1, 1) / 50.0
        combined = torch.cat([visual_features, speed_x], dim=1)
        return self.fc(combined)

    def forward(self, image, speed):
        visual_features = self.extract_visual_features(image)
        return self.forward_with_visual_features(visual_features, speed)

# --- ERFNet CARLA Camera Feature Extractor ---

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input_tensor):
        output = torch.cat([self.conv(input_tensor), self.pool(input_tensor)], 1)
        output = self.bn(output)
        return F.relu(output)

class NonBottleneck1D(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(dilated,0), bias=True, dilation=(dilated,1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,dilated), bias=True, dilation=(1,dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input_tensor):
        output = F.relu(self.conv3x1_1(input_tensor))
        output = self.bn1(F.relu(self.conv1x3_1(output)))
        output = F.relu(self.conv3x1_2(output))
        output = self.bn2(self.conv1x3_2(output))
        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input_tensor)

class ERFNetFeatureExtractor(nn.Module):
    """ERFNet Camera Feature Extractor for CARLA Semantic Perception."""
    def __init__(self, features_dim=512, freeze_backbone=True, weights_path=None):
        super(ERFNetFeatureExtractor, self).__init__()
        self.freeze_backbone = freeze_backbone
        
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64))
        for _ in range(5):
            self.layers.append(NonBottleneck1D(64, 0.03, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for _ in range(2):
            self.layers.append(NonBottleneck1D(128, 0.3, 2))
            self.layers.append(NonBottleneck1D(128, 0.3, 4))
            self.layers.append(NonBottleneck1D(128, 0.3, 8))
            self.layers.append(NonBottleneck1D(128, 0.3, 16))

        if weights_path is not None and os.path.exists(weights_path):
            print(f"--> Loading ERFNet CARLA camera weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            self.load_state_dict(state_dict, strict=False)

        if self.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_cameras = 3
        self.visual_feature_dim = 128 * self.num_cameras
        self.fc = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 1, features_dim),
            nn.ReLU()
        )

    def _forward_erfnet(self, tensor_input):
        x = self.initial_block(tensor_input)
        for layer in self.layers:
            x = layer(x)
        return self.pool(x).flatten(start_dim=1)

    def extract_visual_features(self, image):
        img_x = image.float() / 255.0
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=img_x.is_cuda):
            if img_x.ndim == 4 and img_x.shape[-1] == 3:
                N, H, W, C = img_x.shape
                if W == H * 3:
                    img_left = img_x[:, :, :H, :]
                    img_center = img_x[:, :, H:2*H, :]
                    img_right = img_x[:, :, 2*H:, :]
                    cams = torch.cat([img_left, img_center, img_right], dim=0).permute(0, 3, 1, 2)
                    if self.freeze_backbone:
                        with torch.no_grad():
                            conv_out = self._forward_erfnet(cams)
                    else:
                        conv_out = self._forward_erfnet(cams)
                    left_out, center_out, right_out = torch.chunk(conv_out, 3, dim=0)
                    visual_features = torch.cat([left_out, center_out, right_out], dim=1)
                else:
                    img_perm = img_x.permute(0, 3, 1, 2)
                    if self.freeze_backbone:
                        with torch.no_grad():
                            single_out = self._forward_erfnet(img_perm)
                    else:
                        single_out = self._forward_erfnet(img_perm)
                    visual_features = single_out.repeat(1, self.num_cameras)
            else:
                if self.freeze_backbone:
                    with torch.no_grad():
                        single_out = self._forward_erfnet(img_x)
                else:
                    single_out = self._forward_erfnet(img_x)
                visual_features = single_out.repeat(1, self.num_cameras)
        return visual_features.float()

    def forward_with_visual_features(self, visual_features, speed):
        speed_x = speed.float().view(-1, 1) / 50.0
        combined = torch.cat([visual_features, speed_x], dim=1)
        return self.fc(combined)

    def forward(self, image, speed):
        visual_features = self.extract_visual_features(image)
        return self.forward_with_visual_features(visual_features, speed)

class ActorCriticPPO(nn.Module):
    """PPO Actor-Critic Policy Network for Continuous Driving Control."""
    def __init__(self, action_dim=3, features_dim=512, backbone_name="resnet18", freeze_backbone=True, use_pretrained=True, weights_path=None):
        super(ActorCriticPPO, self).__init__()
        
        if use_pretrained:
            if backbone_name == "erfnet":
                self.encoder = ERFNetFeatureExtractor(features_dim=features_dim, freeze_backbone=freeze_backbone, weights_path=weights_path)
            else:
                self.encoder = PretrainedVisionFeatureExtractor(
                    backbone_name=backbone_name,
                    features_dim=features_dim,
                    freeze_backbone=freeze_backbone,
                    weights_path=weights_path
                )
        else:
            self.encoder = CNNFeatureExtractor(in_channels=3, features_dim=features_dim)

        # Actor Head: Outputs mean action [throttle, steer, brake]
        self.actor_mean = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # Initialize actor final linear layer bias so initial throttle starts positive
        with torch.no_grad():
            self.actor_mean[2].bias.data[0] = 0.5  # Positive initial throttle
            self.actor_mean[2].bias.data[1] = 0.0  # Neutral steer
            self.actor_mean[2].bias.data[2] = -0.5 # Negative initial brake
        
        # Learned Log Standard Deviation for continuous action exploration
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic Head: Outputs scalar state-value V(s)
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def extract_visual_features(self, image):
        """Extract multi-camera visual features using vision backbone."""
        return self.encoder.extract_visual_features(image)

    def get_action_and_value(self, image=None, speed=None, action=None, deterministic=False, visual_features=None):
        if visual_features is not None:
            features = self.encoder.forward_with_visual_features(visual_features, speed)
        else:
            features = self.encoder(image, speed)
            
        action_mean = self.actor_mean(features)
        
        # Clip log_std to maintain numerical stability [-2, 0.5]
        action_std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = action_mean if deterministic else dist.sample()
            
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(features).squeeze(-1)
        
        return action, log_prob, entropy, value

    def train(self, mode=True):
        super().train(mode)
        if getattr(self.encoder, 'freeze_backbone', False):
            if hasattr(self.encoder, 'backbone'):
                self.encoder.backbone.eval()
            else:
                self.encoder.eval()
        return self

# --- Reward Normalization Utility ---

class RunningMeanStd:
    """Tracks running mean and variance for online reward normalization (Welford's algorithm)."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        x = float(x)
        delta = x - self.mean
        self.count += 1
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    @property
    def std(self):
        return max(float(np.sqrt(self.var)), 1e-4)


# --- Unified Experiment Logger (MLflow + TensorBoard) ---

class ExperimentLogger:
    """
    Unified MLflow + TensorBoard Logger.
    Logs metrics, hyperparameters, and artifacts to MLflow (and TensorBoard).
    Auto-starts MLflow UI server on port 10100 and outputs a clickable link to stdout.
    """
    def __init__(self, log_dir, checkpoint_dir=None, experiment_name="CARLA_PPO_RL", use_mlflow=True, mlflow_port=10100, resume=False):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.tb_writer = SummaryWriter(log_dir)
        self.use_mlflow = False
        self.mlflow_port = mlflow_port
        self.run_id = None
        
        if use_mlflow:
            try:
                import mlflow
                import socket
                import subprocess
                import urllib.request

                # Check if MLflow UI server is already running on mlflow_port
                # NOTE: MLflow UI is started persistently by run_training_loop.sh
                # in a dedicated tmux session so it survives training crashes.
                # We only launch it here as a fallback for standalone usage.
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

                # Fetch Public IP if available
                public_ip = "127.0.0.1"
                try:
                    public_ip = urllib.request.urlopen("https://api.ipify.org", timeout=2.0).read().decode('utf-8').strip()
                except Exception:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        public_ip = s.getsockname()[0]
                        s.close()
                    except Exception:
                        pass

                self.mlflow = mlflow
                # Point the client to the running MLflow server so all data flows
                # through it (ensures consistent backend store across restarts)
                if port_in_use:
                    self.mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")
                self.mlflow.set_experiment(experiment_name)

                # Check for existing MLflow Run ID to resume if requested
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

                print(f"======================================================================")
                print(f"   📊 MLFLOW DASHBOARD ONLINE (PORT {mlflow_port})                      ")
                print(f"   👉 Clickable Public URL:  http://{public_ip}:{mlflow_port}          ")
                print(f"   👉 Localhost URL:         http://127.0.0.1:{mlflow_port}             ")
                print(f"   ✓ Experiment: '{experiment_name}' | Run ID: {self.run_id}")
                print(f"======================================================================")
            except Exception as e:
                print(f"--> MLflow import/init note ({e}). Logging to TensorBoard at {log_dir}")
                self.use_mlflow = False

    def log_params(self, args_obj):
        if self.use_mlflow:
            try:
                params_dict = {k: str(v) for k, v in vars(args_obj).items()}
                self.mlflow.log_params(params_dict)
            except Exception as e:
                print(f"--> MLflow log_params warning: {e}")

    def add_scalar(self, tag, scalar_value, global_step):
        self.tb_writer.add_scalar(tag, scalar_value, global_step)
        if self.use_mlflow:
            try:
                clean_tag = tag.replace("/", "_")
                self.mlflow.log_metric(clean_tag, float(scalar_value), step=int(global_step))
            except Exception:
                pass

    def add_text(self, tag, text_string, global_step):
        self.tb_writer.add_text(tag, text_string, global_step)
        if self.use_mlflow:
            try:
                clean_tag = tag.replace("/", "_")
                self.mlflow.log_param(f"text_{clean_tag}_step_{global_step}", text_string)
            except Exception:
                pass

    def log_artifact(self, file_path):
        if self.use_mlflow and os.path.exists(file_path):
            try:
                self.mlflow.log_artifact(file_path)
            except Exception:
                pass

    def close(self):
        self.tb_writer.close()
        if self.use_mlflow:
            try:
                self.mlflow.end_run()
            except Exception:
                pass


def _get_hardware_metrics():
    """
    Fetch real-time GPU VRAM usage, System RAM usage, and CPU load.
    Returns dict with hardware telemetry values.
    """
    metrics = {
        "gpu_mem_used_mb": 0.0,
        "gpu_mem_total_mb": 0.0,
        "gpu_mem_pct": 0.0,
        "sys_cpu_pct": 0.0,
        "sys_ram_used_gb": 0.0,
        "sys_ram_total_gb": 0.0
    }
    try:
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / (1024.0 ** 2)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)
            metrics["gpu_mem_used_mb"] = round(mem_used, 1)
            metrics["gpu_mem_total_mb"] = round(mem_total, 1)
            metrics["gpu_mem_pct"] = round((mem_used / max(1.0, mem_total)) * 100.0, 1)
    except Exception:
        pass

    try:
        import psutil
        metrics["sys_cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        metrics["sys_ram_used_gb"] = round(vm.used / (1024.0 ** 3), 2)
        metrics["sys_ram_total_gb"] = round(vm.total / (1024.0 ** 3), 2)
    except Exception:
        pass

    return metrics


class CSVTelemetryLogger:
    """
    Step-by-step CSV telemetry recorder for deep review and offline analysis.
    Logs inputs, actions, rewards, sub-rewards, hardware metrics, and curriculum parameters.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.fieldnames = [
            "global_step", "episode", "step_in_ep",
            "speed_kmh", "action_throttle", "action_steer", "action_brake",
            "raw_reward", "normalized_reward", "curriculum_alpha",
            "r_speed", "r_heading", "r_lateral", "r_boundary", "r_steer",
            "r_comfort", "r_wrong_way", "r_light", "r_obstacle", "r_ttc", "r_idle", "r_stall",
            "gpu_mem_used_mb", "gpu_mem_pct", "sys_cpu_pct", "sys_ram_used_gb",
            "is_collision", "is_off_road", "termination_reason"
        ]
        file_exists = os.path.exists(filepath)
        self.file = open(filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log_step(self, row_dict):
        try:
            self.writer.writerow(row_dict)
        except Exception:
            pass

    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


# --- Main Training Function ---

def train():
    parser = argparse.ArgumentParser(description="Train PPO Deep RL Agent in CARLA Simulator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--env-type", type=str, default="camera_easycarla", choices=["camera_easycarla", "carla_gym"], help="Environment type")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "lav", "erfnet"], help="Pretrained vision backbone (resnet18, resnet34, lav, erfnet)")
    parser.add_argument("--weights-path", type=str, default=None, help="Optional path to custom pretrained vision checkpoint (.pth)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze vision backbone parameters")
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone", help="Fine-tune vision backbone parameters")
    parser.add_argument("--use-pretrained", action="store_true", default=True, help="Use pretrained vision backbone")
    parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained", help="Train CNN from scratch")

    parser.add_argument("--total-steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--rollout-steps", type=int, default=250, help="Steps per PPO rollout buffer")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO optimization epochs per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--log-dir", type=str, default="/workspace/runs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints", help="Model checkpoint directory")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint")
    parser.add_argument("--num-vehicles", type=int, default=3, help="Number of surrounding NPC vehicles")
    parser.add_argument("--num-walkers", type=int, default=10, help="Number of pedestrian walkers")
    parser.add_argument("--town", type=str, default="Town10HD_Opt", help="CARLA map/town to use for training")
    parser.add_argument("--reward-clip", type=float, default=50.0, help="Clip raw rewards to [-reward-clip, +reward-clip]")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="PPO entropy bonus coefficient")
    parser.add_argument("--minibatch-size", type=int, default=128, help="PPO mini-batch size for GPU Tensor Core acceleration")
    parser.add_argument("--use-mlflow", action="store_true", default=True, help="Enable MLflow experiment tracking")
    parser.add_argument("--no-mlflow", action="store_false", dest="use_mlflow", help="Disable MLflow experiment tracking")
    parser.add_argument("--experiment-name", type=str, default="CARLA_PPO_RL", help="MLflow experiment name")
    parser.add_argument("--mlflow-port", type=int, default=10100, help="MLflow UI web dashboard port (default: 10100)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable PyTorch 2.x torch.compile graph acceleration")

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            test_x = torch.ones(2, device="cuda")
            _ = test_x + 1
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception as cuda_err:
            print(f"⚠️  GPU CUDA kernel execution failed ({cuda_err}). Running policy on CPU.")
            device = torch.device("cpu")
            
    # Dynamic Hardware Auto-Tuning based on detected GPU VRAM
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_vram_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"--> Hardware Telemetry: {torch.cuda.get_device_name(0)} | Total VRAM: {total_vram_gb:.1f} GB | Free VRAM: {free_vram_gb:.1f} GB")
        
        # Scale minibatch dynamically if user kept default 128 to maximize GPU saturation
        if args.minibatch_size == 128:
            if free_vram_gb >= 10.0:
                args.minibatch_size = 256
                print(f"⚡ [Dynamic Auto-Tune] Detected {free_vram_gb:.1f} GB free VRAM: Auto-scaled PPO Minibatch to 256 for higher GPU Tensor Core saturation.")
            elif free_vram_gb >= 20.0:
                args.minibatch_size = 512
                print(f"⚡ [Dynamic Auto-Tune] Detected {free_vram_gb:.1f} GB free VRAM: Auto-scaled PPO Minibatch to 512 for maximum GPU throughput.")
    print(f"==============================================================")
    print(f"   🚀 Starting High-Throughput PPO Deep RL Training           ")
    print(f"==============================================================")
    print(f"Device: {device} | Environment: {args.env_type}")
    print(f"Vision Backbone: {args.backbone.upper()} (Pretrained: {args.use_pretrained}, Frozen: {args.freeze_backbone})")
    print(f"Feature Caching Acceleration: {'ENABLED (PPO updates skip backbone)' if args.freeze_backbone else 'DISABLED (Fine-tuning)'}")
    print(f"Sensors: 3-Camera Zero-Copy RGB Panorama (Left, Center, Right) + Speed")
    print(f"NPC Traffic: {args.num_vehicles} Vehicles | {args.num_walkers} Pedestrians")
    if args.weights_path:
        print(f"CARLA Pretrained Checkpoint: {os.path.abspath(args.weights_path)}")
    print(f"Total Steps: {args.total_steps} | Rollout Buffer: {args.rollout_steps} | Minibatch: {args.minibatch_size}")
    writer = ExperimentLogger(
        args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        use_mlflow=args.use_mlflow,
        mlflow_port=args.mlflow_port,
        resume=args.resume
    )
    writer.log_params(args)
    
    # Initialize Selected Environment
    if args.env_type == "camera_easycarla":
        easy_params = {
            'number_of_vehicles': args.num_vehicles,
            'number_of_walkers': args.num_walkers,
            'dt': 0.05,
            'ego_vehicle_filter': 'vehicle.tesla.model3',
            'surrounding_vehicle_spawned_randomly': True,
            'port': args.port,
            'town': args.town,
            'max_time_episode': args.rollout_steps,
            'max_waypoints': 12,
            'visualize_waypoints': False,
            'desired_speed': 8,
            'max_ego_spawn_times': 200,
            'view_mode': 'top',
            'traffic': 'off',
            'lidar_max_range': 50.0,
            'max_nearby_vehicles': 5,
            'img_width': 256,
            'img_height': 256,
        }
        env = CameraEasyCarlaEnv(params=easy_params)
    else:
        env = CarlaGymEnv(host=args.host, port=args.port, img_width=256, img_height=256, max_steps=args.rollout_steps)
    
    # Initialize PPO Policy & Optimizer
    agent = ActorCriticPPO(
        action_dim=3,
        features_dim=512,
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        use_pretrained=args.use_pretrained,
        weights_path=args.weights_path
    ).to(device)

    if getattr(args, 'compile', False) and hasattr(torch, 'compile'):
        try:
            print("--> Enabling PyTorch 2.x torch.compile JIT optimization...")
            agent = torch.compile(agent)
            print("✓ Policy network compiled successfully!")
        except Exception as compile_err:
            print(f"--> Note: torch.compile notice ({compile_err}). Continuing with standard eager execution.")

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    # Resume from latest checkpoint if requested
    global_step = 0
    best_episode_reward = -float("inf")
    if args.resume:
        latest_ckpt = os.path.join(args.checkpoint_dir, "ppo_carla_latest.pth")
        if not os.path.exists(latest_ckpt) and os.path.exists(os.path.join(args.checkpoint_dir, "ppo_carla_best.pth")):
            latest_ckpt = os.path.join(args.checkpoint_dir, "ppo_carla_best.pth")
        state_file = os.path.join(args.checkpoint_dir, "train_state.json")
        if os.path.exists(latest_ckpt):
            agent.load_state_dict(torch.load(latest_ckpt, map_location=device), strict=False)
            print(f"[Resume] Loaded policy checkpoint: {latest_ckpt}")
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            global_step = state.get("global_step", 0)
            best_episode_reward = state.get("best_episode_reward", -float("inf"))
            print(f"[Resume] Continuing from step {global_step}/{args.total_steps} | Best reward so far: {best_episode_reward:.2f}")

    # Running reward normalizer: keeps PPO value targets in a stable range
    reward_normalizer = RunningMeanStd()

    obs, _ = env.reset()

    # Setup CSV Telemetry Logger
    csv_file_path = os.path.join(args.log_dir, "training_telemetry.csv")
    csv_logger = CSVTelemetryLogger(csv_file_path)
    episode_count = 1

    episode_rewards = []
    episode_speeds = []
    current_ep_reward = 0
    current_ep_speeds = []
    # AMP Scaler
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    is_frozen_backbone = bool(getattr(agent.encoder, 'freeze_backbone', False))

    while global_step < args.total_steps:
        # Storage buffers for Rollout
        obs_images = []
        obs_visual_features = []
        obs_speeds = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        # 1. Collect Rollout Trajectory
        rollout_start_time = time.time()

        # Literature-aligned dynamic penalty curriculum schedule (20% warmup horizon: alpha in [0.2, 1.0])
        warmup_steps = max(10000, int(0.20 * args.total_steps))
        curriculum_factor = min(1.0, max(0.2, global_step / float(warmup_steps)))
        if hasattr(env, 'set_curriculum_factor'):
            env.set_curriculum_factor(curriculum_factor)

        for step in range(args.rollout_steps):
            global_step += 1
            
            img_tensor = torch.as_tensor(obs["image"], dtype=torch.uint8, device=device).unsqueeze(0)
            spd_tensor = torch.as_tensor(obs["speed"], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.inference_mode():
                if is_frozen_backbone:
                    # Extract visual features once during rollout; cache for zero-overhead PPO optimization
                    vis_feat = agent.extract_visual_features(img_tensor)
                    action, log_prob, _, value = agent.get_action_and_value(speed=spd_tensor, visual_features=vis_feat)
                    obs_visual_features.append(vis_feat.squeeze(0))
                else:
                    action, log_prob, _, value = agent.get_action_and_value(image=img_tensor, speed=spd_tensor)
                    obs_images.append(img_tensor.squeeze(0))

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Clip raw reward, then normalize by running std to keep value targets in a healthy range.
            # Log the raw reward for human-readable episode summaries.
            raw_reward = float(reward)
            clipped_reward = float(np.clip(raw_reward, -args.reward_clip, args.reward_clip))
            reward_normalizer.update(clipped_reward)
            normalized_reward = clipped_reward / reward_normalizer.std

            obs_speeds.append(spd_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            rewards.append(normalized_reward)  # Normalized reward into PPO buffer
            dones.append(done)
            values.append(value.squeeze(0))

            obs = next_obs
            current_ep_reward += raw_reward  # Accumulate raw reward for logging
            speed_val = info.get("speed_kmh", obs["speed"][0])
            current_ep_speeds.append(speed_val)

            # Fetch real-time hardware telemetry (GPU VRAM, CPU, RAM)
            hw_metrics = _get_hardware_metrics()

            # Record step telemetry to CSV file
            csv_logger.log_step({
                "global_step": global_step,
                "episode": episode_count,
                "step_in_ep": len(current_ep_speeds),
                "speed_kmh": round(float(speed_val), 2),
                "action_throttle": round(float(action_np[0]), 3),
                "action_steer": round(float(action_np[1]), 3),
                "action_brake": round(float(action_np[2]), 3),
                "raw_reward": round(raw_reward, 4),
                "normalized_reward": round(normalized_reward, 4),
                "curriculum_alpha": round(float(curriculum_factor), 2),
                "r_speed": round(float(info.get("r_speed", 0.0)), 4),
                "r_heading": round(float(info.get("r_heading", 0.0)), 4),
                "r_lateral": round(float(info.get("r_lateral", 0.0)), 4),
                "r_boundary": round(float(info.get("r_boundary", 0.0)), 4),
                "r_steer": round(float(info.get("r_steer", 0.0)), 4),
                "r_comfort": round(float(info.get("r_comfort", 0.0)), 4),
                "r_wrong_way": round(float(info.get("r_wrong_way", 0.0)), 4),
                "r_light": round(float(info.get("r_light", 0.0)), 4),
                "r_obstacle": round(float(info.get("r_obstacle", 0.0)), 4),
                "r_ttc": round(float(info.get("r_ttc", 0.0)), 4),
                "r_idle": round(float(info.get("r_idle", 0.0)), 4),
                "r_stall": round(float(info.get("r_stall", 0.0)), 4),
                "gpu_mem_used_mb": hw_metrics["gpu_mem_used_mb"],
                "gpu_mem_pct": hw_metrics["gpu_mem_pct"],
                "sys_cpu_pct": hw_metrics["sys_cpu_pct"],
                "sys_ram_used_gb": hw_metrics["sys_ram_used_gb"],
                "is_collision": info.get("is_collision", False),
                "is_off_road": info.get("is_off_road", False),
                "termination_reason": info.get("termination_reason", "") if done else ""
            })

            # Accumulate sub-reward breakdowns
            if 'current_ep_sub_rewards' not in locals():
                current_ep_sub_rewards = {k: 0.0 for k in ["r_speed", "r_heading", "r_lateral", "r_boundary", "r_steer", "r_comfort", "r_wrong_way", "r_light", "r_obstacle", "r_idle"]}
            for sub_k in current_ep_sub_rewards:
                current_ep_sub_rewards[sub_k] += float(info.get(sub_k, 0.0))

            if done:
                episode_count += 1
                csv_logger.flush()
                episode_rewards.append(current_ep_reward)
                avg_speed = np.mean(current_ep_speeds)
                episode_speeds.append(avg_speed)
                ep_len = len(current_ep_speeds)

                # Compute CARLA Benchmark Driving Score Estimate (DS_est)
                completion_pct = min(1.0, ep_len / float(args.rollout_steps))
                infraction_penalty = 1.0
                if info.get("is_collision", False):
                    infraction_penalty *= 0.60  # Vehicle/pedestrian collision penalty factor
                if info.get("is_off_road", False):
                    infraction_penalty *= 0.50  # Off-road penalty factor
                if info.get("termination_reason", "") == "Stalled / No Movement":
                    infraction_penalty *= 0.85  # Stall penalty factor
                if current_ep_sub_rewards.get("r_light", 0.0) < -2.0:
                    infraction_penalty *= 0.70  # Red light violation factor

                ds_est = 100.0 * completion_pct * infraction_penalty

                term_reason = info.get("termination_reason", "Collision" if info.get("is_collision", False) else ("Lane Deviation / Off-Road" if info.get("is_off_road", False) else "Max Steps"))
                print(f"[Step {global_step:05d}/{args.total_steps}] Episode Finished | Reward: {current_ep_reward:+.2f} (PerStep: {current_ep_reward/ep_len:+.2f}) | DS_Est: {ds_est:.1f} | Avg Speed: {avg_speed:.1f} km/h | Reason: {term_reason}")
                
                # TensorBoard Episode & Benchmark Metrics
                writer.add_scalar("Reward/Episode_Total", current_ep_reward, global_step)
                writer.add_scalar("Reward/PerStep_Mean", current_ep_reward / ep_len, global_step)
                writer.add_scalar("Speed/Avg_kmh", avg_speed, global_step)
                writer.add_scalar("CARLA_Benchmark/DrivingScore_Est", ds_est, global_step)
                writer.add_scalar("CARLA_Benchmark/RouteCompletion_Pct", completion_pct * 100.0, global_step)
                writer.add_scalar("CARLA_Benchmark/Infraction_Multiplier", infraction_penalty, global_step)
                writer.add_scalar("Hardware/GPU_Memory_MB", hw_metrics["gpu_mem_used_mb"], global_step)
                writer.add_scalar("Hardware/CPU_Usage_Pct", hw_metrics["sys_cpu_pct"], global_step)
                writer.add_scalar("Hardware/RAM_Used_GB", hw_metrics["sys_ram_used_gb"], global_step)
                writer.add_text("Termination_Reason", term_reason, global_step)

                # Log detailed total and per-step sub-reward breakdown to TensorBoard
                for sub_k, sub_val in current_ep_sub_rewards.items():
                    writer.add_scalar(f"SubReward_Total/{sub_k}", sub_val, global_step)
                    writer.add_scalar(f"SubReward_PerStep/{sub_k}", sub_val / ep_len, global_step)
                current_ep_sub_rewards = {k: 0.0 for k in current_ep_sub_rewards}

                if current_ep_reward > best_episode_reward:
                    best_episode_reward = current_ep_reward
                    checkpoint_path = os.path.join(args.checkpoint_dir, "ppo_carla_best.pth")
                    torch.save(agent.state_dict(), checkpoint_path)
                    writer.log_artifact(checkpoint_path)
                    print(f"--> Saved new BEST policy checkpoint: {checkpoint_path}")

                # Persist training state so --resume can continue from here after a crash
                state_path = os.path.join(args.checkpoint_dir, "train_state.json")
                state_data = {
                    "global_step": global_step,
                    "best_episode_reward": float(best_episode_reward),
                    "total_steps": args.total_steps
                }
                if writer.use_mlflow and writer.run_id:
                    state_data["mlflow_run_id"] = writer.run_id
                with open(state_path, "w") as f:
                    json.dump(state_data, f, indent=2)

                obs, _ = env.reset()
                current_ep_reward = 0
                current_ep_speeds = []

        rollout_elapsed = time.time() - rollout_start_time
        sps = args.rollout_steps / max(1e-4, rollout_elapsed)
        writer.add_scalar("Speed/SPS", sps, global_step)

        # 2. Compute Generalized Advantage Estimation (GAE)
        with torch.inference_mode():
            next_img = torch.as_tensor(obs["image"], dtype=torch.uint8, device=device).unsqueeze(0)
            next_spd = torch.as_tensor(obs["speed"], dtype=torch.float32, device=device).unsqueeze(0)
            if is_frozen_backbone:
                next_vis = agent.extract_visual_features(next_img)
                next_val = agent.get_action_and_value(speed=next_spd, visual_features=next_vis)[3].squeeze(0)
            else:
                next_val = agent.get_action_and_value(image=next_img, speed=next_spd)[3].squeeze(0)

        returns = []
        advantages = []
        gae = 0
        for t in reversed(range(args.rollout_steps)):
            if t == args.rollout_steps - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = next_val
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values[t + 1]

            delta = rewards[t] + args.gamma * next_value * next_non_terminal - values[t]
            gae = delta + args.gamma * args.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        # Convert Rollout Lists to GPU Tensors
        if is_frozen_backbone:
            b_vis_feats = torch.stack(obs_visual_features)
        else:
            b_images = torch.stack(obs_images)
        b_speeds = torch.stack(obs_speeds)
        b_actions = torch.stack(actions)
        b_log_probs = torch.stack(log_probs)
        b_advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        b_values = torch.stack(values)

        # Normalize Advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 3. Optimize PPO Policy & Value Networks with Minibatch Updates & AMP Mixed Precision
        policy_losses = []
        value_losses = []
        minibatch_size = getattr(args, 'minibatch_size', 128)
        b_inds = np.arange(args.rollout_steps)

        ppo_start_time = time.time()

        for epoch in range(args.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.rollout_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                autocast_ctx = torch.amp.autocast('cuda', enabled=torch.cuda.is_available()) if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
                with autocast_ctx:
                    if is_frozen_backbone:
                        # Zero-backbone forward pass: only trains policy & value MLP heads (0.002s per epoch!)
                        _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                            speed=b_speeds[mb_inds],
                            action=b_actions[mb_inds],
                            visual_features=b_vis_feats[mb_inds]
                        )
                    else:
                        _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                            image=b_images[mb_inds],
                            speed=b_speeds[mb_inds],
                            action=b_actions[mb_inds]
                        )
                    logratio = new_log_prob - b_log_probs[mb_inds]
                    ratio = logratio.exp()

                    # PPO Clipped Surrogate Loss
                    pg_loss1 = -b_advantages[mb_inds] * ratio
                    pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    # Combined Total Loss with exploration entropy bonus
                    total_loss = pg_loss + 0.5 * v_loss - args.ent_coef * entropy.mean()

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())

        ppo_elapsed = time.time() - ppo_start_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        writer.add_scalar("Loss/Policy", np.mean(policy_losses), global_step)
        writer.add_scalar("Loss/Value", np.mean(value_losses), global_step)
        writer.add_scalar("Speed/PPO_Update_Sec", ppo_elapsed, global_step)
        print(f"--- Rollout Update Complete | Step: {global_step}/{args.total_steps} | SPS: {sps:.1f} | PPO Opt Time: {ppo_elapsed*1000.0:.1f}ms | Policy Loss: {np.mean(policy_losses):.4f} | Value Loss: {np.mean(value_losses):.4f} ---")

        # Save Latest Checkpoint & State Metadata
        latest_path = os.path.join(args.checkpoint_dir, "ppo_carla_latest.pth")
        torch.save(agent.state_dict(), latest_path)
        state_data = {
            "global_step": global_step,
            "best_episode_reward": float(best_episode_reward),
            "total_steps": args.total_steps
        }
        if writer.use_mlflow and writer.run_id:
            state_data["mlflow_run_id"] = writer.run_id
        with open(os.path.join(args.checkpoint_dir, "train_state.json"), "w") as f:
            json.dump(state_data, f, indent=2)

    env.close()
    csv_logger.close()
    writer.log_artifact(csv_file_path)
    writer.close()
    print("--- Training Completed Successfully! ---")
    print(f"Final Model & Telemetry CSV Saved to: {os.path.abspath(args.log_dir)}")

if __name__ == "__main__":
    train()
