"""World on Rails (WoR) Offline Trajectory Dataset and DataLoader.

Loads CARLA driving logs (RGB frames, ego telemetry, high-level commands,
and precomputed Q-values / future waypoint targets).

Supports two on-disk layouts:
  - "wor": the original WoR log format (`<route>/data.json` + `<route>/rgbs/*.jpg`),
    which carries precomputed Q-values.
  - "pdm_lite": the carla_garage/PlanT route-log format used by the
    `autonomousvision/PDM_Lite_Carla_LB2` HF dataset (`<route>/measurements/*.json.gz`
    + `<route>/rgb/*.jpg`). It has no Q-values, so waypoints are derived from each
    frame's future `ego_matrix` poses and Q-value targets are left at zero.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union
import gzip
import json
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class WorldOnRailsDataset(Dataset):
    """
    Dataset loader for World on Rails offline distillation training.
    """
    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        num_rails: int = 9,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        synthetic_samples: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_rails = num_rails
        self.transform = transform
        self.is_train = is_train
        self.synthetic_samples = synthetic_samples

        self.samples = []
        if synthetic_samples > 0:
            self.samples = list(range(synthetic_samples))
            self.is_synthetic = True
        elif not os.path.exists(data_dir):
            self.samples = list(range(20))
            self.is_synthetic = True
        else:
            self.is_synthetic = False
            self._index_dataset()

    def _index_dataset(self):
        """Indexes all trajectory frames in the dataset directory (recursively, to
        allow a `<root>/<town>/<route>/...` layout as used by PDM-Lite)."""
        candidate_dirs = sorted(glob.glob(os.path.join(self.data_dir, "**"), recursive=True))
        for r_dir in candidate_dirs:
            if not os.path.isdir(r_dir):
                continue

            data_json_path = os.path.join(r_dir, "data.json")
            measurements_dir = os.path.join(r_dir, "measurements")
            if os.path.exists(data_json_path):
                try:
                    with open(data_json_path, "r") as f:
                        data = json.load(f)
                    for frame_info in data:
                        frame_info["route_dir"] = r_dir
                        frame_info["format"] = "wor"
                        self.samples.append(frame_info)
                except Exception as e:
                    print(f"[Warning] Failed to parse {data_json_path}: {e}")
            elif os.path.isdir(measurements_dir):
                self._index_pdm_lite_route(r_dir, measurements_dir)

        if len(self.samples) == 0:
            print(f"[Warning] No frames found in {self.data_dir}. Falling back to synthetic mode.")
            self.is_synthetic = True
            self.samples = list(range(20))

    def _index_pdm_lite_route(self, route_dir: str, measurements_dir: str, pred_len: int = 5):
        """Indexes one PDM-Lite-format route (carla_garage log layout).

        Each frame's waypoint target is derived from the ego poses of the next
        `pred_len` frames, transformed into the current frame's coordinate system
        (same approach as PlanT's dataset.py). No Q-values are available here.
        """
        meas_files = sorted(glob.glob(os.path.join(measurements_dir, "*.json.gz")))
        rgb_dir = os.path.join(route_dir, "rgb")
        num_frames = len(meas_files)
        if num_frames < pred_len + 6:
            return

        for i in range(5, num_frames - pred_len - 2):
            frame_id = os.path.basename(meas_files[i]).split(".")[0]
            rgb_path = os.path.join(rgb_dir, f"{frame_id}.jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")

            self.samples.append({
                "format": "pdm_lite",
                "rgb_path": rgb_path,
                "meas_path": meas_files[i],
                "future_meas_paths": meas_files[i + 1:i + 1 + pred_len],
                "route_dir": route_dir
            })

    def __len__(self) -> int:
        return len(self.samples)

    # carla_garage's raw command ids -> the WoR policy's 6-way command space
    # (LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT).
    _PDM_LITE_COMMAND_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def _load_pdm_lite_sample(self, item: Dict) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """Loads one PDM-Lite-format frame: RGB image, speed, command, and
        waypoints derived from the ego-frame-relative future trajectory."""
        rgb_path = item["rgb_path"]
        if os.path.exists(rgb_path):
            img = Image.open(rgb_path).convert("RGB")
            img = img.resize((self.img_size[1], self.img_size[0]))
            rgb = np.array(img, dtype=np.uint8)
        else:
            rgb = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        with gzip.open(item["meas_path"], "rt") as f:
            meas = json.load(f)
        speed = float(meas.get("speed", 0.0))
        raw_command = int(meas.get("command", meas.get("next_command", 4)))
        command = self._PDM_LITE_COMMAND_MAP.get(raw_command, 3)

        ref_matrix = np.array(meas["ego_matrix"], dtype=np.float64)
        ref_inv = np.linalg.inv(ref_matrix)

        waypoints = []
        for fut_path in item["future_meas_paths"]:
            try:
                with gzip.open(fut_path, "rt") as f:
                    fut_meas = json.load(f)
                fut_matrix = np.array(fut_meas["ego_matrix"], dtype=np.float64)
                rel = ref_inv @ fut_matrix
                waypoints.append([rel[0, 3], rel[1, 3]])
            except Exception:
                waypoints.append([0.0, 0.0])

        while len(waypoints) < 5:
            waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])

        return rgb, speed, command, np.array(waypoints[:5], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_synthetic:
            # Synthetic tensor generation for fast verification
            rgb = np.random.randint(0, 255, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            speed = float(np.random.uniform(0.0, 30.0))
            command = int(np.random.randint(0, 4))
            target_q = np.random.randn(self.num_rails).astype(np.float32)
            target_waypoints = np.random.randn(5, 2).astype(np.float32) * 5.0
        elif self.samples[idx].get("format") == "pdm_lite":
            rgb, speed, command, target_waypoints = self._load_pdm_lite_sample(self.samples[idx])
            target_q = np.zeros(self.num_rails, dtype=np.float32)
        else:
            item = self.samples[idx]
            rgb_path = item.get("rgb_path", os.path.join(item.get("route_dir", ""), "rgbs", f"{idx:05d}.jpg"))
            if os.path.exists(rgb_path):
                img = Image.open(rgb_path).convert("RGB")
                img = img.resize((self.img_size[1], self.img_size[0]))
                rgb = np.array(img, dtype=np.uint8)
            else:
                rgb = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

            speed = float(item.get("speed", 0.0))
            command = int(item.get("command", item.get("cmd", 2)))
            target_q = np.array(item.get("q_values", np.zeros(self.num_rails)), dtype=np.float32)
            target_waypoints = np.array(item.get("waypoints", np.zeros((5, 2))), dtype=np.float32)

        # Convert to PyTorch tensors (CHW RGB format normalized to [0, 1])
        try:
            rgb_tensor = torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            target_q_tensor = torch.as_tensor(target_q, dtype=torch.float32)
            target_waypoints_tensor = torch.as_tensor(target_waypoints, dtype=torch.float32)
        except Exception:
            rgb_tensor = torch.tensor(rgb.tolist(), dtype=torch.float32).permute(2, 0, 1) / 255.0
            target_q_tensor = torch.tensor(target_q.tolist(), dtype=torch.float32)
            target_waypoints_tensor = torch.tensor(target_waypoints.tolist(), dtype=torch.float32)

        speed_tensor = torch.tensor([speed], dtype=torch.float32)
        command_tensor = torch.tensor(command, dtype=torch.long)

        return {
            "rgb": rgb_tensor,
            "speed": speed_tensor,
            "command": command_tensor,
            "target_q": target_q_tensor,
            "target_waypoints": target_waypoints_tensor
        }


def create_wor_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    is_train: bool = True,
    synthetic_samples: int = 0
) -> DataLoader:
    """Creates a DataLoader for World on Rails training/validation."""
    dataset = WorldOnRailsDataset(
        data_dir=data_dir,
        is_train=is_train,
        synthetic_samples=synthetic_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
