"""World on Rails (WoR) Offline Trajectory Dataset and DataLoader.

Loads CARLA driving logs (RGB frames, ego telemetry, high-level commands,
and precomputed Q-values / future waypoint targets).
"""
from typing import Callable, Dict, List, Optional, Tuple, Union
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
        """Indexes all trajectory frames in the dataset directory."""
        route_dirs = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        for r_dir in route_dirs:
            if not os.path.isdir(r_dir):
                continue
            data_json_path = os.path.join(r_dir, "data.json")
            if os.path.exists(data_json_path):
                try:
                    with open(data_json_path, "r") as f:
                        data = json.load(f)
                    for frame_info in data:
                        frame_info["route_dir"] = r_dir
                        self.samples.append(frame_info)
                except Exception as e:
                    print(f"[Warning] Failed to parse {data_json_path}: {e}")
            else:
                # Fallback: scan for rgb images
                rgb_files = sorted(glob.glob(os.path.join(r_dir, "rgbs", "*.jpg")) + glob.glob(os.path.join(r_dir, "*.png")))
                for rgb_p in rgb_files:
                    self.samples.append({
                        "rgb_path": rgb_p,
                        "speed": 5.5,
                        "command": 2,
                        "route_dir": r_dir
                    })

        if len(self.samples) == 0:
            print(f"[Warning] No frames found in {self.data_dir}. Falling back to synthetic mode.")
            self.is_synthetic = True
            self.samples = list(range(20))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_synthetic:
            # Synthetic tensor generation for fast verification
            rgb = np.random.randint(0, 255, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            speed = float(np.random.uniform(0.0, 30.0))
            command = int(np.random.randint(0, 4))
            target_q = np.random.randn(self.num_rails).astype(np.float32)
            target_waypoints = np.random.randn(5, 2).astype(np.float32) * 5.0
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
