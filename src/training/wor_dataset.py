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
        synthetic_samples: int = 0,
        cache_decoded: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_rails = num_rails
        self.transform = transform
        self.is_train = is_train
        self.synthetic_samples = synthetic_samples
        # JPEG decode + resize is the same work every epoch for a frame that never
        # changes, and it's what capped throughput at ~280-340 samples/sec regardless
        # of batch size (batch size only changes how many already-decoded samples get
        # grouped per GPU step - it can't speed up decoding itself). Caching each
        # decoded+resized frame as a raw .npy next to its source .jpg pays that cost
        # once instead of once per epoch; ~196KB/frame at 256x256x3 uint8.
        self.cache_decoded = cache_decoded

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

    # carla_garage's raw command ids -> the WoR policy's 6-way command space
    # (LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT).
    _PDM_LITE_COMMAND_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def _index_pdm_lite_route(self, route_dir: str, measurements_dir: str, pred_len: int = 5):
        """Indexes one PDM-Lite-format route (carla_garage log layout).

        Every measurement file is gzip-decompressed exactly once, here, rather than
        from __getitem__ - which would otherwise re-open up to `pred_len + 1` gzip
        files per sample on every single epoch (the dominant cost of an offline
        training run - see the "36/218 vision backbone parameters matched" episode
        with GPU util stuck at 17%). Waypoint targets are derived from the ego poses
        of the next `pred_len` frames, transformed into the current frame's
        coordinate system (same approach as PlanT's dataset.py). No Q-values are
        available here.
        """
        meas_files = sorted(glob.glob(os.path.join(measurements_dir, "*.json.gz")))
        rgb_dir = os.path.join(route_dir, "rgb")
        num_frames = len(meas_files)
        if num_frames < pred_len + 6:
            return

        parsed = [None] * num_frames
        for i, meas_path in enumerate(meas_files):
            try:
                with gzip.open(meas_path, "rt") as f:
                    meas = json.load(f)
                raw_command = int(meas.get("command", meas.get("next_command", 4)))
                parsed[i] = {
                    "speed": float(meas.get("speed", 0.0)),
                    "command": self._PDM_LITE_COMMAND_MAP.get(raw_command, 3),
                    "ego_matrix": np.array(meas["ego_matrix"], dtype=np.float64)
                }
            except Exception:
                parsed[i] = None

        for i in range(5, num_frames - pred_len - 2):
            cur = parsed[i]
            if cur is None:
                continue
            frame_id = os.path.basename(meas_files[i]).split(".")[0]
            rgb_path = os.path.join(rgb_dir, f"{frame_id}.jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")

            ref_inv = np.linalg.inv(cur["ego_matrix"])
            waypoints = []
            for j in range(i + 1, i + 1 + pred_len):
                fut = parsed[j] if j < num_frames else None
                if fut is not None:
                    rel = ref_inv @ fut["ego_matrix"]
                    waypoints.append([float(rel[0, 3]), float(rel[1, 3])])
                else:
                    waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])

            self.samples.append({
                "format": "pdm_lite",
                "rgb_path": rgb_path,
                "speed": cur["speed"],
                "command": cur["command"],
                "waypoints": waypoints
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, rgb_path: str) -> np.ndarray:
        """Loads one RGB frame at self.img_size, transparently caching the
        decoded+resized array as a sibling .npy file so later epochs (or later runs
        entirely) skip JPEG decode. Cache filename is keyed by img_size so switching
        resolutions can't silently serve a stale-size array. Written via a temp file +
        atomic rename so a worker process crashing mid-write can't leave a corrupt
        cache entry for the next epoch to read.
        """
        h, w = self.img_size
        cache_path = f"{rgb_path}.{h}x{w}.npy" if self.cache_decoded else None

        if cache_path is not None and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fall through and re-decode if the cache file is corrupt.

        if os.path.exists(rgb_path):
            img = Image.open(rgb_path).convert("RGB")
            img = img.resize((w, h))
            rgb = np.array(img, dtype=np.uint8)
        else:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        if cache_path is not None:
            try:
                # np.save appends ".npy" if the target doesn't already end with it, so
                # the tmp name must end in .npy too or the rename below targets the
                # wrong (unsuffixed) path.
                tmp_path = f"{cache_path}.tmp{os.getpid()}.npy"
                np.save(tmp_path, rgb)
                os.replace(tmp_path, cache_path)
            except Exception:
                pass  # Caching is a pure optimization - never let it fail the sample.

        return rgb

    def _load_pdm_lite_sample(self, item: Dict) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """Loads one PDM-Lite-format frame's RGB image. speed/command/waypoints were
        already parsed once at index time in `_index_pdm_lite_route`."""
        rgb = self._load_rgb(item["rgb_path"])
        return rgb, item["speed"], item["command"], np.array(item["waypoints"], dtype=np.float32)

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
            rgb = self._load_rgb(rgb_path)

            speed = float(item.get("speed", 0.0))
            command = int(item.get("command", item.get("cmd", 2)))
            target_q = np.array(item.get("q_values", np.zeros(self.num_rails)), dtype=np.float32)
            target_waypoints = np.array(item.get("waypoints", np.zeros((5, 2))), dtype=np.float32)

        # RGB stays uint8 HWC here: converting to float32 CHW in the worker would
        # quadruple both the CPU work and the bytes crossing PCIe (786KB vs 196KB per
        # frame). The trainer does the permute/scale on the GPU instead, where it's
        # nearly free - and HWC uint8 is already the channels_last layout the conv
        # kernels want, so the permute costs no copy.
        try:
            rgb_tensor = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.uint8)
            target_q_tensor = torch.as_tensor(target_q, dtype=torch.float32)
            target_waypoints_tensor = torch.as_tensor(target_waypoints, dtype=torch.float32)
        except Exception:
            rgb_tensor = torch.tensor(rgb.tolist(), dtype=torch.uint8)
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
    synthetic_samples: int = 0,
    cache_decoded: bool = True
) -> DataLoader:
    """Creates a DataLoader for World on Rails training/validation."""
    dataset = WorldOnRailsDataset(
        data_dir=data_dir,
        is_train=is_train,
        synthetic_samples=synthetic_samples,
        cache_decoded=cache_decoded
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
