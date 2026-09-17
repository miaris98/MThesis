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
from typing import Callable, Dict, List, Optional, Tuple
import gzip
import json
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config.camera import preprocess_rgb


def _camera_offset_matrix(translation_y: float, rotation_yaw_deg: float) -> np.ndarray:
    """4x4 rigid transform mapping a point given in the *augmented* camera's own local
    frame into the *true* ego's local frame - the same lateral+yaw offset
    DataAgent.augment_camera (carla_garage/team_code/data_agent.py) applies when it mounts
    the second camera that renders rgb_augmented/. translation_y is meters along the
    ego's right axis, rotation_yaw_deg is degrees about the ego's up axis - both are
    recorded per-frame as measurements['augmentation_translation']/['augmentation_rotation'].
    Its inverse (a plain transpose+negate for a rigid transform) reprojects a point
    recorded in the true frame - e.g. an already-computed waypoint or route point - into
    what the augmented camera would have seen, which is the corrective label recovery
    augmentation needs. See struggle-solutions.md S-020 for why the label cannot simply be
    recomputed without this: the image really is a different render, so the target must be
    re-expressed in that render's own frame, not left as the on-route target.
    """
    theta = np.radians(rotation_yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    m[1, 3] = translation_y
    return m


def _reproject_points(points_xy, offset_matrix_inv: np.ndarray) -> list:
    """Applies a rigid local-frame transform to a list of [x, y] points (z=0 assumed -
    waypoints and route points are both stored as ground-plane offsets, never full poses)."""
    out = []
    for x, y in points_xy:
        p = offset_matrix_inv @ np.array([x, y, 0.0, 1.0])
        out.append([float(p[0]), float(p[1])])
    return out


def feature_cache_path(rgb_path: str, pixel_tag: str, feature_cache_tag: str) -> str:
    """Sidecar path for one frame's cached frozen-backbone output.

    A module-level function, not just a dataset method, so build_feature_cache.py can compute
    the exact same path the dataset will later look for without having to construct a dataset
    in "feature-loading" mode first (which would try to load features that do not exist yet).
    Keyed on the pixel tag (resolution/crop/overlay - a shape change makes a cached map
    meaningless) and the backbone identity (a feature cache built from one architecture's
    weights would silently corrupt training under another's) - see WorldOnRailsDataset's
    feature_cache_tag docstring for why a missing entry is a hard error rather than a silent
    per-sample fallback.
    """
    return f"{rgb_path}.{pixel_tag}.{feature_cache_tag}.feat.npy"


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
        cache_decoded: bool = True,
        route_points: int = 4,
        crop_bottom_frac: float = 0.0,
        route_overlay: bool = False,
        overlay_kwargs: Optional[Dict] = None,
        feature_cache_tag: Optional[str] = None,
        use_augmented_camera: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        # Fraction of image height to cut off the bottom before resizing. 0.25 on PDM-Lite's
        # 1024x512 render reproduces TransFuser++'s 1024x384 crop, which removes the ego bonnet
        # and is the geometry the CARLA-pretrained encoder was trained on. 0.0 keeps the legacy
        # full-frame behaviour so old runs stay reproducible.
        self.crop_bottom_frac = crop_bottom_frac
        # Render the planned route into the image like a reversing camera's guide lines, so the
        # route and the road share a spatial frame before the frozen encoder sees either. Must
        # be set identically at evaluation or it becomes another 11.4 entry - the agent reads
        # the same flag off the checkpoint's run_config.json.
        self.route_overlay = route_overlay
        self.overlay_kwargs = overlay_kwargs or {}
        self.num_rails = num_rails
        self.transform = transform
        self.is_train = is_train
        self.synthetic_samples = synthetic_samples
        # How many points of the ego-frame planned route to expose as the policy's
        # navigation input. PDM-Lite leaves the legacy `command` enum at LANEFOLLOW on
        # every frame, so without this the policy has no way to know which way the
        # route turns and correctly learns to drive straight. The full 20-point route
        # correlates ~0.84 with the lateral target and starts to hand over the answer;
        # a sparse subsample keeps this closer to the Leaderboard's sparse-goal
        # convention, and this is a knob so the leakage can be ablated.
        self.route_points = route_points
        # JPEG decode + resize is the same work every epoch for a frame that never
        # changes, and it's what capped throughput at ~280-340 samples/sec regardless
        # of batch size (batch size only changes how many already-decoded samples get
        # grouped per GPU step - it can't speed up decoding itself). Caching each
        # decoded+resized frame as a raw .npy next to its source .jpg pays that cost
        # once instead of once per epoch; ~196KB/frame at 256x256x3 uint8.
        self.cache_decoded = cache_decoded
        # When set, __getitem__ returns the frozen backbone's cached output for each frame
        # instead of the raw pixels, and skips JPEG decode entirely. This is a bitwise-lossless
        # optimization, not an approximation: the backbone is frozen and nothing upstream of it
        # is randomized, so its output for a given frame is identical on every epoch of a run,
        # and repeating that computation 50 times is pure waste. Requires the cache to already
        # exist for every frame (build it with build_feature_cache.py first) - a missing entry
        # raises rather than silently falling back to live rgb, because a per-sample fallback
        # would mean some frames in a batch carry "vision_features" and others carry "rgb",
        # which the default collate can't merge into one batch key.
        self.feature_cache_tag = feature_cache_tag
        # Recovery-data augmentation (struggle-solutions.md S-020/S-021's follow-up): PDM-Lite's
        # DataAgent renders a second camera per frame at a random per-route lateral+yaw offset
        # (measurements' augmentation_translation/augmentation_rotation), saved as
        # rgb_augmented/. Behaviour cloning otherwise never sees an off-center state, since the
        # expert's own trajectory is centered by construction - closed-loop evaluation showed
        # this as outside_route_lanes + vehicle_blocked with no recovery (challenges_03 3.15).
        # Train-only: mixing recovery views into validation would make the held-out metric
        # measure something other than on-route driving, and stop it being comparable across
        # runs that don't use this flag.
        self.use_augmented_camera = bool(use_augmented_camera) and is_train

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

    def _subsample_route(self, route) -> List[List[float]]:
        """Reduces the ego-frame planned route to `self.route_points` evenly spaced
        points. Routes shorten near the end of an episode, so short ones are padded
        by repeating the last point rather than dropped - a truncated route still
        carries the turn direction.
        """
        n = self.route_points
        if not route:
            return [[0.0, 0.0] for _ in range(n)]
        arr = np.asarray(route, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return [[0.0, 0.0] for _ in range(n)]
        idx = np.linspace(0, arr.shape[0] - 1, n).round().astype(int)
        return arr[idx, :2].tolist()

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

        When self.use_augmented_camera is set, each frame that has a corresponding
        rgb_augmented/ render also contributes a second sample: the same future
        trajectory, re-expressed in the augmented camera's own (laterally/yaw offset)
        frame instead of the true ego frame - a corrective "how to get back onto the
        route" target, matched to an image that actually shows the off-center view.
        See _camera_offset_matrix and struggle-solutions.md S-020.
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
                    "ego_matrix": np.array(meas["ego_matrix"], dtype=np.float64),
                    "route": self._subsample_route(meas.get("route")),
                    # The expert's *intended* speed, not its current one. This is the label for
                    # the target-speed head: PDM-Lite decides a target and a controller chases
                    # it, so `speed` lags the decision while `target_speed` is the decision.
                    # Falling back to the measured speed keeps older dumps loadable.
                    "target_speed": float(meas.get("target_speed", meas.get("speed", 0.0))),
                    # Present on PDM-Lite DataAgent dumps that ship rgb_augmented/ (confirmed
                    # against the released autonomousvision/PDM_Lite_Carla_LB2 archives - see
                    # struggle-solutions.md S-020); absent, and harmlessly defaulted to no
                    # offset, on the "wor" native format and any dump collected without
                    # --augment.
                    "augmentation_translation": float(meas.get("augmentation_translation", 0.0)),
                    "augmentation_rotation": float(meas.get("augmentation_rotation", 0.0)),
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
                "route": cur["route"],
                "target_speed": cur["target_speed"],
                "waypoints": waypoints
            })

            # Recovery-augmentation sample: a second, genuinely re-rendered camera at this
            # same instant, offset laterally/in yaw from the true ego pose (see
            # _camera_offset_matrix's docstring and S-020). Only emitted where the frame
            # actually exists - DataAgent only renders it "at the beginning of the route" per
            # its own comment, so coverage is partial and checked per-frame, not assumed.
            if self.use_augmented_camera:
                aug_rgb_path = os.path.join(route_dir, "rgb_augmented", f"{frame_id}.jpg")
                if not os.path.exists(aug_rgb_path):
                    aug_rgb_path = os.path.join(route_dir, "rgb_augmented", f"{frame_id}.png")
                if os.path.exists(aug_rgb_path):
                    offset = _camera_offset_matrix(
                        cur.get("augmentation_translation", 0.0),
                        cur.get("augmentation_rotation", 0.0))
                    offset_inv = np.linalg.inv(offset)
                    aug_waypoints = _reproject_points(waypoints, offset_inv)
                    aug_route = _reproject_points(cur["route"], offset_inv)
                    self.samples.append({
                        "format": "pdm_lite",
                        "rgb_path": aug_rgb_path,
                        "speed": cur["speed"],
                        "command": cur["command"],
                        "route": aug_route,
                        "target_speed": cur["target_speed"],
                        "waypoints": aug_waypoints,
                        "is_recovery_augmented": True
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def pixel_cache_tag(self) -> str:
        """The part of the cache key that identifies which pixel transform was applied -
        shared between the decoded-pixel cache and the feature cache, since a feature cache
        built from one img_size/crop/overlay combination is meaningless for another."""
        h, w = self.img_size
        # Every transform that changes the pixels has to be in the cache key, not just the
        # size: 512x192 cropped, 512x192 squashed and 512x192 with a route drawn on it are three
        # different images at identical dimensions, and serving one for another is exactly the
        # kind of silent train/eval mismatch 11.4 is a catalogue of.
        return f"{h}x{w}{'c' if self.crop_bottom_frac else ''}{'o' if self.route_overlay else ''}"

    def _feature_cache_path(self, rgb_path: str) -> Optional[str]:
        """This instance's sidecar path for one frame, or None when feature caching is off."""
        if not self.feature_cache_tag:
            return None
        return feature_cache_path(rgb_path, self.pixel_cache_tag(), self.feature_cache_tag)

    def _load_features(self, rgb_path: str) -> np.ndarray:
        """Loads the cached frozen-backbone output for one frame. Raises if it is missing -
        see feature_cache_tag's docstring for why this cannot silently fall back to live
        encoding per-sample."""
        cache_path = self._feature_cache_path(rgb_path)
        try:
            return np.load(cache_path)
        except Exception as exc:
            raise RuntimeError(
                f"feature_cache_tag={self.feature_cache_tag!r} is set but no cached feature "
                f"found at {cache_path}. Run build_feature_cache.py for this data_dir/img_size/"
                f"crop_bottom_frac/route_overlay/backbone combination first - this dataset mode "
                f"never computes features live, so a missing entry is a hard error rather than "
                f"a silent slowdown or a silently mixed batch.") from exc

    def _load_rgb(self, rgb_path: str, route_xy=None) -> np.ndarray:
        """Loads one RGB frame at self.img_size, transparently caching the
        decoded+resized array as a sibling .npy file so later epochs (or later runs
        entirely) skip JPEG decode. Cache filename is keyed by img_size so switching
        resolutions can't silently serve a stale-size array. Written via a temp file +
        atomic rename so a worker process crashing mid-write can't leave a corrupt
        cache entry for the next epoch to read.
        """
        h, w = self.img_size
        tag = self.pixel_cache_tag()
        cache_path = f"{rgb_path}.{tag}.npy" if self.cache_decoded else None

        if cache_path is not None and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fall through and re-decode if the cache file is corrupt.

        if os.path.exists(rgb_path):
            # Deliberately the same function the agent calls at evaluation time, not an
            # equivalent-looking copy of it. Crop removes PDM-Lite's constant bonnet strip;
            # resizing a 1024x384 crop to a square would squash horizontal geometry ~2.7x
            # relative to vertical. See src/config/camera.py for why both live in one place.
            rgb = preprocess_rgb(Image.open(rgb_path).convert("RGB"),
                                 img_size=(h, w), crop_bottom_frac=self.crop_bottom_frac,
                                 route_xy=route_xy, overlay=self.route_overlay,
                                 overlay_kwargs=self.overlay_kwargs)
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
        already parsed once at index time in `_index_pdm_lite_route`.

        The overlay is handed exactly the route the policy's own route MLP receives, not the
        denser 20-point original. That keeps it a pure change of *representation*: the same
        information, additionally presented in the image plane where the frozen encoder can
        associate it with the road. Drawing the full route instead would hand the network more
        of the answer than the ablation baseline gets (the 20-point route correlates ~0.84 with
        the lateral target), and the comparison would no longer isolate the overlay.
        """
        rgb = self._load_rgb(item["rgb_path"], route_xy=item.get("route"))
        return rgb, item["speed"], item["command"], np.array(item["waypoints"], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        features = None
        if self.is_synthetic:
            # Synthetic tensor generation for fast verification
            rgb = np.random.randint(0, 255, (self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            speed = float(np.random.uniform(0.0, 30.0))
            command = int(np.random.randint(0, 4))
            target_q = np.random.randn(self.num_rails).astype(np.float32)
            target_waypoints = np.random.randn(5, 2).astype(np.float32) * 5.0
            route = np.random.randn(self.route_points, 2).astype(np.float32)
        elif self.samples[idx].get("format") == "pdm_lite":
            item = self.samples[idx]
            if self.feature_cache_tag:
                # Metadata only - the frozen backbone's output is read from disk below instead
                # of being recomputed from pixels, so the JPEG is never even opened.
                features = self._load_features(item["rgb_path"])
                rgb = None
                speed, command = item["speed"], item["command"]
                target_waypoints = np.array(item["waypoints"], dtype=np.float32)
            else:
                rgb, speed, command, target_waypoints = self._load_pdm_lite_sample(item)
            target_q = np.zeros(self.num_rails, dtype=np.float32)
            route = np.array(item["route"], dtype=np.float32)
        else:
            item = self.samples[idx]
            rgb_path = item.get("rgb_path", os.path.join(item.get("route_dir", ""), "rgbs", f"{idx:05d}.jpg"))
            if self.feature_cache_tag:
                features = self._load_features(rgb_path)
                rgb = None
            else:
                rgb = self._load_rgb(rgb_path)

            speed = float(item.get("speed", 0.0))
            command = int(item.get("command", item.get("cmd", 2)))
            target_q = np.array(item.get("q_values", np.zeros(self.num_rails)), dtype=np.float32)
            target_waypoints = np.array(item.get("waypoints", np.zeros((5, 2))), dtype=np.float32)
            route = np.array(item.get("route", np.zeros((self.route_points, 2))), dtype=np.float32)

        # RGB stays uint8 HWC here: converting to float32 CHW in the worker would
        # quadruple both the CPU work and the bytes crossing PCIe (786KB vs 196KB per
        # frame). The trainer does the permute/scale on the GPU instead, where it's
        # nearly free - and HWC uint8 is already the channels_last layout the conv
        # kernels want, so the permute costs no copy.
        try:
            target_q_tensor = torch.as_tensor(target_q, dtype=torch.float32)
            target_waypoints_tensor = torch.as_tensor(target_waypoints, dtype=torch.float32)
        except Exception:
            target_q_tensor = torch.tensor(target_q.tolist(), dtype=torch.float32)
            target_waypoints_tensor = torch.tensor(target_waypoints.tolist(), dtype=torch.float32)

        speed_tensor = torch.tensor([speed], dtype=torch.float32)
        command_tensor = torch.tensor(command, dtype=torch.long)

        # Label for the target-speed head. Always emitted (it costs 4 bytes) so enabling the
        # head never requires rebuilding the loader or invalidating a cache; AuxiliaryHeads
        # skips any target whose head is disabled.
        if self.is_synthetic:
            tgt_speed = float(np.random.uniform(0.0, 20.0))
        else:
            tgt_speed = float(self.samples[idx].get("target_speed", speed))

        sample = {
            "speed": speed_tensor,
            "command": command_tensor,
            "route": torch.as_tensor(route, dtype=torch.float32),
            "target_q": target_q_tensor,
            "target_waypoints": target_waypoints_tensor,
            "target_speed": torch.tensor(tgt_speed, dtype=torch.float32)
        }
        # Exactly one of "rgb"/"vision_features" is present, never both and never neither -
        # every sample in a batch takes the same branch above (feature_cache_tag is dataset-wide),
        # so the collated batch always has a consistent key set.
        if features is not None:
            sample["vision_features"] = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        else:
            try:
                sample["rgb"] = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.uint8)
            except Exception:
                sample["rgb"] = torch.tensor(rgb.tolist(), dtype=torch.uint8)
        return sample




# Re-exported so every existing `from src.training.wor_dataset import create_wor_dataloader`
# (and the other three) keeps working. The implementations moved to wor_dataloaders.py when this
# module outgrew the 500-line limit; see that module's docstring for the seam. The import sits
# at the bottom, not the top, because wor_dataloaders imports WorldOnRailsDataset from here -
# putting it above the class definition would be a circular import.
from src.training.wor_dataloaders import (  # noqa: E402,F401
    _wrap_loader, create_wor_dataloader, route_group_split,
    create_wor_train_val_dataloaders)
