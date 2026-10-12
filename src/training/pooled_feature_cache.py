"""TODO A59 (S-125): pooled-token feature cache for head-only CARLA training.

The Qwen head applies `AdaptiveAvgPool2d(vision_grid)` to the frozen encoder's map *before* any learnable layer, and nothing random sits upstream of the frozen
encoder in the J-family recipe (`color_aug_prob 0`, stored recovery frames, deterministic overlay). So the pooled (C, gh, gw) map of a frame is the same on every
epoch of every arm, and pooling an already (gh, gw) map to (gh, gw) is the identity: the cached map goes in as `vision_features` with no model change.

Storage: one raw fp16 memmap `<prefix>.f16` of shape (N, C, gh, gw) (48 KB per frame at C = 1512, 4x4: 20 GB for 427k frames), `<prefix>.done` (N uint8 flags,
resumable and shardable build) and `<prefix>.json` (the frame keys = paths relative to the data dir, the pixel tag, and the settings the cache was built with).
`WorldOnRailsDataset` switches to it when the environment variable WOR_POOLED_CACHE holds a comma-separated prefix list (set by `train_wor.py --pooled_cache`).
A pixel-tag mismatch, a missing frame or an incomplete cache is a hard error, never a silent fallback (same contract as the full-map cache).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def cache_files(prefix: str) -> Tuple[str, str, str]:
    return prefix + ".json", prefix + ".f16", prefix + ".done"


def rel_key(rgb_path: str, data_dir: str) -> str:
    """Frame key: the image path relative to the data dir, with forward slashes (the same on the build box, the training box and Windows)."""
    return os.path.relpath(rgb_path, data_dir).replace("\\", "/")


def init_cache(prefix: str, keys: Sequence[str], channels: int, grid: Tuple[int, int], meta: Dict) -> None:
    """Create the meta json and the (zero-filled) fp16 and done files. Idempotent for the same keys."""
    jp, fp, dp = cache_files(prefix)
    N, (gh, gw) = len(keys), grid
    if os.path.exists(jp):
        old = json.load(open(jp))
        assert old["keys"] == list(keys) and old["channels"] == channels and tuple(old["grid"]) == (gh, gw), f"{jp} exists with a different layout"
        return
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    np.memmap(fp, dtype=np.float16, mode="w+", shape=(N, channels, gh, gw)).flush()
    np.memmap(dp, dtype=np.uint8, mode="w+", shape=(N,)).flush()
    tmp = jp + ".tmp"
    json.dump({"keys": list(keys), "n": N, "channels": channels, "grid": [gh, gw], **meta}, open(tmp, "w"))
    os.replace(tmp, jp)  # written last: its existence means the layout is ready


class PooledFeatureCache:
    """Read side: key -> (C, gh, gw) fp16 array from the memmap. The memmaps open lazily in each DataLoader worker (a memmap does not pickle)."""

    def __init__(self, prefixes: Sequence[str], pixel_tag: str, data_dir: str):
        self.index: Dict[str, Tuple[int, int]] = {}
        self.caches: List[Dict] = []
        want = os.path.basename(os.path.normpath(data_dir))
        for p in prefixes:
            jp, fp, dp = cache_files(p)
            meta = json.load(open(jp))
            if meta.get("data_dir_name") != want:
                continue  # a cache built for another data dir (e.g. the validation set)
            if meta["pixel_tag"] != pixel_tag:
                raise ValueError(f"pooled cache {p} was built for pixel tag {meta['pixel_tag']!r}, this run uses {pixel_tag!r} "
                                 f"(img_size / crop / overlay / route_key differ): rebuild it")
            done = np.memmap(dp, dtype=np.uint8, mode="r", shape=(meta["n"],))
            if int(done.min()) == 0:
                raise RuntimeError(f"pooled cache {p} is incomplete ({int((done == 0).sum())} of {meta['n']} frames missing): finish build_pooled_cache.py")
            ci = len(self.caches)
            self.caches.append({"meta": meta, "path": fp, "mm": None})
            for r, k in enumerate(meta["keys"]):
                self.index[k] = (ci, r)
        if not self.caches:
            raise RuntimeError(f"no pooled cache in {list(prefixes)} was built for data dir {want!r}")
        self.channels = self.caches[0]["meta"]["channels"]
        self.grid = tuple(self.caches[0]["meta"]["grid"])

    def get(self, key: str) -> np.ndarray:
        try:
            ci, r = self.index[key]
        except KeyError:
            raise RuntimeError(f"frame {key!r} is not in the pooled cache (WOR_POOLED_CACHE): this dataset mode never encodes live, so a missing frame is a hard error "
                               f"(rebuild with the same --data_dir / --use_augmented_camera as the run)") from None
        c = self.caches[ci]
        if c["mm"] is None:
            m = c["meta"]
            c["mm"] = np.memmap(c["path"], dtype=np.float16, mode="r", shape=(m["n"], m["channels"], m["grid"][0], m["grid"][1]))
        return np.array(c["mm"][r])  # a copy: memmap slices are read-only

    def __getstate__(self):
        d = dict(self.__dict__)
        d["caches"] = [{**c, "mm": None} for c in self.caches]
        return d


def prefixes_from_env(value: Optional[str]) -> List[str]:
    return [x for x in (value or "").split(",") if x]
