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
import math
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
        # 2026-11-07, route-window arm (train_wor.py --route_windows K): each read also packs the K cached route windows, (C, K) from
        # <prefix>.win<K>.f16, after the grid as extra columns -> (C, gh, gw + K/gh). Built by build_pooled_cache.py --route_windows K.
        self.route_windows = int(os.environ.get("WOR_ROUTE_WINDOWS", "0") or 0)
        # 2026-11-07 (spatial tests): which window file ("route" = placed on the route, "fixed" = the same K windows on a straight-ahead line),
        # the PCA-compressed full map instead of the pooled grid (WOR_PCA_FULL=K), and the per-cell box labels (WOR_BOX_LABELS=1).
        self.window_kind = os.environ.get("WOR_WINDOW_KIND", "route") or "route"
        self.pca_full = int(os.environ.get("WOR_PCA_FULL", "0") or 0)
        self.box_labels = os.environ.get("WOR_BOX_LABELS", "0") == "1"
        # 2026-11-08: WOR_ROUTE_WINDOWS stays the per-kind K (8); WOR_WINDOW_KIND may join kinds with "+" ("route+side" = 8 + 16 windows).
        # WOR_WINDOW_LABELS=1 serves the path-relative window targets; WOR_DECODER_STATS=1 appends the 10 TF++ decoder-statistic channels and
        # WOR_WINDOW_COORDS=1 two ground-coordinate channels (zero for grid cells) to every token, in that order, after the encoder channels.
        self.window_kinds = [k for k in self.window_kind.split("+") if k]
        self.window_labels = os.environ.get("WOR_WINDOW_LABELS", "0") == "1"
        self.decoder_stats = os.environ.get("WOR_DECODER_STATS", "0") == "1"
        self.window_coords = os.environ.get("WOR_WINDOW_COORDS", "0") == "1"
        self.extra_channels = (DSTAT_CHANNELS if self.decoder_stats else 0) + (2 if self.window_coords else 0)
        # 2026-11-08 wave 2: WOR_PRIVILEGED=M serves the M nearest logged objects (<prefix>.boxtok<M>.f16, build_box_targets.py) INSTEAD of
        # the camera features (idea 8, the privileged teacher); WOR_FULL_LABELS=HxW serves the full-map per-cell labels (idea 12).
        self.privileged = int(os.environ.get("WOR_PRIVILEGED", "0") or 0)
        # 2026-11-08 teacher ceiling: WOR_PRIVILEGED_VARIANT picks the token file ("" = all objects, "_vis" = only objects in the camera
        # image, "_visns" = in the image and speed zeroed); WOR_PRIVILEGED_NOISE="sigma_m,drop" simulates a detector (train and val).
        self.privileged_variant = os.environ.get("WOR_PRIVILEGED_VARIANT", "") or ""
        _pn = (os.environ.get("WOR_PRIVILEGED_NOISE", "") or "0,0").split(",")
        self.privileged_noise = (float(_pn[0]), float(_pn[1]))
        self.full_labels = os.environ.get("WOR_FULL_LABELS", "") or ""
        if (self.window_labels or self.window_coords) and not self.route_windows:
            raise RuntimeError("WOR_WINDOW_LABELS / WOR_WINDOW_COORDS need route windows (WOR_ROUTE_WINDOWS)")
        want = os.path.basename(os.path.normpath(data_dir))
        for p in prefixes:
            jp, fp, dp = cache_files(p)
            meta = json.load(open(jp))
            if meta.get("data_dir_name") != want:
                continue  # a cache built for another data dir (e.g. the validation set)
            if meta["pixel_tag"] != pixel_tag:
                raise ValueError(f"pooled cache {p} was built for pixel tag {meta['pixel_tag']!r}, this run uses {pixel_tag!r} "
                                 f"(img_size / crop / overlay / route_key differ): rebuild it")
            # A partial cache (build_pooled_cache.py --val_only, 2026-11-06) holds only the validation frames: each read checks its done flag.
            partial = bool(meta.get("partial"))
            done = np.memmap(dp, dtype=np.uint8, mode="r", shape=(meta["n"],))
            if not partial and int(done.min()) == 0:
                raise RuntimeError(f"pooled cache {p} is incomplete ({int((done == 0).sum())} of {meta['n']} frames missing): finish build_pooled_cache.py")
            ci = len(self.caches)
            win_path = pca_path = pca_shape = None
            win_sets = []
            if self.route_windows:
                for kind in self.window_kinds:
                    wp = window_file(p, self.route_windows, kind)
                    side = window_side_files(p, self.route_windows, kind)
                    need = [wp] + ([side["lab"]] if self.window_labels else []) + ([side["xy"]] if self.window_coords else []) \
                        + ([side["dstat"]] if self.decoder_stats else [])
                    for f in need:
                        if not os.path.exists(f):
                            raise RuntimeError(f"windows requested (K={self.route_windows}, {kind}) but {f} is missing: build_pooled_cache.py "
                                               f"--route_windows / --side_windows / --window_labels / --decoder_stats")
                    win_sets.append({"kind": kind, "m": window_count(self.route_windows, kind), "path": wp, **side})
                win_path = win_sets[0]["path"]
                if self.window_labels and not os.path.exists(window_label_valid_file(p)):
                    raise RuntimeError(f"window labels requested but {window_label_valid_file(p)} is missing")
            if self.decoder_stats and not os.path.exists(grid_dstat_file(p)):
                raise RuntimeError(f"decoder statistics requested but {grid_dstat_file(p)} is missing: build_pooled_cache.py --decoder_stats 1")
            if self.pca_full:
                pca_path = pca_file(p, self.pca_full)
                pca_shape = tuple(json.load(open(pca_path + ".json"))["shape"])
            if self.box_labels:
                for f in label_files(p):
                    if not os.path.exists(f):
                        raise RuntimeError(f"box labels requested but {f} is missing: build_pooled_cache.py --box_labels 1")
            self.caches.append({"meta": meta, "path": fp, "mm": None, "done_path": dp, "dm": None, "partial": partial,
                                "win_path": win_path, "wm": None, "pca_path": pca_path, "pca_shape": pca_shape, "pm": None,
                                "lab_paths": label_files(p), "lm": None, "lv": None,
                                "win_sets": win_sets, "wsm": {}, "dstat_path": grid_dstat_file(p), "dsm": None,
                                "wlv_path": window_label_valid_file(p), "wlv": None, "prefix": p, "btm": None, "flm": None, "flv": None})
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
        if c["partial"]:
            if c["dm"] is None:
                c["dm"] = np.memmap(c["done_path"], dtype=np.uint8, mode="r", shape=(c["meta"]["n"],))
            if not c["dm"][r]:
                raise RuntimeError(f"frame {key!r} is not encoded in the partial pooled cache {c['path']} (built with --val_only): "
                                   f"it is outside the validation split that cache was built for")
        m = c["meta"]
        if self.privileged:  # (M, BOX_FEATS) logged objects, the teacher's input
            from scripts.training.build_box_targets import BOX_FEATS, VALID_SLOT
            tok = np.array(self._mm(c, "btm", f"{c['prefix']}.boxtok{self.privileged}{self.privileged_variant}.f16",
                                    (m["n"], self.privileged, BOX_FEATS))[r], dtype=np.float32)
            if self.privileged_noise[0] > 0 or self.privileged_noise[1] > 0:   # a simulated detector: jitter positions, drop objects
                sig, drop = self.privileged_noise
                rng = np.random.default_rng()   # fresh entropy per call: forked loader workers would share np.random's state
                valid = tok[:, VALID_SLOT] > 0.5
                tok[valid, 0] += rng.normal(0.0, sig, valid.sum()) / 20.0
                tok[valid, 1] += rng.normal(0.0, sig, valid.sum()) / 10.0
                tok[valid & (rng.random(len(tok)) < drop)] = 0.0
            return tok
        if self.pca_full:  # the full stride-32 map, PCA-compressed: (K, H, W)
            if c["pm"] is None:
                c["pm"] = np.memmap(c["pca_path"], dtype=np.float16, mode="r", shape=c["pca_shape"])
            return np.array(c["pm"][r])
        if c["mm"] is None:
            c["mm"] = np.memmap(c["path"], dtype=np.float16, mode="r", shape=(m["n"], m["channels"], m["grid"][0], m["grid"][1]))
        arr = np.array(c["mm"][r])  # a copy: memmap slices are read-only
        gh, gw = arr.shape[1], arr.shape[2]
        if self.decoder_stats:
            arr = np.concatenate([arr, self._mm(c, "dsm", c["dstat_path"], (m["n"], DSTAT_CHANNELS, gh, gw))[r]], axis=0)
        if self.window_coords:
            arr = np.concatenate([arr, np.zeros((2, gh, gw), dtype=arr.dtype)], axis=0)
        if self.route_windows:
            parts = []
            for ws in c["win_sets"]:
                w = [self._mm(c, ("w", ws["kind"]), ws["path"], (m["n"], m["channels"], ws["m"]))[r]]
                if self.decoder_stats:
                    w.append(self._mm(c, ("d", ws["kind"]), ws["dstat"], (m["n"], DSTAT_CHANNELS, ws["m"]))[r])
                if self.window_coords:
                    w.append(self._mm(c, ("x", ws["kind"]), ws["xy"], (m["n"], 2, ws["m"]))[r])
                parts.append(np.concatenate(w, axis=0))
            win = np.concatenate(parts, axis=1)                               # (C', M) in kind order, route order within a kind
            assert win.shape[1] % gh == 0, f"{win.shape[1]} windows do not pack into {gh} rows"
            arr = np.concatenate([arr, win.reshape(arr.shape[0], gh, -1)], axis=2)   # (C', M) -> (C', gh, M/gh), row-major = window order
        return arr

    def _mm(self, c: Dict, slot, path: str, shape) -> np.memmap:
        """A read-only memmap of one side file, opened once per worker (slots: "dsm", ("w"|"d"|"x"|"l", kind))."""
        store = c if isinstance(slot, str) else c["wsm"]
        if store.get(slot) is None:
            store[slot] = np.memmap(path, dtype=np.float16, mode="r", shape=shape)
        return store[slot]

    def row_of(self, key: str) -> Tuple[int, int]:
        return self.index[key]

    def get_full_labels(self, key: str) -> Tuple[np.ndarray, float]:
        """(2, H, W) full-map per-cell labels (build_box_targets.py --full_grid HxW) and the frame's validity (2026-11-08, idea 12)."""
        ci, r = self.index[key]
        c = self.caches[ci]
        n = c["meta"]["n"]
        h, w = (int(v) for v in self.full_labels.lower().split("x"))
        lab = np.asarray(self._mm(c, "flm", f"{c['prefix']}.lab{h}x{w}.f16", (n, 2, h, w))[r], dtype=np.float32)
        if c["flv"] is None:
            c["flv"] = np.memmap(f"{c['prefix']}.lab{h}x{w}.valid.u8", dtype=np.uint8, mode="r", shape=(n,))
        return lab, float(c["flv"][r])

    def get_window_labels(self, key: str) -> Tuple[np.ndarray, float]:
        """(2, M) path-relative window targets in the same window order as get() (2026-11-08) and the frame's 0/1 validity."""
        ci, r = self.index[key]
        c = self.caches[ci]
        n = c["meta"]["n"]
        lab = np.concatenate([np.asarray(self._mm(c, ("l", ws["kind"]), ws["lab"], (n, 2, ws["m"]))[r], dtype=np.float32)
                              for ws in c["win_sets"]], axis=1)
        if c["wlv"] is None:
            c["wlv"] = np.memmap(c["wlv_path"], dtype=np.uint8, mode="r", shape=(n,))
        return lab, float(c["wlv"][r])

    def get_labels(self, key: str) -> Tuple[np.ndarray, float]:
        """(2, 4, 4) per-cell box labels (occupancy, nearest distance / 40 m) and a 0/1 validity flag (2026-11-07)."""
        ci, r = self.index[key]
        c = self.caches[ci]
        if c["lm"] is None:
            n = c["meta"]["n"]
            c["lm"] = np.memmap(c["lab_paths"][0], dtype=np.float16, mode="r", shape=(n, 2, 4, 4))
            c["lv"] = np.memmap(c["lab_paths"][1], dtype=np.uint8, mode="r", shape=(n,))
        return np.array(c["lm"][r], dtype=np.float32), float(c["lv"][r])

    def __getstate__(self):
        d = dict(self.__dict__)
        d["caches"] = [{**c, "mm": None, "dm": None, "wm": None, "pm": None, "lm": None, "lv": None, "wsm": {}, "dsm": None, "wlv": None,
                        "btm": None, "flm": None, "flv": None}
                       for c in self.caches]
        return d


class FusePCAReader:
    """2026-11-09, two-encoder arm: the PCA-compressed full map of a SECOND pooled cache (another encoder / pixel geometry, e.g. the SimLingo
    tower at 512x1024), looked up by the same frame key. Read-only; no pixel-tag check (the second cache is a different geometry on purpose),
    but it must cover every frame of the primary cache (a missing frame is a hard error)."""

    def __init__(self, prefix: str, k: int):
        jp, _, dp = cache_files(prefix)
        meta = json.load(open(jp))
        done = np.memmap(dp, dtype=np.uint8, mode="r", shape=(meta["n"],))
        if int(done.min()) == 0:
            raise RuntimeError(f"fuse cache {prefix} is incomplete: finish build_pooled_cache.py")
        self.path = pca_file(prefix, k)
        self.shape = tuple(json.load(open(self.path + ".json"))["shape"])
        self.index = {key: r for r, key in enumerate(meta["keys"])}
        self.mm = None

    def get(self, key: str) -> np.ndarray:
        if key not in self.index:
            raise RuntimeError(f"frame {key!r} is not in the fuse cache (WOR_FUSE_CACHE)")
        if self.mm is None:
            self.mm = np.memmap(self.path, dtype=np.float16, mode="r", shape=self.shape)
        return np.asarray(self.mm[self.index[key]])

    def __getstate__(self):
        st = dict(self.__dict__)
        st["mm"] = None
        return st


def window_file(prefix: str, k: int, kind: str = "route") -> str:
    """The window memmap next to a pooled cache: (N, C, M) fp16, rows aligned with the pooled cache's keys (2026-11-07).
    kind "route" = K windows on the frame's route; "fixed" = the same K windows on a straight-ahead line (the placement control);
    "side" (2026-11-08) = 2K windows one lane left (first K) and right (last K) of the route windows."""
    return f"{prefix}.{_WIN_TAG[kind]}{int(k)}.f16"


def window_count(k: int, kind: str) -> int:
    return 2 * int(k) if kind == "side" else int(k)


def pca_file(prefix: str, k: int) -> str:
    """The PCA-compressed full stride-32 map next to a pooled cache: (N, K, H, W) fp16; shape in <file>.json, basis in <file>.basis.pt."""
    return f"{prefix}.pca{int(k)}.f16"


def label_files(prefix: str) -> Tuple[str, str]:
    """Per-cell box labels next to a pooled cache: (N, 2, 4, 4) fp16 (occupancy, nearest distance / 40 m) and (N,) uint8 validity."""
    return f"{prefix}.lab.f16", f"{prefix}.labvalid.u8"


LABEL_CLASSES = ("car", "static_car", "walker", "static")
LABEL_MAX_DIST = 40.0


def cell_labels_from_boxes(boxes, offset_inv=None, mirror: bool = False) -> np.ndarray:
    """(2, 4, 4) labels for one frame from PDM-Lite boxes (ego frame, x forward / y right, the route's frame): for each 4x4 image cell,
    1 if any car / static car / walker / static object within 40 m projects into it, and its nearest distance / 40 m (1.0 when empty).
    offset_inv re-expresses the boxes in a recovery camera's frame (same transform as its route); mirror flips them left-right."""
    from src.config.camera import PDM_LITE_CAMERA
    from src.config.route_overlay import project_ego_points
    W, H = float(PDM_LITE_CAMERA["width"]), float(PDM_LITE_CAMERA["height"])
    lab = np.zeros((2, 4, 4), dtype=np.float32)
    lab[1] = 1.0
    for b in boxes:
        if b.get("class") not in LABEL_CLASSES:
            continue
        x, y = float(b["position"][0]), float(b["position"][1])
        if offset_inv is not None:
            p = offset_inv @ np.array([x, y, 0.0, 1.0])
            x, y = float(p[0]), float(p[1])
        if mirror:
            y = -y
        d = math.hypot(x, y)
        if d > LABEL_MAX_DIST:
            continue
        uv = project_ego_points([[x, y]])
        if uv.shape[0] == 0:
            continue
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if not (0.0 <= u < W and 0.0 <= v < H):
            continue
        r, cc = min(int(v / H * 4), 3), min(int(u / W * 4), 3)
        lab[0, r, cc] = 1.0
        lab[1, r, cc] = min(lab[1, r, cc], d / LABEL_MAX_DIST)
    return lab


def route_window_points(route_xy, k: int) -> np.ndarray:
    """K points evenly spaced by arc length along the ego-frame route polyline (the sample's route points, x forward / y right).
    A degenerate route (one point, or all points equal) gives K copies of its first point."""
    pts = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((k, 2))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-6:
        return np.repeat(pts[:1], k, axis=0)
    t = np.linspace(0.0, cum[-1], k)
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], axis=1)


def route_window_fracs(route_xy, k: int, crop_bottom_frac: float = 0.0, points=None) -> np.ndarray:
    """(K, 2) image fractions (v / height, u / width) of the K route points on the source camera image (the overlay's own projection,
    src/config/route_overlay.py). A point behind or too close to the camera falls back to the bottom centre of the frame.
    2026-11-08: `points` (M, 2) ego points replace the route points (side windows); with crop_bottom_frac > 0 the fractions are of the
    cropped frame (TF++'s 1024x384 geometry: crop 0.25), and a point below the crop line is clamped to the bottom row."""
    from src.config.camera import PDM_LITE_CAMERA
    from src.config.route_overlay import project_ego_points
    W, H = float(PDM_LITE_CAMERA["width"]), float(PDM_LITE_CAMERA["height"])
    Hc = H * (1.0 - float(crop_bottom_frac))
    pts = route_window_points(route_xy, k) if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty((len(pts), 2), dtype=np.float32)
    for i, p in enumerate(pts):
        uv = project_ego_points([p])
        u, v = (float(uv[0, 0]), float(uv[0, 1])) if uv.shape[0] else (W / 2.0, H - 1.0)
        out[i] = (min(max(v, 0.0), Hc - 1.0) / Hc, min(max(u, 0.0), W - 1.0) / W)
    return out


SIDE_WINDOW_OFFSET = 3.5   # metres: one lane width to each side of the route


def side_window_points(route_xy, k: int, offset: float = SIDE_WINDOW_OFFSET) -> np.ndarray:
    """(2K, 2) ego points one lane to the left (first K) and right (last K) of the K route-window points, along the route's local normal
    (2026-11-08, side-window arm). x forward / y right: the right normal of a tangent (tx, ty) is (-ty, tx)."""
    p = route_window_points(route_xy, k)
    t = np.gradient(p, axis=0) if k > 1 else np.array([[1.0, 0.0]])
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(n > 1e-6, t / np.maximum(n, 1e-6), np.array([[1.0, 0.0]]))
    right = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return np.concatenate([p - offset * right, p + offset * right], axis=0)


WINDOW_LABEL_RADIUS = 1.5   # metres from the window's ground point to an obstacle's footprint = "occupied"
WINDOW_LABEL_MAX = 10.0


def window_labels_from_boxes(points, boxes, offset_inv=None) -> np.ndarray:
    """(2, M) path-relative targets for M ego ground points (2026-11-08, idea 2): occupancy (an obstacle's footprint within 1.5 m of the
    point) and the footprint distance / 10 m (1.0 when nothing is within 10 m). Obstacles: car / static car / walker / static boxes
    (PDM-Lite extents are half-sizes, yaw in radians). offset_inv re-expresses the boxes in a recovery camera's frame (as its route)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lab = np.zeros((2, len(pts)), dtype=np.float32)
    best = np.full(len(pts), WINDOW_LABEL_MAX)
    for b in boxes:
        if b.get("class") not in LABEL_CLASSES:
            continue
        x, y = float(b["position"][0]), float(b["position"][1])
        yaw = float(b.get("yaw", 0.0) or 0.0)
        if offset_inv is not None:
            q = offset_inv @ np.array([x, y, 0.0, 1.0])
            x, y = float(q[0]), float(q[1])
            yaw += math.atan2(offset_inv[1, 0], offset_inv[0, 0])
        ex, ey = (float(b["extent"][0]), float(b["extent"][1])) if b.get("extent") else (0.5, 0.5)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = pts[:, 0] - x, pts[:, 1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy           # the point in the box's own frame
        d = np.hypot(np.maximum(np.abs(lx) - ex, 0.0), np.maximum(np.abs(ly) - ey, 0.0))
        best = np.minimum(best, d)
    lab[0] = (best < WINDOW_LABEL_RADIUS).astype(np.float32)
    lab[1] = np.minimum(best, WINDOW_LABEL_MAX) / WINDOW_LABEL_MAX
    return lab


def window_coord_channels(points) -> np.ndarray:
    """(2, M) the window ground points as two input channels, x / 20 m and y / 10 m (2026-11-08, idea 5)."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return np.stack([p[:, 0] / 20.0, p[:, 1] / 10.0], axis=0)


_WIN_TAG = {"route": "win", "fixed": "fwin", "side": "swin"}


def window_side_files(prefix: str, k: int, kind: str = "route") -> Dict[str, str]:
    """Side files of one window kind (2026-11-08): labels (N, 2, M) fp16, ground-point coordinates (N, 2, M) fp16 and TF++ decoder
    statistics (N, DSTAT, M) fp16, M = K (route, fixed) or 2K (side)."""
    base = f"{prefix}.{_WIN_TAG[kind]}{int(k)}"
    return {"lab": base + ".lab.f16", "xy": base + ".xy.f16", "dstat": base + ".dstat.f16"}


DSTAT_CHANNELS = 10   # mean depth, min depth, depth / flat-ground depth, 7 TF++ semantic class fractions


def grid_dstat_file(prefix: str) -> str:
    """TF++ decoder statistics of the token grid (N, DSTAT, gh, gw) fp16 (2026-11-08, idea 10)."""
    return f"{prefix}.dstat.f16"


def window_label_valid_file(prefix: str) -> str:
    return f"{prefix}.wlabvalid.u8"


def prefixes_from_env(value: Optional[str]) -> List[str]:
    return [x for x in (value or "").split(",") if x]


def prev_frame_rgb_path(rgb_path: str) -> Optional[str]:
    """The image path of the frame before this one in the same route folder (same zero-padded number minus one), or None when the
    frame is the first of its route. 2026-11-07, for the previous-frame arm (policy prev_frame / WOR_PREV_FRAME=1). The caller
    checks that the earlier frame is in the cache."""
    d, base = os.path.split(rgb_path)
    stem, ext = os.path.splitext(base)
    if not stem.isdigit() or int(stem) == 0:
        return None
    return os.path.join(d, f"{int(stem) - 1:0{len(stem)}d}{ext}")
