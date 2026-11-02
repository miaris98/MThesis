"""TODO A71 / A72: the per-route label files written by scripts/training/build_q_labels.py (split out of counterfactual_q.py, 2026-11-02).

<route>/<tag>.npz holds frame ids, Q (n_frames, n_speed, n_lat, NaN where unlabelled), the QConfig as JSON, the frame stride, and (A72) each
frame's expert action and logged step reward. A file is used only if its shape and every QConfig field match the defaults.
"""
from __future__ import annotations

import dataclasses
import json
from typing import List, Optional, Tuple

import numpy as np

from src.training.counterfactual_q import QConfig, solve_q


def load_q_transitions(route_dir: str, tag: Optional[str]) -> dict:
    """TODO A72: {frame id: (flattened expert action, reward of its logged step)} from the same label file; frames without one are absent."""
    z = _open_labels(route_dir, tag)
    if z is None or "a_expert" not in z.files:
        return {}
    a, r = z["a_expert"], z["r_expert"]
    return {str(f): (int(a[i]), float(r[i])) for i, f in enumerate(z["frame"]) if a[i] >= 0 and np.isfinite(r[i])}


def load_q_labels(route_dir: str, tag: Optional[str]) -> dict:
    """{frame id: flattened Q row} from `<route_dir>/<tag>.npz` (written by scripts/training/build_q_labels.py); rows with a NaN are dropped."""
    z = _open_labels(route_dir, tag)
    if z is None:
        return {}
    q = z["q"].reshape(len(z["frame"]), -1).astype(np.float32)
    return {str(f): q[i] for i, f in enumerate(z["frame"]) if np.isfinite(q[i]).all()}


def _open_labels(route_dir: str, tag: Optional[str]):
    """The validated label npz (None without one): shape, stored config and every QConfig field must match the defaults."""
    import os
    path = os.path.join(route_dir, f"{tag}.npz") if tag else ""
    if not tag or not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    want = QConfig()
    if z["q"].shape[1:] != (len(want.speed_targets), len(want.lat_targets)):
        raise ValueError(f"{path}: Q labels of shape {z['q'].shape[1:]}, the policy's action grid is "
                         f"{(len(want.speed_targets), len(want.lat_targets))}: rebuild with build_q_labels.py --overwrite")
    if "config" not in z.files:
        raise ValueError(f"{path}: no config stored with the labels: rebuild with build_q_labels.py --overwrite")
    cfg = json.loads(str(z["config"]))
    if [round(v, 4) for v in cfg.get("speed_targets", want.speed_targets)] != [round(v, 4) for v in want.speed_targets] or \
            [round(v, 4) for v in cfg.get("lat_targets", want.lat_targets)] != [round(v, 4) for v in want.lat_targets]:
        raise ValueError(f"{path}: built for another action grid ({cfg.get('speed_targets')}, {cfg.get('lat_targets')})")
    # every other field changes the label values, not just their shape (a stale tag such as qcf_v2, hold 8, would otherwise mix in silently)
    rnd = lambda v: [round(x, 6) for x in v] if isinstance(v, (list, tuple)) else round(v, 6) if isinstance(v, float) else v
    bad = [k for k, v in dataclasses.asdict(want).items() if k not in cfg or rnd(cfg[k]) != rnd(v)]
    if bad:
        raise ValueError(f"{path}: built with another QConfig ({', '.join(f'{k}={cfg.get(k)} vs {getattr(want, k)}' for k in bad)}): "
                         f"train only on labels built with the QConfig defaults (--set / --hold ablation tags are for label statistics)")
    return z


def check_q_label_coverage(samples: list, tag: Optional[str]) -> Optional[float]:
    """Share of the real-camera samples that carry a Q label; raises below WOR_Q_MIN_COVERAGE (default 0.5) when labels were asked for. A
    wrong tag or data path otherwise gives a loss of exactly 0, a head that stays at its zero init, and a run that looks converged."""
    import os
    if not tag:
        return None
    real = [s for s in samples if not s.get("is_recovery_augmented")]
    cov = sum(s.get("qsa") is not None for s in real) / max(len(real), 1)
    print(f"--> Q labels '{tag}': {cov:.1%} of {len(real)} real-camera samples labelled", flush=True)
    if cov < float(os.environ.get("WOR_Q_MIN_COVERAGE", "0.5")):
        raise RuntimeError(f"Q labels '{tag}' cover only {cov:.1%} of the samples (WOR_Q_MIN_COVERAGE): wrong --q_labels tag or data dir, "
                           f"or build_q_labels.py has not run on these routes")
    return cov


def _build_scene(frames, k, cfg):
    from src.training.q_scene import build_scene
    return build_scene(frames, k, cfg)


def label_route(frames: List[Tuple[dict, list]], cfg: QConfig, stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q labels for every `stride`-th frame of a route: (n_frames, n_speed, n_lat) float32 with NaN where a frame is skipped or has too few logged
    successors, the mask, and the collision flags of each action's best sequence."""
    q = np.full((len(frames), len(cfg.speed_targets), len(cfg.lat_targets)), np.nan, dtype=np.float32)
    hit = np.zeros(q.shape, dtype=bool)
    ok = np.zeros(len(frames), dtype=bool)
    for k in range(0, len(frames), max(1, stride)):
        scene = _build_scene(frames, k, cfg)
        if scene is not None:
            q[k], hit[k] = solve_q(scene, cfg, return_hits=True)
            ok[k] = True
    return q, ok, hit
