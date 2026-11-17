"""Box-derived targets aligned with a pooled cache's rows, CPU only (2026-11-08, wave 2 of the spatial tests).

  --full_grid HxW   per-cell obstacle labels of the full stride-32 map (idea 12): <prefix>.lab<H>x<W>.f16 (N, 2, H, W) = occupancy (any car /
                    static car / walker / static object within 40 m whose projected footprint covers the cell) and nearest distance / 40 m;
                    validity <prefix>.lab<H>x<W>.valid.u8
  --box_tokens M    the M nearest objects as privileged teacher inputs (idea 8): <prefix>.boxtok<M>.f16 (N, M, BOX_FEATS), zero rows = padding

Boxes are the logged PDM-Lite boxes (labels / teacher inputs only, never a policy input at test time). A recovery-camera frame
(rgb_augmented/) gets its boxes in that camera's frame (same offset as its route). The image geometry (crop) must be the cache's.

    python scripts/training/build_box_targets.py --prefix <pooled cache prefix> --data_dir <dataset> --full_grid 9x24 --box_tokens 20 --workers 12
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

BOX_CLASSES = ("car", "static_car", "walker", "static", "traffic_light", "stop_sign")
BOX_FEATS = 16   # x/20, y/10, cos yaw, sin yaw, ex/3, ey/3, speed/10, light red, light yellow (0-8), 6 class one-hot (9-14), valid (15)
VALID_SLOT = 15
OBSTACLES = ("car", "static_car", "walker", "static")
MAX_DIST = 40.0


def box_token(b, x, y, yaw):
    cls = b.get("class")
    f = np.zeros(BOX_FEATS, dtype=np.float32)
    ex, ey = (float(b["extent"][0]), float(b["extent"][1])) if b.get("extent") else (0.5, 0.5)
    state = str(b.get("state", "")).lower()
    f[:8] = [x / 20.0, y / 10.0, math.cos(yaw), math.sin(yaw), ex / 3.0, ey / 3.0, float(b.get("speed", 0.0) or 0.0) / 10.0,
             1.0 if ("red" in state or state == "0") else 0.0]
    f[8] = 1.0 if "yellow" in state else 0.0
    f[9 + BOX_CLASSES.index(cls)] = 1.0
    f[VALID_SLOT] = 1.0
    return f


def in_image(x, y, yaw, ex, ey, W, Hc):
    """True when the object's ground footprint (4 corners + centre) projects inside the cropped camera image (no occlusion test)."""
    from src.config.route_overlay import project_ego_points
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [[x + c * dx - s * dy, y + s * dx + c * dy] for dx, dy in ((ex, ey), (ex, -ey), (-ex, ey), (-ex, -ey), (0.0, 0.0))]
    uv = project_ego_points(pts)
    if uv.shape[0] == 0:
        return False
    return float(uv[:, 0].max()) >= 0.0 and float(uv[:, 0].min()) <= W - 1.0 and float(uv[:, 1].min()) <= Hc - 1.0


def frame_targets(args):
    rgb_path, gh, gw, crop, M, variant = args
    from scripts.training.build_pooled_cache import frame_boxes
    from src.config.camera import PDM_LITE_CAMERA
    from src.config.route_overlay import project_ego_points
    boxes, offset_inv = frame_boxes(rgb_path)
    lab = np.zeros((2, gh, gw), dtype=np.float32)
    lab[1] = 1.0
    tok = np.zeros((M, BOX_FEATS), dtype=np.float32)
    if boxes is None:
        return lab, 0, tok
    W, H = float(PDM_LITE_CAMERA["width"]), float(PDM_LITE_CAMERA["height"])
    Hc = H * (1.0 - crop)
    rot = math.atan2(offset_inv[1, 0], offset_inv[0, 0]) if offset_inv is not None else 0.0
    objs = []
    for b in boxes:
        cls = b.get("class")
        if cls not in BOX_CLASSES:
            continue
        x, y = float(b["position"][0]), float(b["position"][1])
        if offset_inv is not None:
            q = offset_inv @ np.array([x, y, 0.0, 1.0])
            x, y = float(q[0]), float(q[1])
        yaw = float(b.get("yaw", 0.0) or 0.0) + rot
        d = math.hypot(x, y)
        if d > 50.0:
            continue
        if variant:   # _vis / _visns: the teacher only gets what the front camera could show
            bex, bey = (float(b["extent"][0]), float(b["extent"][1])) if b.get("extent") else (0.5, 0.5)
            if not in_image(x, y, yaw, bex, bey, W, Hc):
                continue
        objs.append((d, b, x, y, yaw))
        if gh and cls in OBSTACLES and d <= MAX_DIST:
            ex, ey = (float(b["extent"][0]), float(b["extent"][1])) if b.get("extent") else (0.5, 0.5)
            c, s = math.cos(yaw), math.sin(yaw)
            pts = [[x + c * dx - s * dy, y + s * dx + c * dy] for dx, dy in ((ex, ey), (ex, -ey), (-ex, ey), (-ex, -ey), (0.0, 0.0))]
            uv = project_ego_points(pts)
            if uv.shape[0] == 0:
                continue
            u0, u1 = max(float(uv[:, 0].min()), 0.0), min(float(uv[:, 0].max()), W - 1.0)
            v0, v1 = max(float(uv[:, 1].min()), 0.0), min(float(uv[:, 1].max()), Hc - 1.0)
            if u0 > u1 or v0 > v1:
                continue
            c0, c1 = int(u0 / W * gw), int(u1 / W * gw)
            r0, r1 = int(v0 / Hc * gh), int(v1 / Hc * gh)
            lab[0, r0:r1 + 1, c0:c1 + 1] = 1.0
            lab[1, r0:r1 + 1, c0:c1 + 1] = np.minimum(lab[1, r0:r1 + 1, c0:c1 + 1], d / MAX_DIST)
    objs.sort(key=lambda o: o[0])
    for i, (_, b, x, y, yaw) in enumerate(objs[:M]):
        tok[i] = box_token(b, x, y, yaw)
        if variant == "_visns":
            tok[i, 6] = 0.0   # no speed: a single camera frame does not show it
    return lab, 1, tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--full_grid", default="")
    ap.add_argument("--box_tokens", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--box_variant", default="", choices=["", "_vis", "_visns"],
                    help="teacher ceiling: '' all objects within 50 m; _vis only objects inside the camera image; _visns also speed zeroed")
    a = ap.parse_args()
    meta = json.load(open(a.prefix + ".json"))
    N, keys = meta["n"], meta["keys"]
    crop = float(meta.get("crop_bottom_frac", 0.0) or 0.0)
    gh, gw = (int(v) for v in a.full_grid.lower().split("x")) if a.full_grid else (0, 0)
    M = max(int(a.box_tokens), 1)
    lab_mm = val_mm = tok_mm = None
    if gh:
        lab_mm = np.memmap(f"{a.prefix}.lab{gh}x{gw}.f16", dtype=np.float16, mode="w+", shape=(N, 2, gh, gw))
        val_mm = np.memmap(f"{a.prefix}.lab{gh}x{gw}.valid.u8", dtype=np.uint8, mode="w+", shape=(N,))
    if a.box_tokens:
        tok_mm = np.memmap(f"{a.prefix}.boxtok{M}{a.box_variant}.f16", dtype=np.float16, mode="w+", shape=(N, M, BOX_FEATS))
    jobs = [(os.path.join(a.data_dir, k), gh, gw, crop, M, a.box_variant) for k in keys]
    with Pool(a.workers) as pool:
        for i, (lab, valid, tok) in enumerate(pool.imap(frame_targets, jobs, chunksize=256)):
            if lab_mm is not None:
                lab_mm[i], val_mm[i] = lab, valid
            if tok_mm is not None:
                tok_mm[i] = tok
            if i % 50000 == 0:
                print(f"--> {i}/{N}", flush=True)
    for m in (lab_mm, val_mm, tok_mm):
        if m is not None:
            m.flush()
    if lab_mm is not None:
        print(f"--> full-grid labels {gh}x{gw}: valid {int(np.asarray(val_mm).sum())} of {N}; occupied cells {float(np.asarray(lab_mm[:20000, 0], dtype=np.float32).mean()):.3f}")
    if tok_mm is not None:
        print(f"--> box tokens: {M} per frame; mean valid objects {float(np.asarray(tok_mm[:20000, :, VALID_SLOT], dtype=np.float32).sum(1).mean()):.1f}")
    print("BOX_TARGETS_DONE")


if __name__ == "__main__":
    main()
