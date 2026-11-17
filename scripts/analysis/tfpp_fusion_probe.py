"""Does our extracted TF++ camera encoder lose information because it skips TF++'s fusion blocks? (2026-11-08)

carla_encoder.py runs TF++'s regnety_032 image branch stage after stage. In TF++ itself (carla_garage team_code/transfuser.py,
TransfuserBackbone.forward) every one of the 4 stages is followed by fuse_features(): a GPT block over image + LiDAR tokens whose
output is ADDED to the image features before the next stage. So stages 2-4 of our encoder see inputs they never saw in training.

This probe loads the released TF++ model and decodes TF++'s own perspective depth and semantic heads from the image features of:
  V2  native   : full backbone, real LiDAR BEV                        (TF++ as trained; the reference)
  V1  zero     : full backbone, empty LiDAR BEV                       (camera only, fusion kept)
  V3  pseudo   : full backbone, LiDAR BEV splatted from V1's own depth (camera only, pseudo-LiDAR)
  V0  ours     : image stages only, no fusion                          (what carla_encoder.py / our caches compute)
and scores them against the logged depth / semantics of PDM-Lite frames (same 1024x384 crop TF++ trains on). Also reports the
cosine similarity of each variant's stride-32 image features to V2's.

    python scripts/analysis/tfpp_fusion_probe.py --tfpp_dir <pretrained_models/all_towns> --route_dir <PDM-Lite route folder> [--max_frames 73]
"""
import argparse
import glob
import gzip
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TC = os.path.join(_REPO, "Carla-utils", "carla_garage", "team_code")


def _import_tfpp():
    if "carla" not in sys.modules:  # config.py / transfuser_utils use carla only for colours and red-light helpers: a permissive stub
        class _Any:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, name):
                return _Any()

            def __call__(self, *a, **k):
                return _Any()
        def _mod_getattr(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Any
        # nav_planner.py (imported by model.py, used only for driving control) pulls in CARLA's PythonAPI agents package
        for mod in ("carla", "agents", "agents.navigation", "agents.navigation.global_route_planner", "agents.navigation.local_planner",
                    "agents.tools", "agents.tools.misc"):
            stub = types.ModuleType(mod)
            stub.__getattr__ = _mod_getattr
            stub.__path__ = []
            sys.modules[mod] = stub
    if _TC not in sys.path:
        sys.path.insert(0, _TC)
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy
    jsonpickle_numpy.register_handlers()
    from config import GlobalConfig  # noqa
    from model import LidarCenterNet  # noqa
    return jsonpickle, GlobalConfig, LidarCenterNet


def load_tfpp(tfpp_dir: str, device: str):
    jsonpickle, GlobalConfig, LidarCenterNet = _import_tfpp()
    cfg = GlobalConfig()
    with open(os.path.join(tfpp_dir, "config.json"), "rt", encoding="utf-8") as f:
        cfg.__dict__.update(jsonpickle.decode(f.read()).__dict__)
    import timm
    _create = timm.create_model
    timm.create_model = lambda *a, **k: _create(*a, **{**k, "pretrained": False})   # every tensor comes from the TF++ checkpoint below
    try:
        net = LidarCenterNet(cfg)
    finally:
        timm.create_model = _create
    ck = sorted(glob.glob(os.path.join(tfpp_dir, "model*.pth")))[0]
    sd = torch.load(ck, map_location="cpu")
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"--> {ck}: missing {len(missing)}, unexpected {len(unexpected)}; backbone={cfg.backbone} lidar={cfg.lidar_architecture} "
          f"use_depth={cfg.use_depth} use_semantic={cfg.use_semantic} crop={cfg.crop_image}/{cfg.cropped_height}", flush=True)
    return net.to(device).eval(), cfg


def lidar_hist(cfg, xyz: np.ndarray) -> np.ndarray:
    """TF++'s data.py lidar_to_histogram_features (use_ground_plane False -> 1 channel)."""
    xyz = xyz[xyz[:, 2] < cfg.max_height_lidar]
    above = xyz[xyz[:, 2] > cfg.lidar_split_height]
    xb = np.linspace(cfg.min_x, cfg.max_x, (cfg.max_x - cfg.min_x) * int(cfg.pixels_per_meter) + 1)
    yb = np.linspace(cfg.min_y, cfg.max_y, (cfg.max_y - cfg.min_y) * int(cfg.pixels_per_meter) + 1)
    h = np.histogramdd(above[:, :2], bins=(xb, yb))[0]
    h = np.minimum(h, cfg.hist_max_per_pixel) / cfg.hist_max_per_pixel
    ch = [h.T]
    if cfg.use_ground_plane:
        below = xyz[xyz[:, 2] <= cfg.lidar_split_height]
        hb = np.minimum(np.histogramdd(below[:, :2], bins=(xb, yb))[0], cfg.hist_max_per_pixel) / cfg.hist_max_per_pixel
        ch = [hb.T, h.T]
    return np.stack(ch, 0).astype(np.float32)


def pseudo_lidar_points(cfg, depth01: np.ndarray, lidar_z_offset: float, sx: float = 1.0, sy: float = 1.0, side=None) -> np.ndarray:
    """Back-projects a (H, W) normalised depth map (0..1 = 0..50 m, planar camera depth; camera 1024x512, FOV 110 at x -1.5 m, z 2.0 m)
    into points in the logged LiDAR's frame: x forward, y right, z relative to the LiDAR mount (lidar_z_offset = the camera-to-LiDAR
    height difference measured from the logged clouds). Pixel (u, v) of the map is camera pixel (u * sx + side, v * sy): TF++'s
    1024x384 crop is sx = sy = 1 with a centred side crop; our 288x768 resize of the full frame is sx = 1024/768, sy = 512/288, side 0."""
    H, W = depth01.shape
    f = cfg.camera_width / (2.0 * np.tan(np.deg2rad(cfg.camera_fov) / 2.0))
    cx, cy = cfg.camera_width / 2.0, cfg.camera_height / 2.0
    if side is None:
        side = (cfg.camera_width - W * sx) // 2
    vv, uu = np.mgrid[0:H, 0:W]
    uu = uu * sx + side
    vv = vv * sy
    d = depth01 * 50.0
    ok = (d > 0.5) & (d < 49.0)
    x = d[ok]
    y = (uu[ok] - cx) / f * x
    z = -(vv[ok] - cy) / f * x
    return np.stack([x + cfg.camera_pos[0] - cfg.lidar_pos[0], y, z + lidar_z_offset], 1)


@torch.no_grad()
def image_stages_only(net, rgb):
    """V0: the image branch's stages back to back, as carla_encoder.py runs them (no fusion residuals)."""
    import transfuser_utils as t_u
    x = t_u.normalize_imagenet(rgb)
    for _, m in net.backbone.image_encoder.items():
        x = m(x)
    return x


@torch.no_grad()
def decode(net, feat):
    depth = torch.sigmoid(net.depth_decoder(feat)).squeeze(1)
    sem = net.semantic_decoder(feat).argmax(1)
    return depth, sem


GT_OBSTACLES = ("car", "static_car", "walker")


def frustum(x, y, max_d=30.0):
    """In front of the camera, inside its 110 degree field of view, within max_d metres."""
    return (x > 0.5) & (np.hypot(x, y) < max_d) & (np.abs(np.degrees(np.arctan2(y, x))) < 55.0)


@torch.no_grad()
def detect(net, cfg, bev_feat):
    """TF++'s CenterNet head on the BEV features -> (N, 2) box centres in the vehicle frame (x forward, y right), score > threshold."""
    import transfuser_utils as t_u
    pred = net.head(bev_feat)
    bb = net.head.get_bboxes(*pred)[0]
    bb = bb[bb[:, -1] > cfg.bb_confidence_threshold].detach().cpu().numpy()
    out = [t_u.bb_image_to_vehicle_system(b.copy(), cfg.pixels_per_meter, cfg.min_x, cfg.min_y)[:2] for b in bb]
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def box_match(pred_xy, gt_xy, radius=2.0):
    """(matched GT, GT count, matched predictions, prediction count) inside the camera frustum, centre distance < radius."""
    pm = frustum(pred_xy[:, 0], pred_xy[:, 1]) if len(pred_xy) else np.zeros(0, bool)
    gm = frustum(gt_xy[:, 0], gt_xy[:, 1]) if len(gt_xy) else np.zeros(0, bool)
    p, g = pred_xy[pm], gt_xy[gm]
    if len(p) == 0 or len(g) == 0:
        return 0, len(g), 0, len(p)
    d = np.linalg.norm(p[:, None] - g[None], axis=2)
    return int((d.min(0) < radius).sum()), len(g), int((d.min(1) < radius).sum()), len(p)


def iou(pred, gt, c):
    p, g = pred == c, gt == c
    u = (p | g).sum()
    return float((p & g).sum() / u) if u > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfpp_dir", required=True)
    ap.add_argument("--route_dir", required=True, nargs="+")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--input_mode", default="tfpp", choices=["tfpp", "ours"],
                    help="tfpp: TF++'s own 1024x384 crop; ours: the full 1024x512 frame resized to 288x768 as our caches see it (no overlay)")
    a = ap.parse_args()

    import cv2
    import laspy
    net, cfg = load_tfpp(a.tfpp_dir, a.device)
    import transfuser_utils as t_u
    conv = np.uint8(cfg.converter)
    frames = []
    for rd in a.route_dir:
        for p in sorted(glob.glob(os.path.join(rd, "rgb", "*.jpg"))):
            frames.append((rd, os.path.splitext(os.path.basename(p))[0]))
    frames = frames[:a.max_frames]

    # LiDAR frame check: z of the ground in the logged clouds (the 5th percentile of z), to place pseudo-LiDAR points in the same frame
    zs = []
    for rd, s in frames[:10]:
        zs.append(np.percentile(laspy.read(os.path.join(rd, "lidar", f"{s}.laz")).xyz[:, 2], 5))
    ground_z = float(np.median(zs))
    lidar_z_offset = ground_z + cfg.camera_pos[2]   # a camera-frame point at height -2.0 m (the ground) maps to ground_z
    print(f"--> logged LiDAR ground at z={ground_z:.2f} (5th percentile); pseudo-LiDAR z offset {lidar_z_offset:.2f}", flush=True)

    names = ["V2_native_lidar", "V1_zero_lidar", "V3_pseudo_lidar", "V0_no_fusion(ours)"]
    acc = {n: {"depth_mae_m": [], "depth_mae_near_m": [], "iou_vehicle": [], "iou_road": [], "iou_pedestrian": [], "iou_roadline": [],
               "feat_cos_to_V2": []} for n in names}
    det = {n: [0, 0, 0, 0] for n in names[:3]}   # V0 has no BEV branch
    sx, sy = (1.0, 1.0) if a.input_mode == "tfpp" else (cfg.camera_width / 768.0, cfg.camera_height / 288.0)

    def prep(arr, interp):
        if a.input_mode == "tfpp":
            return t_u.crop_array(cfg, arr)
        return cv2.resize(arr, (768, 288), interpolation=interp)
    for i, (rd, s) in enumerate(frames):
        img = cv2.cvtColor(cv2.imread(os.path.join(rd, "rgb", f"{s}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = prep(img, cv2.INTER_AREA)
        gt_d = prep(cv2.imread(os.path.join(rd, "depth", f"{s}.png"), cv2.IMREAD_UNCHANGED), cv2.INTER_NEAREST).astype(np.float32) / 255.0
        gt_s = conv[prep(cv2.imread(os.path.join(rd, "semantics", f"{s}.png"), cv2.IMREAD_UNCHANGED), cv2.INTER_NEAREST)]
        with gzip.open(os.path.join(rd, "boxes", f"{s}.json.gz"), "rt") as fb:
            gt_xy = np.array([b["position"][:2] for b in json.load(fb) if b.get("class") in GT_OBSTACLES], dtype=np.float64).reshape(-1, 2)
        xyz = laspy.read(os.path.join(rd, "lidar", f"{s}.laz")).xyz
        rgb = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().unsqueeze(0).to(a.device)
        lid = torch.from_numpy(lidar_hist(cfg, xyz)).unsqueeze(0).to(a.device)
        with torch.no_grad():
            b2, _, f2 = net.backbone(rgb, lid)
            b1, _, f1 = net.backbone(rgb, torch.zeros_like(lid))
            d1, _ = decode(net, f1)
            pl = pseudo_lidar_points(cfg, d1[0].float().cpu().numpy(), lidar_z_offset, sx, sy, None if a.input_mode == "tfpp" else 0.0)
            b3, _, f3 = net.backbone(rgb, torch.from_numpy(lidar_hist(cfg, pl)).unsqueeze(0).to(a.device))
            f0 = image_stages_only(net, rgb)
            for n, bf in zip(names[:3], [b2, b1, b3]):
                det[n] = [x + y for x, y in zip(det[n], box_match(detect(net, cfg, bf), gt_xy))]
        for n, f in zip(names, [f2, f1, f3, f0]):
            d, sm = decode(net, f)
            d, sm = d[0].float().cpu().numpy(), sm[0].cpu().numpy()
            if d.shape != gt_d.shape:
                d = cv2.resize(d, (gt_d.shape[1], gt_d.shape[0]), interpolation=cv2.INTER_LINEAR)
                sm = cv2.resize(sm.astype(np.uint8), (gt_s.shape[1], gt_s.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = gt_d < 0.999
            near = valid & (gt_d < 0.6)   # < 30 m
            acc[n]["depth_mae_m"].append(float(np.abs(d - gt_d)[valid].mean() * 50.0))
            acc[n]["depth_mae_near_m"].append(float(np.abs(d - gt_d)[near].mean() * 50.0))
            for key, c in (("iou_vehicle", 1), ("iou_road", 2), ("iou_pedestrian", 4), ("iou_roadline", 5)):
                acc[n][key].append(iou(sm, gt_s, c))
            acc[n]["feat_cos_to_V2"].append(float(F.cosine_similarity(f.flatten(1).float(), f2.flatten(1).float()).item()))
        if i % 10 == 0:
            print(f"    {i + 1}/{len(frames)} frames", flush=True)

    summary = {n: {k: float(np.nanmean(v)) for k, v in m.items()} for n, m in acc.items()}
    for n, (mg, ng, mp, npred) in det.items():
        summary[n]["box_recall_30m"] = mg / max(ng, 1)
        summary[n]["box_precision_30m"] = mp / max(npred, 1)
        summary[n]["gt_boxes_30m"] = ng
    print(f"\ninput {a.input_mode}; {len(frames)} frames from {len(a.route_dir)} route(s); depth error in metres (all valid pixels / pixels < 30 m), IoU of TF++'s 7-class semantics")
    print(f"{'variant':22s} {'depth_mae':>9s} {'near<30m':>9s} {'IoU veh':>8s} {'IoU road':>8s} {'IoU ped':>8s} {'IoU line':>8s} {'cos(V2)':>8s} {'box rec':>8s} {'box prec':>8s}")
    for n, m in summary.items():
        print(f"{n:22s} {m['depth_mae_m']:9.2f} {m['depth_mae_near_m']:9.2f} {m['iou_vehicle']:8.3f} {m['iou_road']:8.3f} "
              f"{m['iou_pedestrian']:8.3f} {m['iou_roadline']:8.3f} {m['feat_cos_to_V2']:8.3f} "
              f"{m.get('box_recall_30m', float('nan')):8.3f} {m.get('box_precision_30m', float('nan')):8.3f}")
    print(f"(boxes: car / static car / walker centres within 30 m in the camera's view, matched within 2 m; {det[names[0]][1]} GT boxes)")
    if a.out_json:
        json.dump({"frames": len(frames), "routes": a.route_dir, "input_mode": a.input_mode, "ground_z": ground_z, "summary": summary}, open(a.out_json, "w"), indent=1)
    try:  # MLflow (project rule)
        import mlflow
        mlflow.set_experiment("tfpp_fusion_probe")
        with mlflow.start_run(run_name="fusion_probe"):
            mlflow.log_params({"frames": len(frames), "routes": len(a.route_dir), "input_mode": a.input_mode})
            for n, m in summary.items():
                mlflow.log_metrics({f"{n.split('_')[0]}_{k}": v for k, v in m.items()})
    except Exception as exc:
        print(f"[mlflow skipped: {exc}]")


if __name__ == "__main__":
    main()
