#!/usr/bin/env python3
"""Does the painted route overlay hurt the frozen TF++ backbone's view of vehicles? (TODO_ACTIVE A47)

The route polyline (alpha 0.55, 5 px) is drawn into the RGB before the frozen backbone reads it, over the lane where lead
vehicles and crossing pedestrians appear. TF++ was trained on clean images. This runs TF++'s pretrained depth / semantic
decoders (zero-LiDAR, as in A34 step 1) on the same PDM-Lite frames twice, clean and with the overlay our policy sees, and
scores both against the dataset's ground truth, over the whole frame and inside the route corridor (the overlay's pixels
dilated; the corridor mask is rendered through the same TF++ preprocessing as the frame, so the geometry matches).

Decision (A47): vehicle IoU in the corridor drops by more than 2 points -> the route should enter as its own token/channel
and the image stays clean (A17c); no drop -> close the question.

    python scripts/analysis/overlay_effect_on_tfpp_decoders.py --data /workspace/dataset/wor_trajectories \
        --tfpp /workspace/tfpp_pretrained/pretrained_models/all_towns --garage /workspace/carla_garage/team_code \
        --route_key route_original --route_points 4 --frames 500
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from tfpp_decoders_rgb_only import link_routes, stub_modules  # noqa: E402

VEHICLE, PEDESTRIAN = 1, 4  # carla_garage semantic classes (config.py `classes`)


class OverlayReader:
    """Replaces cv2.imread for the dataset's rgb files only: mode 'clean' passes the image through, 'overlay' draws the route,
    'mask' returns the thick route corridor as a white-on-black image so the dataset's own crop/resize carries it along."""

    def __init__(self, cv2, route_key, n_points, corridor_px):
        self.cv2, self.route_key, self.n, self.corridor_px = cv2, route_key, n_points, corridor_px
        self.mode = "clean"
        self.orig = cv2.imread
        self.missing = 0

    def route(self, path):
        mpath = path.replace(os.sep + "rgb" + os.sep, os.sep + "measurements" + os.sep).replace("/rgb/", "/measurements/")
        mpath = os.path.splitext(mpath)[0] + ".json.gz"
        try:
            with gzip.open(mpath, "rt") as f:
                meas = json.load(f)
        except Exception:
            self.missing += 1
            return None
        r = meas.get(self.route_key) or meas.get("route")
        if not r:
            return None
        arr = np.asarray(r, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        idx = np.linspace(0, arr.shape[0] - 1, self.n).round().astype(int)
        return arr[idx, :2].tolist()  # same subsampling as WorldOnRailsDataset._subsample_route

    def __call__(self, path, flags=None):
        img = self.orig(path) if flags is None else self.orig(path, flags)
        is_rgb = img is not None and flags in (None, self.cv2.IMREAD_COLOR) and (os.sep + "rgb" + os.sep in path or "/rgb/" in path)
        if not is_rgb or self.mode == "clean":
            return img
        route = self.route(path)
        from src.config.route_overlay import draw_route_overlay
        rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        if route is None or len(route) < 2:
            return img if self.mode == "overlay" else np.zeros_like(img)
        if self.mode == "overlay":
            return self.cv2.cvtColor(draw_route_overlay(rgb, route), self.cv2.COLOR_RGB2BGR)
        black = np.zeros_like(rgb)
        m = draw_route_overlay(black, route, color=(255, 255, 255), alpha=1.0, thickness=self.corridor_px)
        return self.cv2.cvtColor(m, self.cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tfpp", required=True)
    ap.add_argument("--garage", required=True)
    ap.add_argument("--routes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route_key", default="route_original")
    ap.add_argument("--route_points", type=int, default=4)
    ap.add_argument("--stratify_ped", type=int, default=0, help="if > 0, only frames with at least this many ground-truth pedestrian pixels")
    ap.add_argument("--corridor_px", type=int, default=60, help="thickness of the corridor mask in source pixels")
    ap.add_argument("--workdir", default="/workspace/overlay_check")
    ap.add_argument("--use_mlflow", type=int, default=1)
    a = ap.parse_args()

    stub_modules()
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    import cv2
    sys.path.insert(0, a.garage)
    import jsonpickle
    from config import GlobalConfig
    from data import CARLA_Data
    from model import LidarCenterNet

    reader = OverlayReader(cv2, a.route_key, a.route_points, a.corridor_px)
    cv2.imread = reader
    cfg = GlobalConfig()
    cfg.__dict__.update(jsonpickle.decode(open(os.path.join(a.tfpp, "config.json")).read()).__dict__)
    cfg.val_towns = []
    cfg.augment = 1
    cfg.augment_percentage = 0.0
    cfg.use_color_aug = 0
    cfg.num_repetitions = max(getattr(cfg, "num_repetitions", 1), 3)
    picked = link_routes(a.data, os.path.join(a.workdir, "routes"), a.routes, a.seed)
    ds = CARLA_Data(root=[os.path.join(a.workdir, "routes")], config=cfg)
    ds.image_augmenter_func = ds.lidar_augmenter_func = lambda image: image
    print(f"routes linked {len(picked)}, frames available {len(ds)}", flush=True)

    dev = torch.device("cuda")
    model = LidarCenterNet(cfg).to(dev).eval()
    sd = torch.load(os.path.join(a.tfpp, "model_0030_0.pth"), map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"state_dict: missing {len(missing)} unexpected {len(unexpected)}", flush=True)

    n_cls = len(cfg.semantic_weights)
    scopes = ("frame", "corridor")
    stats = {m: {s: {"dl1": [], "inter": np.zeros(n_cls), "union": np.zeros(n_cls), "n": 0, "correct": 0} for s in scopes}
             for m in ("clean", "overlay")}
    idx = random.Random(a.seed).sample(range(len(ds)), min(a.frames, len(ds)))
    if a.stratify_ped:
        # Keep frames whose ground-truth semantic map holds at least `stratify_ped` pedestrian pixels. Selection is on the ground truth only
        # (never on a prediction), so the result reads: given a pedestrian in view, what does the overlay cost inside the corridor.
        conv = np.uint8(cfg.converter)
        cand = []
        for i in range(len(ds)):
            p0 = ds.images[i]
            while isinstance(p0, np.ndarray) and p0.ndim > 0:
                p0 = p0.flat[0]
            p0 = p0.decode("utf-8") if isinstance(p0, (bytes, np.bytes_)) else str(p0)
            sp = os.path.realpath(p0).replace(os.sep + "rgb" + os.sep, os.sep + "semantics" + os.sep).replace("/rgb/", "/semantics/")
            sp = os.path.splitext(sp)[0] + ".png"
            sem = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
            if sem is not None and int((conv[sem] == PEDESTRIAN).sum()) >= a.stratify_ped:
                cand.append(i)
        random.Random(a.seed).shuffle(cand)
        print(f"stratified: {len(cand)} of {len(ds)} frames have >= {a.stratify_ped} pedestrian pixels", flush=True)
        idx = cand[:a.frames]
    corridor_frac = []
    perframe = {}         # (frame k, mode) -> {class: (intersection, union)} inside the corridor, for the bootstrap
    route_of_frame = []   # route directory of frame k, the resampling unit (frames of a route are near-duplicates)
    with torch.inference_mode():
        for k, i in enumerate(idx):
            reader.mode = "clean"
            d0 = ds[i]
            reader.mode = "overlay"
            d1 = ds[i]
            reader.mode = "mask"
            dm = ds[i]
            reader.mode = "clean"
            corridor = torch.from_numpy(np.ascontiguousarray(dm["rgb"])).float().to(dev).amax(0) > 127  # HxW
            corridor_frac.append(corridor.float().mean().item())
            lidar = torch.zeros_like(torch.from_numpy(np.ascontiguousarray(d0["lidar"])).float()[None].to(dev))
            gt_d = torch.from_numpy(d0["depth"]).float().to(dev)
            gt_s = torch.from_numpy(d0["semantic"]).long().to(dev)
            if corridor.shape != gt_s.shape:
                corridor = torch.nn.functional.interpolate(corridor[None, None].float(), size=gt_s.shape, mode="nearest")[0, 0] > 0.5
            for mode, d in (("clean", d0), ("overlay", d1)):
                rgb = torch.from_numpy(np.ascontiguousarray(d["rgb"])).float()[None].to(dev)
                _, _, img_grid = model.backbone(rgb, lidar)
                pd = torch.sigmoid(model.depth_decoder(img_grid)).squeeze(1)[0]
                ps = model.semantic_decoder(img_grid).argmax(1)[0]
                for scope, m in (("frame", torch.ones_like(corridor)), ("corridor", corridor)):
                    if int(m.sum()) == 0:
                        continue
                    s = stats[mode][scope]
                    s["dl1"].append((pd - gt_d).abs()[m].mean().item())
                    s["correct"] += int(((ps == gt_s) & m).sum())
                    s["n"] += int(m.sum())
                    for c in range(n_cls):
                        p, g = (ps == c) & m, (gt_s == c) & m
                        s["inter"][c] += int((p & g).sum())
                        s["union"][c] += int((p | g).sum())
                        if scope == "corridor" and c in (VEHICLE, PEDESTRIAN):
                            perframe.setdefault((k, mode), {})[c] = (int((p & g).sum()), int((p | g).sum()))
            p0 = ds.images[i]
            while isinstance(p0, np.ndarray) and p0.ndim > 0:
                p0 = p0.flat[0]
            p0 = p0.decode("utf-8") if isinstance(p0, (bytes, np.bytes_)) else str(p0)
            route_of_frame.append(p0.split("/")[-3])  # <workdir>/routes/<Town_Scenario_RouteN_Rep0>/rgb/NNNN.jpg
            if k % 50 == 0:
                print(f"{k}/{len(idx)}", flush=True)

    res = {}
    for mode, per in stats.items():
        res[mode] = {}
        for scope, s in per.items():
            iou = s["inter"] / np.maximum(s["union"], 1)
            res[mode][scope] = {"depth_l1": float(np.mean(s["dl1"])), "sem_acc": s["correct"] / max(s["n"], 1),
                                "vehicle_iou": float(iou[VEHICLE]), "pedestrian_iou": float(iou[PEDESTRIAN]),
                                "vehicle_pixels": int(s["union"][VEHICLE]), "pedestrian_pixels": int(s["union"][PEDESTRIAN]),
                                "sem_iou": [round(float(x), 3) for x in iou]}
    delta = {sc: {k: res["overlay"][sc][k] - res["clean"][sc][k] for k in ("depth_l1", "sem_acc", "vehicle_iou", "pedestrian_iou")} for sc in scopes}
    # Cluster bootstrap over routes: resample whole routes with replacement, pool the corridor counts, difference of IoU (overlay - clean).
    boot = {}
    routes_u = sorted(set(route_of_frame))
    rid = np.array([routes_u.index(r) for r in route_of_frame])
    n_f = len(route_of_frame)
    arr = {}
    for cname, c in (("vehicle", VEHICLE), ("pedestrian", PEDESTRIAN)):
        for mode in ("clean", "overlay"):
            arr[(cname, mode)] = np.array([perframe.get((k, mode), {}).get(c, (0, 0)) for k in range(n_f)], dtype=np.float64)  # (n, 2)
    rng = np.random.default_rng(a.seed)
    by_route = [np.where(rid == r)[0] for r in range(len(routes_u))]
    for cname in ("vehicle", "pedestrian"):
        diffs = []
        for _ in range(2000):
            pick = rng.integers(0, len(routes_u), len(routes_u))
            sel = np.concatenate([by_route[r] for r in pick])
            iou = {m: arr[(cname, m)][sel, 0].sum() / max(arr[(cname, m)][sel, 1].sum(), 1.0) for m in ("clean", "overlay")}
            if arr[(cname, "clean")][sel, 1].sum() > 0:
                diffs.append(iou["overlay"] - iou["clean"])
        diffs = np.array(diffs)
        frames_with = int(((arr[(cname, "clean")][:, 1] > 0)).sum())
        boot[cname] = {"delta_iou_mean": float(diffs.mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                       "frames_with_class_in_corridor": frames_with, "routes": len(routes_u), "resamples": len(diffs)}
    out = {"routes": picked, "frames": len(idx), "corridor_fraction_of_pixels": float(np.mean(corridor_frac)), "missing_measurements": reader.missing,
           "result": res, "overlay_minus_clean": delta, "bootstrap_over_routes": boot}
    print(json.dumps(out, indent=1))
    os.makedirs(a.workdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.workdir, "overlay_effect_on_tfpp_decoders.json"), "w"), indent=1)

    if a.use_mlflow:
        try:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            import mlflow
            from src.config import paths
            mlflow.set_tracking_uri(f"file:{paths.mlruns_dir()}")
            mlflow.set_experiment("A47_overlay_effect_on_tfpp_decoders")
            with mlflow.start_run(run_name="tfpp_all_towns_e30"):
                mlflow.log_params({"routes": len(picked), "frames": len(idx), "seed": a.seed, "route_key": a.route_key, "route_points": a.route_points})
                for sc in scopes:
                    for mode in ("clean", "overlay"):
                        mlflow.log_metrics({f"{mode}/{sc}/{k}": v for k, v in res[mode][sc].items() if not isinstance(v, list)})
                    mlflow.log_metrics({f"delta/{sc}/{k}": v for k, v in delta[sc].items()})
                mlflow.log_artifact(os.path.join(a.workdir, "overlay_effect_on_tfpp_decoders.json"))
        except Exception as e:
            print("mlflow logging failed:", e)


if __name__ == "__main__":
    main()
