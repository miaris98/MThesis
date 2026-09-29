#!/usr/bin/env python3
"""Can TF++'s pretrained depth / semantic decoders run on our RGB camera alone? (TODO_ACTIVE A34, step 1)

The TF++ checkpoint we take the frozen backbone from also carries `depth_decoder` and `semantic_decoder`. They read
the image branch *after* TransFuser's fusion with LiDAR, and our agent has no LiDAR. This runs the full TF++ model on
PDM-Lite frames twice - with the frame's real LiDAR BEV (what TF++ was trained with) and with an all-zero LiDAR BEV
(what an RGB-only agent can give it) - and scores both against the dataset's ground truth:

* depth: mean absolute error of the normalised depth (TF++'s own target, png / 255), and within the nearest 30%;
* semantics: pixel accuracy and per-class IoU over TF++'s 7 perspective classes.

If the zero-LiDAR numbers are close to the real-LiDAR ones, the decoders can be computed on the fly from our camera.

    python scripts/analysis/tfpp_decoders_rgb_only.py --data /workspace/dataset/wor_trajectories \
        --tfpp /workspace/tfpp_pretrained/pretrained_models/all_towns --garage /workspace/carla_garage/team_code
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys
import types

import numpy as np
import torch


def stub_modules():
    """`transfuser_utils` imports carla and `data.py` imports imgaug; neither is used on this path (imgaug also
    fails to import on numpy 2), so both are replaced by inert modules."""
    class _Any:
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    def _mod_getattr(name):  # config.py builds carla.Color(...) at class level (PEP 562); dunders stay missing
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()

    # carla and its `agents` package (nav_planner imports GlobalRoutePlanner) are only used for driving
    for name in ("carla", "agents", "agents.navigation", "agents.navigation.global_route_planner",
                 "agents.navigation.local_planner", "agents.tools", "agents.tools.misc"):
        m = types.ModuleType(name)
        m.__getattr__ = _mod_getattr
        m.__path__ = []
        sys.modules[name] = m

    ia = types.ModuleType("imgaug")
    ia.augmenters = _Any()
    sys.modules["imgaug"] = ia
    sys.modules["imgaug.augmenters"] = ia.augmenters


def link_routes(data_root, out_dir, n_routes, seed):
    """TF++'s CARLA_Data wants <scenario dir>/Town01_..._Rep0; PDM-Lite LB2 is Town01/data/<Scenario>/Route0_Rep0."""
    routes = sorted(os.path.dirname(m) for m in glob.glob(os.path.join(data_root, "Town*", "data", "*", "*", "measurements")))
    routes = [r for r in routes if os.path.isdir(os.path.join(r, "lidar")) and os.path.isdir(os.path.join(r, "depth"))]
    random.Random(seed).shuffle(routes)
    os.makedirs(out_dir, exist_ok=True)
    picked = routes[:n_routes]
    for r in picked:
        town, _, scen, name = r.split(os.sep)[-4:]
        dst = os.path.join(out_dir, f"{town}_{scen}_{name}")
        os.makedirs(dst, exist_ok=True)
        for sub in os.listdir(r):
            if not os.path.exists(os.path.join(dst, sub)):
                os.symlink(os.path.join(r, sub), os.path.join(dst, sub))
        # the LB2 release ships route results in separate archives; CARLA_Data skips a route without this file
        res = os.path.join(dst, "results.json.gz")
        if not os.path.exists(res):
            with gzip.open(res, "wt") as f:
                json.dump({"scores": {"score_composed": 100.0}, "status": "Completed", "num_infractions": 0,
                           "infractions": {"min_speed_infractions": []}}, f)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tfpp", required=True, help="dir with config.json and model_0030_0.pth")
    ap.add_argument("--garage", required=True, help="carla_garage/team_code")
    ap.add_argument("--routes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default="/workspace/tfpp_check")
    ap.add_argument("--use_mlflow", type=int, default=1)
    a = ap.parse_args()

    stub_modules()
    if not hasattr(np, "string_"):  # data.py predates NumPy 2
        np.string_ = np.bytes_
    sys.path.insert(0, a.garage)
    import jsonpickle
    from config import GlobalConfig
    from data import CARLA_Data
    from model import LidarCenterNet

    cfg = GlobalConfig()
    cfg.__dict__.update(jsonpickle.decode(open(os.path.join(a.tfpp, "config.json")).read()).__dict__)
    cfg.val_towns = []          # use every town
    cfg.augment = 1               # data.py slices the augmented BEV image even when augment=0, so read the files...
    cfg.augment_percentage = 0.0  # ...but never use the shifted/rotated camera
    cfg.use_color_aug = 0         # nor colour augmentation (imgaug is stubbed)
    cfg.num_repetitions = max(getattr(cfg, "num_repetitions", 1), 3)
    picked = link_routes(a.data, os.path.join(a.workdir, "routes"), a.routes, a.seed)
    ds = CARLA_Data(root=[os.path.join(a.workdir, "routes")], config=cfg)
    ds.image_augmenter_func = ds.lidar_augmenter_func = lambda image: image  # imgaug is stubbed; no augmentation
    print(f"routes linked {len(picked)}, frames available {len(ds)}", flush=True)

    dev = torch.device("cuda")
    model = LidarCenterNet(cfg).to(dev).eval()
    sd = torch.load(os.path.join(a.tfpp, "model_0030_0.pth"), map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"state_dict: missing {len(missing)} unexpected {len(unexpected)}", flush=True)

    n_cls = len(cfg.semantic_weights)
    stats = {m: {"dl1": [], "dl1_near": [], "acc": [], "inter": np.zeros(n_cls), "union": np.zeros(n_cls)}
             for m in ("lidar", "zero")}
    idx = random.Random(a.seed).sample(range(len(ds)), min(a.frames, len(ds)))
    with torch.inference_mode():
        for k, i in enumerate(idx):
            d = ds[i]
            rgb = torch.from_numpy(np.ascontiguousarray(d["rgb"])).float()[None].to(dev)
            lidar = torch.from_numpy(np.ascontiguousarray(d["lidar"])).float()[None].to(dev)
            gt_d = torch.from_numpy(d["depth"]).float().to(dev)
            gt_s = torch.from_numpy(d["semantic"]).long().to(dev)
            for mode, lid in (("lidar", lidar), ("zero", torch.zeros_like(lidar))):
                _, _, img_grid = model.backbone(rgb, lid)
                pd = torch.sigmoid(model.depth_decoder(img_grid)).squeeze(1)[0]
                ps = model.semantic_decoder(img_grid).argmax(1)[0]
                s = stats[mode]
                err = (pd - gt_d).abs()
                s["dl1"].append(err.mean().item())
                near = gt_d <= torch.quantile(gt_d.flatten(), 0.3)
                s["dl1_near"].append(err[near].mean().item())
                s["acc"].append((ps == gt_s).float().mean().item())
                for c in range(n_cls):
                    p, g = ps == c, gt_s == c
                    s["inter"][c] += (p & g).sum().item()
                    s["union"][c] += (p | g).sum().item()
            if k % 50 == 0:
                print(f"{k}/{len(idx)}", flush=True)

    res = {}
    for mode, s in stats.items():
        iou = s["inter"] / np.maximum(s["union"], 1)
        res[mode] = {"depth_l1": float(np.mean(s["dl1"])), "depth_l1_near30": float(np.mean(s["dl1_near"])),
                     "sem_acc": float(np.mean(s["acc"])), "sem_miou": float(iou[s["union"] > 0].mean()),
                     "sem_iou": [round(float(x), 3) for x in iou]}
    print(json.dumps(res, indent=1))
    json.dump({"routes": picked, "frames": len(idx), "result": res},
              open(os.path.join(a.workdir, "tfpp_decoders_rgb_only.json"), "w"), indent=1)

    if a.use_mlflow:
        try:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            import mlflow
            from src.config import paths
            mlflow.set_tracking_uri(f"file:{paths.mlruns_dir()}")
            mlflow.set_experiment("A34_tfpp_decoders_rgb_only")
            with mlflow.start_run(run_name="tfpp_all_towns_e30"):
                mlflow.log_params({"routes": len(picked), "frames": len(idx), "seed": a.seed, "tfpp": a.tfpp})
                for mode, r in res.items():
                    mlflow.log_metrics({f"{mode}/{k}": v for k, v in r.items() if not isinstance(v, list)})
                mlflow.log_artifact(os.path.join(a.workdir, "tfpp_decoders_rgb_only.json"))
        except Exception as e:  # tracking must never lose the result
            print("mlflow logging failed:", e)


if __name__ == "__main__":
    main()
