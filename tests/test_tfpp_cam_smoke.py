"""TODO A86 end to end on a tiny fake PDM-Lite dataset (2026-11-21): build_pooled_cache.py --grid 9x24 with E15's frozen encoder, then
train_tfpp_cam.py for 2 epochs on CPU, then the saved directory loads into a fresh TF++ model with strict=True. Slow (~2-4 min, CPU
encoder); skipped without E: (E15's frozen backbone)."""
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

FB = "E:/MThesis_EXP/live_20261006_box0_train_3090ti/checkpoints/carla_armE_aug1_hires/frozen_backbone.pth"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(FB), reason="E15's frozen backbone (E:) not available")


def _write_routes(root, n_routes=6, n=16):
    from PIL import Image
    rng = np.random.RandomState(0)
    for r in range(n_routes):
        rd = root / "wor_trajectories" / "Town01" / f"Route{r}_Rep0"
        for sub in ("measurements", "rgb", "rgb_augmented"):
            (rd / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            route = [[1.0 + j, 0.05 * j * (r - 2)] for j in range(25)]
            meas = {"speed": 4.0 + r, "target_speed": [0.0, 4.0, 8.0][i % 3], "brake": bool(i % 3 == 0), "angle": 0.0, "command": 4,
                    "next_command": 4, "theta": 0.0, "steer": 0.0, "throttle": 0.5, "light_hazard": False, "stop_sign_hazard": False,
                    "junction": False, "ego_matrix": [[1, 0, 0, 1.25 * i], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "route": route, "route_original": route, "target_point": [20.0, 0.5 * (r - 2)], "target_point_next": [40.0, 0.0],
                    "augmentation_rotation": 5.0, "augmentation_translation": 0.8}
            with gzip.open(rd / "measurements" / f"{i:04d}.json.gz", "wt") as f:
                json.dump(meas, f)
            for sub in ("rgb", "rgb_augmented"):
                Image.fromarray(rng.randint(0, 255, (512, 1024, 3), dtype=np.uint8)).save(rd / sub / f"{i:04d}.jpg")
    return root / "wor_trajectories"


def _run(args, cwd):
    env = {**os.environ, "PYTHONPATH": REPO, "MLFLOW_ALLOW_FILE_STORE": "true", "WOR_ROUTE_KEY": "route_original"}
    r = subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def _build(data, out, grid, extra=""):
    _run([os.path.join(REPO, "scripts", "training", "build_pooled_cache.py"), "--data_dir", str(data), "--out_dir", str(out), "--backbone",
          "regnety_032", "--frozen_backbone", FB, "--img_size", "288x768", "--crop_bottom_frac", "0.0", "--route_overlay", "0", "--route_points",
          "4", "--route_key", "route_original", "--use_augmented_camera", "1", "--grid", grid, "--device", "cpu", "--shard", "0",
          "--num_shards", "1", "--num_workers", "0", "--batch_size", "8"] + (["--extra_grid", extra] if extra else []), REPO)
    return {p.name.split("_g")[-1][:-5]: str(p)[:-5] for p in out.glob("*.json") if "pca" not in p.name and "basis" not in p.name}


def test_cache_then_train_then_load(tmp_path):
    data = _write_routes(tmp_path)
    caches = _build(data, tmp_path / "cache", "9x24", extra="4")
    assert set(caches) == {"9x24", "4x4"}, caches
    direct = _build(data, tmp_path / "cache_direct", "4")["4x4"]
    a = np.fromfile(caches["4x4"] + ".f16", dtype=np.float16)
    b = np.fromfile(direct + ".f16", dtype=np.float16)
    assert a.size == b.size and np.array_equal(a, b), "the extra 4x4 grid must equal a direct 4x4 build bit for bit"
    assert json.load(open(caches["4x4"] + ".json"))["keys"] == json.load(open(direct + ".json"))["keys"]
    prefix = caches["9x24"]
    save = tmp_path / "run"
    log = _run([os.path.join(REPO, "scripts", "training", "train_tfpp_cam.py"), "--data_dir", str(data), "--pooled_cache", prefix,
                "--frozen_backbone", FB, "--save_dir", str(save), "--epochs", "2", "--batch_size", "4", "--num_workers", "0", "--device", "cpu",
                "--use_mlflow", "0", "--val_split", "0.34", "--save_freq", "1"], REPO)
    assert "[tfpp_cam] epoch 02/2" in log and "held-out path L1" in log
    assert "main-camera frames" in log
    rows = (save / "tfpp_cam_telemetry.csv").read_text().strip().splitlines()
    assert len(rows) == 3
    from src.baselines import tfpp_cam as tc
    import jsonpickle
    tc.install_import_shims()   # carla_garage's team_code on the path, as in its agent: config.GlobalConfig must resolve
    cfg = jsonpickle.decode((save / "tfpp_cam" / "config.json").read_text(encoding="utf-8"))
    assert type(cfg).__name__ == "GlobalConfig" and cfg.backbone == "aim"
    net = tc.build_model(cfg, None)
    net.load_state_dict(torch.load(save / "tfpp_cam" / "model_0002.pth", map_location="cpu"), strict=True)
    fb = torch.load(FB, map_location="cpu", weights_only=False)
    enc = tc.encoder_tensors(fb)
    own = net.backbone.image_encoder.state_dict()
    assert all(torch.equal(own[k], enc[k]) for k in own), "the saved model must carry E15's encoder unchanged (frozen)"
