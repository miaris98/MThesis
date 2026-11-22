"""2026-11-18, step 2 of the teacher ladder: the object detector on the frozen RGB features predicts the teacher's tokens."""
import json
import subprocess
import sys

import numpy as np
import torch

from scripts.training.build_box_targets import BOX_CLASSES, BOX_FEATS, VALID_SLOT
from src.models.world_on_rails.object_detector import DetectionMeter, ObjectDetector, detector_loss, match


def _tok(objs, m=20):
    t = np.zeros((m, BOX_FEATS), np.float32)
    for i, (x, y, cls) in enumerate(objs):
        t[i, :6] = [x / 20, y / 10, 1.0, 0.0, 2.0 / 3, 1.0 / 3]
        t[i, 9 + BOX_CLASSES.index(cls)] = 1.0
        t[i, VALID_SLOT] = 1.0
    return t


def test_matching_pairs_each_object_once_and_loss_is_finite():
    torch.manual_seed(0)
    det = ObjectDetector(in_ch=8, dim=32, queries=6, layers=1, heads=4)
    gt = torch.as_tensor(np.stack([_tok([(10, 0, "car"), (5, 2, "walker")]), _tok([])]))
    out = det(torch.randn(2, 8, 3, 4))
    pairs = match(out, gt)
    assert len(pairs[0][0]) == 2 and len(set(pairs[0][0].tolist())) == 2 and sorted(pairs[0][1].tolist()) == [0, 1]
    assert len(pairs[1][0]) == 0
    loss, parts = detector_loss(out, gt)
    assert torch.isfinite(loss) and set(parts) == {"obj", "box", "cls", "light"}
    loss.backward()


def test_detector_learns_a_fixed_scene_and_emits_teacher_tokens():
    torch.manual_seed(0)
    det = ObjectDetector(in_ch=8, dim=32, queries=6, layers=1, heads=4)
    x = torch.randn(4, 8, 3, 4)
    gt = torch.as_tensor(np.stack([_tok([(12, -1, "car"), (6, 3, "walker"), (25, 2, "traffic_light")])] * 4))
    opt = torch.optim.AdamW(det.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _ = detector_loss(det(x), gt)
        opt.zero_grad(); loss.backward(); opt.step()
    out = det(x)
    m = DetectionMeter(tol_m=1.0)
    m.update(out, gt)
    s = m.summary()
    assert s["recall"] == 1.0 and s["precision"] == 1.0 and s["cls_acc"] == 1.0, s
    tok = det.tokens(out, m=20)
    assert tok.shape == (4, 20, BOX_FEATS) and int(tok[0, :, VALID_SLOT].sum()) == 3
    np.testing.assert_allclose(np.sort(tok[0, :3, 0].numpy() * 20), [6, 12, 25], atol=1.0)   # nearest first, metres
    assert torch.all(tok[..., 6] == 0)   # no speed from one frame


def test_training_script_smoke(tmp_path):
    n, C, H, W, M = 40, 16, 3, 4, 20
    prefix = str(tmp_path / "cache")
    keys = [f"Town01/data/S/route_{i // 4}/rgb/{i % 4:04d}.jpg" for i in range(n)]
    json.dump({"n": n, "keys": keys}, open(prefix + ".json", "w"))
    np.memmap(prefix + ".done", dtype=np.uint8, mode="w+", shape=(n,))[:] = 1
    pm = np.memmap(prefix + f".pca{C}.f16", dtype=np.float16, mode="w+", shape=(n, C, H, W)); pm[:] = np.random.randn(n, C, H, W)
    json.dump({"shape": [n, C, H, W]}, open(prefix + f".pca{C}.f16.json", "w"))
    tm = np.memmap(prefix + f".boxtok{M}_visns.f16", dtype=np.float16, mode="w+", shape=(n, M, BOX_FEATS))
    tm[:] = np.stack([_tok([(10 + i % 5, 0, "car")]) for i in range(n)])
    pm.flush(); tm.flush()
    out = tmp_path / "det"
    r = subprocess.run([sys.executable, "scripts/training/train_object_detector.py", "--prefix", prefix, "--out_dir", str(out), "--pca", str(C),
                        "--dim", "32", "--queries", "6", "--layers", "1", "--epochs", "1", "--batch_size", "8", "--num_workers", "0",
                        "--use_mlflow", "0", "--val_split", "0.25"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "FINAL" in r.stdout and (out / "detector_best.pt").exists() and (out / "metrics.json").exists()
