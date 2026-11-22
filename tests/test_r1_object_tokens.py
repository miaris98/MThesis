"""2026-11-19 R1 (teacher ladder step 2): detector object tokens APPENDED to the camera tokens. Single RGB camera at inference: the live
model runs the frozen detector on its own encoder map; training reads the same tokens exported once from the cached PCA map."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.training.build_box_targets import BOX_FEATS, VALID_SLOT
from src.models.world_on_rails.object_detector import ObjectDetector, pca_project
from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.pooled_feature_cache import PooledFeatureCache, init_cache, pca_file
from src.training.wor_checkpoint import load_trainable_state

ROOT = Path(__file__).resolve().parents[1]
K, M, ENC = 8, 5, 512          # PCA channels, object tokens, resnet18 map channels
DET = {"in_ch": K, "dim": 32, "queries": 6, "layers": 1}


def _files(tmp_path, seed=0, name="det"):
    torch.manual_seed(seed)
    det = ObjectDetector(**DET)
    dp = tmp_path / f"{name}.pt"
    torch.save({"state_dict": det.state_dict(), "config": {"pca": K, "dim": 32, "queries": 6, "layers": 1}}, dp)
    bp = tmp_path / f"{name}_basis.pt"
    torch.save({"mean": torch.randn(ENC), "basis": torch.randn(K, ENC) * 0.05}, bp)
    return str(dp), str(bp)


def _policy(**kw):
    return QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True,
                                  route_points=4, **kw)


def _r1(tmp_path, max_range=1e6, threshold=0.0):
    net = _policy(object_tokens=M, object_det_cfg=DET, object_max_range=max_range, object_threshold=threshold)
    dp, bp = _files(tmp_path)
    net.load_object_detector(dp, bp)
    torch.nn.init.normal_(net.obj_proj.weight, std=0.5)   # the zero init (start = the RGB arm) would hide the tokens in these tests
    return net.eval(), dp, bp


def _detector(dp):
    det = ObjectDetector(**DET)
    det.load_state_dict(torch.load(dp, weights_only=False)["state_dict"])
    return det.eval()


ARGS = (torch.tensor([[5.0], [3.0]]), torch.tensor([3, 1]), torch.zeros(2, 4, 2))


def test_live_full_map_and_cached_tokens_give_the_same_plan(tmp_path):
    net, dp, bp = _r1(tmp_path)
    assert net.num_vision_tokens == 16 + M
    fmap = torch.randn(2, ENC, 3, 8)
    st = torch.load(bp)
    det = _detector(dp)
    with torch.no_grad():
        live = net(None, *ARGS, vision_features=fmap)
        tok = det.tokens(det(pca_project(fmap, st["mean"], st["basis"])), m=M, threshold=0.0, max_range=1e6)
        assert int(tok[..., VALID_SLOT].sum()) > 0
        pooled = net.vision_pool(fmap).flatten(1)
        cached = net(None, *ARGS, vision_features=torch.cat([pooled, tok.flatten(1)], dim=1))
        empty = net(None, *ARGS, vision_features=torch.cat([pooled, torch.zeros_like(tok).flatten(1)], dim=1))
    assert torch.allclose(live["waypoints"], cached["waypoints"], atol=1e-5)
    assert not torch.allclose(live["waypoints"], empty["waypoints"], atol=1e-4)   # the object tokens are really read


def test_tokens_drop_far_and_unlikely_detections():
    det = ObjectDetector(**DET).eval()
    out = {"box": torch.zeros(1, 6, 6), "cls": torch.zeros(1, 6, 6), "light": torch.zeros(1, 6, 3), "obj": torch.full((1, 6), 5.0)}
    out["box"][0, :, 0] = torch.tensor([5.0, 10.0, 25.0, 35.0, 45.0, 60.0]) / 20.0
    out["obj"][0, 1] = -5.0                                                     # p(object) below the threshold
    tok = det.tokens(out, m=6, threshold=0.5, max_range=30.0)
    assert sorted((tok[0, tok[0, :, VALID_SLOT] > 0, 0] * 20).round().tolist()) == [5.0, 25.0]
    assert int((det.tokens(out, m=6, threshold=0.5)[0, :, VALID_SLOT] > 0).sum()) == 5   # max_range 0 = no limit


def _stamp(net):
    return {"backbone": "resnet18", "route_points": 4, "vision_grid": [4, 4], "num_vision_tokens": net.num_vision_tokens,
            "use_target_speed": True, "target_speed_input": net.target_speed_input, "use_rail_q": net.use_rail_q,
            "use_ray_geometry": False, "crop_bottom_frac": net.crop_bottom_frac, "object_tokens": M, "object_det_cfg": DET,
            "object_max_range": 1e6, "object_threshold": 0.0}


def test_checkpoint_carries_the_detector_and_the_basis(tmp_path, capsys):
    net, _, _ = _r1(tmp_path)
    path = tmp_path / "r1.pth"
    torch.save({"epoch": 1, "model": net.state_dict(), "config": _stamp(net)}, path)
    capsys.readouterr()
    loaded = load_wor_model(checkpoint_path=str(path), backbone_name="resnet18", pretrained_backbone=False, device="cpu",
                            policy_arch="qwen10m", route_points=4).eval()
    out = capsys.readouterr().out
    assert "Missing keys: 0" in out and "Unexpected keys: 0" in out, out
    fmap = torch.randn(2, ENC, 3, 8)
    with torch.no_grad():
        assert torch.allclose(net(None, *ARGS, vision_features=fmap)["waypoints"], loaded(None, *ARGS, vision_features=fmap)["waypoints"], atol=1e-6)


def test_a_checkpoint_without_its_detector_is_refused(tmp_path):
    net, _, _ = _r1(tmp_path)
    path = tmp_path / "r1_bad.pth"
    torch.save({"epoch": 1, "model": {k: v for k, v in net.state_dict().items() if not k.startswith("obj_")}, "config": _stamp(net)}, path)
    with pytest.raises(RuntimeError, match="without its detector"):
        load_wor_model(checkpoint_path=str(path), backbone_name="resnet18", pretrained_backbone=False, device="cpu",
                       policy_arch="qwen10m", route_points=4)


def test_live_tokens_need_a_loaded_detector():
    net = _policy(object_tokens=M, object_det_cfg=DET).eval()
    with pytest.raises(RuntimeError, match="load_object_detector"):
        net(None, *ARGS, vision_features=torch.randn(2, ENC, 3, 8))


def test_resuming_an_rgb_checkpoint_keeps_the_detector_and_starts_with_empty_object_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("WOR_RESUME_WIDEN", "1")
    rgb = _policy()
    net = _policy(object_tokens=M, object_det_cfg=DET)
    dp, bp = _files(tmp_path)
    net.load_object_detector(dp, bp)
    det_before = {k: v.clone() for k, v in net.obj_det.state_dict().items()}
    frozen = [k for k in net.state_dict() if k.startswith("encoder.")]
    load_trainable_state(net, {k: v for k, v in rgb.state_dict().items() if not k.startswith("encoder.")}, frozen, "rgb.pth")
    assert float(net.obj_ready) == 1.0
    assert all(torch.equal(v, det_before[k]) for k, v in net.obj_det.state_dict().items())
    assert torch.equal(net.trunk.vision_pos[:, :16], rgb.trunk.vision_pos[:, :16])
    assert not any(p.requires_grad for p in net.obj_det.parameters()) and net.obj_proj.weight.requires_grad
    net.train()
    assert not net.obj_det.training
    net.eval()
    fmap = torch.randn(2, ENC, 3, 8)
    pooled = net.vision_pool(fmap).flatten(1)
    tok = torch.rand(2, M, BOX_FEATS)
    tok[..., VALID_SLOT] = 1.0
    with torch.no_grad():   # zero-initialised obj_proj: at the start the detections do not move the plan
        a = net(None, *ARGS, vision_features=torch.cat([pooled, tok.flatten(1)], 1))["waypoints"]
        b = net(None, *ARGS, vision_features=torch.cat([pooled, torch.zeros_like(tok).flatten(1)], 1))["waypoints"]
    assert torch.allclose(a, b, atol=1e-6)


def _cache(tmp_path, name, basis_src, n=6):
    keys = [f"{name}/r{i}/rgb/{i:04d}.jpg" for i in range(n)]
    prefix = str(tmp_path / name)
    init_cache(prefix, keys, 8, (4, 4), {"pixel_tag": "64x64", "data_dir_name": "dd"})
    np.memmap(prefix + ".done", dtype=np.uint8, mode="r+", shape=(n,))[:] = 1
    pf = pca_file(prefix, K)
    pm = np.memmap(pf, dtype=np.float16, mode="w+", shape=(n, K, 3, 8))
    pm[:] = np.random.RandomState(1).randn(n, K, 3, 8)
    pm.flush()
    bp = pf + ".basis.pt"
    torch.save(torch.load(basis_src), bp)
    json.dump({"shape": [n, K, 3, 8], "basis": bp}, open(pf + ".json", "w"))
    return prefix, keys


def test_export_writes_the_detectors_tokens_and_the_cache_serves_them(tmp_path):
    dp, bp = _files(tmp_path)
    prefix, keys = _cache(tmp_path, "base", bp)
    ck = torch.load(dp, weights_only=False)
    ck["config"]["prefix"] = prefix
    torch.save(ck, dp)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/training/export_detector_tokens.py"), "--prefix", prefix, "--detector", dp,
                        "--m", str(M), "--max_range", "30", "--threshold", "0.5", "--batch_size", "4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.load(open(f"{prefix}.boxtok{M}_det.f16.json"))
    assert meta["max_range"] == 30.0 and meta["threshold"] == 0.5 and meta["frames_exported"] == len(keys)
    det = _detector(dp)
    pm = np.memmap(pca_file(prefix, K), dtype=np.float16, mode="r", shape=(len(keys), K, 3, 8))
    with torch.no_grad():
        want = det.tokens(det(torch.from_numpy(np.asarray(pm, dtype=np.float32))), m=M, threshold=0.5, max_range=30.0).numpy()
    pc = PooledFeatureCache([prefix], "64x64", str(tmp_path / "dd"))
    for i, k in enumerate(keys):
        assert np.allclose(pc.object_tokens(k, M, "_det"), want[i], atol=2e-3)   # f16 storage


def test_export_refuses_a_cache_compressed_with_another_basis(tmp_path):
    dp, bp = _files(tmp_path)
    prefix, _ = _cache(tmp_path, "base", bp)
    _, other_bp = _files(tmp_path, seed=1, name="other")
    cov, _ = _cache(tmp_path, "cov", other_bp)                                  # fitted its own basis: detector input would be meaningless
    ck = torch.load(dp, weights_only=False)
    ck["config"]["prefix"] = prefix
    torch.save(ck, dp)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/training/export_detector_tokens.py"), "--prefix", cov, "--detector", dp],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "another PCA basis" in (r.stderr + r.stdout)


def test_dataset_serves_the_grid_and_the_tokens_as_one_flat_vector(tmp_path, monkeypatch):
    from src.training.pooled_feature_cache import rel_key
    from src.training.wor_dataset import WorldOnRailsDataset
    _, bp = _files(tmp_path)
    prefix, keys = _cache(tmp_path, "base", bp, n=2)
    tok = np.random.RandomState(2).rand(2, M, BOX_FEATS).astype(np.float16)
    mm = np.memmap(f"{prefix}.boxtok{M}_det.f16", dtype=np.float16, mode="w+", shape=(2, M, BOX_FEATS))
    mm[:] = tok
    mm.flush()
    data_dir = tmp_path / "dd"
    data_dir.mkdir()
    rgb_path = data_dir / keys[1]
    assert rel_key(str(rgb_path), str(data_dir)) == keys[1]
    monkeypatch.setenv("WOR_POOLED_CACHE", prefix)
    monkeypatch.setenv("WOR_OBJECT_TOKENS", str(M))
    ds = WorldOnRailsDataset(data_dir=str(data_dir), img_size=(64, 64))
    ds.is_synthetic = False
    ds.samples = [{"format": "pdm_lite", "rgb_path": str(rgb_path), "speed": 5.0, "command": 3, "route": [[0.0, 0.0]] * 4,
                   "target_speed": 5.0, "waypoints": [[1.0, 0.0]] * 5}]
    v = ds[0]["vision_features"]
    assert v.shape == (8 * 16 + M * BOX_FEATS,)
    assert np.allclose(v[8 * 16:].numpy().reshape(M, BOX_FEATS), tok[1].astype(np.float32))


def test_act_drives_from_the_camera_frame_alone(tmp_path):
    """Closed loop: the agent passes only the RGB frame (no simulator objects); the model runs its encoder, the PCA and the detector."""
    net, _, _ = _r1(tmp_path)
    route = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]], np.float32)
    calls = []
    orig = net.object_tokens_from_map
    net.object_tokens_from_map = lambda fmap: (calls.append(tuple(fmap.shape)), orig(fmap))[1]
    steer, throttle, brake = net.act(np.zeros((96, 256, 3), np.uint8), 5.0, command=3, device="cpu", route=route)
    assert np.all(np.isfinite([steer, throttle, brake]))
    assert calls and calls[0][1] == ENC          # the detector read the encoder's own full map
