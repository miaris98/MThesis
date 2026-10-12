"""TODO A59 (S-125): the pooled-token feature cache must be exact (the head's own pooling applied once, offline), strict (a mismatch is an error) and wired into the dataset."""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.training.build_pooled_cache import pool_and_store
from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.training.pooled_feature_cache import PooledFeatureCache, cache_files, init_cache, rel_key
from src.training.wor_dataset import WorldOnRailsDataset

C, GRID = 8, (4, 4)


def _build(tmp_path, keys, tag="64x64", data_dir_name="dd", fill=True, seed=0):
    prefix = str(tmp_path / "cache")
    init_cache(prefix, keys, C, GRID, {"pixel_tag": tag, "data_dir_name": data_dir_name})
    jp, fp, dp = cache_files(prefix)
    rng = np.random.RandomState(seed)
    full = torch.tensor(rng.randn(len(keys), C, 9, 24).astype(np.float32))
    mm = np.memmap(fp, dtype=np.float16, mode="r+", shape=(len(keys), C, *GRID))
    done = np.memmap(dp, dtype=np.uint8, mode="r+", shape=(len(keys),))
    if fill:
        pool_and_store(full, np.arange(len(keys)), mm, done, GRID)
    mm.flush(); done.flush()
    return prefix, full


def test_round_trip_matches_the_heads_own_pooling(tmp_path):
    keys = [f"r{i}/rgb/0.jpg" for i in range(5)]
    prefix, full = _build(tmp_path, keys)
    pc = PooledFeatureCache([prefix], "64x64", str(tmp_path / "dd"))
    want = F.adaptive_avg_pool2d(full, GRID)
    for i, k in enumerate(keys):
        got = torch.tensor(pc.get(k).astype(np.float32).tolist())
        assert torch.allclose(got, want[i], atol=2e-3)  # fp16 storage only


def test_strictness(tmp_path):
    keys = ["a.jpg", "b.jpg"]
    prefix, _ = _build(tmp_path, keys)
    with pytest.raises(ValueError, match="pixel tag"):
        PooledFeatureCache([prefix], "64x64c", str(tmp_path / "dd"))
    with pytest.raises(RuntimeError, match="no pooled cache"):
        PooledFeatureCache([prefix], "64x64", str(tmp_path / "other"))
    pc = PooledFeatureCache([prefix], "64x64", str(tmp_path / "dd"))
    with pytest.raises(RuntimeError, match="not in the pooled cache"):
        pc.get("missing.jpg")


def test_incomplete_cache_is_refused(tmp_path):
    prefix, _ = _build(tmp_path, ["a.jpg", "b.jpg"], fill=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        PooledFeatureCache([prefix], "64x64", str(tmp_path / "dd"))


def test_dataset_serves_pooled_features_not_rgb(tmp_path, monkeypatch):
    data_dir = tmp_path / "dd"
    data_dir.mkdir()
    rgb_path = data_dir / "route0" / "rgb" / "0001.jpg"
    key = rel_key(str(rgb_path), str(data_dir))
    prefix, full = _build(tmp_path, [key])
    monkeypatch.setenv("WOR_POOLED_CACHE", prefix)
    ds = WorldOnRailsDataset(data_dir=str(data_dir), img_size=(64, 64))
    assert ds._pooled is not None and ds.feature_cache_tag == "pooled"
    ds.is_synthetic = False
    ds.samples = [{"format": "pdm_lite", "rgb_path": str(rgb_path), "speed": 5.0, "command": 3, "route": [[0.0, 0.0]] * 4,
                   "target_speed": 5.0, "waypoints": [[1.0, 0.0]] * 5}]
    s = ds[0]
    assert "rgb" not in s and s["vision_features"].shape == (C, *GRID)
    assert torch.allclose(s["vision_features"], F.adaptive_avg_pool2d(full, GRID)[0], atol=2e-3)


def test_policy_output_is_identical_for_full_map_and_pooled_map():
    """The head pools to vision_grid first, so feeding the already pooled map must give the same output (pooling to the same grid is the identity)."""
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, freeze_backbone=True, route_points=4, model_size="10m", vision_grid=4,
                               use_target_speed=True).eval()
    B = 3
    full = torch.randn(B, m.encoder.out_channels, 9, 24)
    pooled = F.adaptive_avg_pool2d(full, (4, 4))
    args = (torch.rand(B, 1) * 10, torch.zeros(B, dtype=torch.long), torch.randn(B, 4, 2))
    with torch.no_grad():
        a = m(None, *args, vision_features=full)
        b = m(None, *args, vision_features=pooled)
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.allclose(a[k], b[k], atol=1e-5), k
