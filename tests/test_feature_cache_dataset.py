"""Verifies the feature-caching read path in `WorldOnRailsDataset` - the mechanism this
session's whole speedup plan depends on (see struggle-solutions.md and the qwen30m/cnn
throughput measurements). Confirmed via the code-review-graph's `tests_for` query that this
path had zero direct test coverage: `test_wor_new_features_integration.py` covers the
overlay/target-speed flags but never sets `feature_cache_tag`, and no test exercises
`_load_features`, the hard-error-on-miss contract, or the rgb/vision_features exclusivity
`__getitem__` documents.

A silent bug here (wrong cache key, wrong dtype, serving a stale/mismatched sidecar) would
not show up as a crash - it would quietly train on the wrong features for hours, which is
exactly the failure class this project's own "11.4" bug catalogue is about.
"""
import numpy as np
import pytest
import torch

from src.training.wor_dataset import WorldOnRailsDataset, feature_cache_path
from tests._sanity import soft_check

FEAT_SHAPE = (8, 3, 4)  # small stand-in for the real (1512, 6, 16) cached backbone output


def _bare_dataset(tmp_path, **kwargs):
    """A dataset instance with indexing skipped (empty data_dir -> synthetic mode), then
    patched into non-synthetic mode with hand-built sample dicts. This exercises the same
    __getitem__ code path a real indexed dataset would, without needing real CARLA logs."""
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(64, 64), **kwargs)
    ds.is_synthetic = False
    ds.samples = []
    return ds


def _write_feature_cache(rgb_path, pixel_tag, feature_cache_tag, arr):
    path = feature_cache_path(str(rgb_path), pixel_tag, feature_cache_tag)
    np.save(path, arr)
    return path


def test_feature_cache_path_changes_with_pixel_tag_and_backbone_tag():
    """Two different pixel transforms, or two different backbones, must never resolve to the
    same sidecar file - that would silently serve one architecture's features to another,
    or a mismatched crop/resolution's features to the wrong run."""
    base = feature_cache_path("frame.jpg", "64x64", "regnety_032")
    diff_pixel = feature_cache_path("frame.jpg", "64x64c", "regnety_032")
    diff_backbone = feature_cache_path("frame.jpg", "64x64", "resnet34")
    assert base != diff_pixel
    assert base != diff_backbone
    assert diff_pixel != diff_backbone


def test_pixel_cache_tag_distinguishes_crop_and_overlay():
    """pixel_cache_tag() is the shared key between the pixel cache and the feature cache -
    if crop/overlay didn't change it, a cropped frame's features could be served for an
    uncropped run and vice versa (exactly the 11.4 bug class)."""
    plain = WorldOnRailsDataset(data_dir="__does_not_exist__", img_size=(64, 64))
    cropped = WorldOnRailsDataset(data_dir="__does_not_exist__", img_size=(64, 64),
                                  crop_bottom_frac=0.25)
    overlaid = WorldOnRailsDataset(data_dir="__does_not_exist__", img_size=(64, 64),
                                   route_overlay=True)
    tags = {plain.pixel_cache_tag(), cropped.pixel_cache_tag(), overlaid.pixel_cache_tag()}
    assert len(tags) == 3, tags


def test_missing_cache_entry_raises_rather_than_silently_falling_back(tmp_path):
    """feature_cache_tag's own docstring: a missing sidecar must be a hard error, never a
    per-sample fallback to live rgb (which would silently mix batch keys)."""
    ds = _bare_dataset(tmp_path, feature_cache_tag="regnety_032")
    rgb_path = tmp_path / "frame_0001.jpg"
    ds.samples = [{
        "format": "pdm_lite", "rgb_path": str(rgb_path), "speed": 5.0, "command": 3,
        "route": [[0.0, 0.0]] * 4, "target_speed": 5.0, "waypoints": [[1.0, 0.0]] * 5,
    }]
    with pytest.raises(RuntimeError, match="no cached feature"):
        ds[0]


def test_present_cache_entry_is_loaded_as_vision_features_not_rgb(tmp_path):
    """The documented invariant: exactly one of rgb/vision_features, never both."""
    ds = _bare_dataset(tmp_path, feature_cache_tag="regnety_032")
    rgb_path = tmp_path / "frame_0002.jpg"
    arr = np.random.randn(*FEAT_SHAPE).astype(np.float32)
    _write_feature_cache(rgb_path, ds.pixel_cache_tag(), "regnety_032", arr)
    ds.samples = [{
        "format": "pdm_lite", "rgb_path": str(rgb_path), "speed": 5.0, "command": 3,
        "route": [[0.0, 0.0]] * 4, "target_speed": 5.0, "waypoints": [[1.0, 0.0]] * 5,
    }]
    sample = ds[0]
    assert "vision_features" in sample
    assert "rgb" not in sample
    assert sample["vision_features"].shape == FEAT_SHAPE
    # torch.as_tensor(arr) hits this machine's numpy2/torch1.12 ABI break; tolist() sidesteps
    # the array protocol entirely and is only used here in the test's own assertion, not in
    # the production code path under test (which is exercised above via ds[0]).
    assert torch.allclose(sample["vision_features"], torch.tensor(arr.tolist()))
    # A cache entry that loads with the right shape/dtype could still be logically dead: a
    # backbone-forward that silently produced all-zeros (e.g. from a broken normalize step, or
    # reading the wrong array from a batched write) would still pass every check above. This
    # array is random-normal by construction here, so any of these firing means the fixture or
    # the read path is doing something the shape checks can't see - a real, live cache should
    # never look like this on an actual trained backbone either.
    feat = sample["vision_features"]
    soft_check(torch.isfinite(feat).all().item(),
               "loaded feature cache contains NaN/Inf - a real backbone forward would not "
               "produce this even on garbage input")
    soft_check(feat.std().item() > 1e-6,
               f"loaded feature cache is near-constant (std={feat.std().item():.2e}) - a real "
               f"cached feature map should vary across channels/positions")


def test_cache_is_bypassed_without_feature_cache_tag(tmp_path):
    """The default (no feature_cache_tag) path must keep returning raw rgb, and must not
    require any cache file to exist - this is the pre-caching behaviour every non-cached
    run still depends on."""
    ds = _bare_dataset(tmp_path)  # feature_cache_tag defaults to None
    rgb_path = tmp_path / "frame_0003.jpg"  # deliberately never created
    ds.samples = [{
        "format": "pdm_lite", "rgb_path": str(rgb_path), "speed": 5.0, "command": 3,
        "route": [[0.0, 0.0]] * 4, "target_speed": 5.0, "waypoints": [[1.0, 0.0]] * 5,
    }]
    sample = ds[0]
    assert "rgb" in sample
    assert "vision_features" not in sample


def test_wor_format_samples_also_honour_the_feature_cache_flag(tmp_path):
    """The "wor"-layout branch (route_dir + rgbs/<idx>.jpg) takes a separate code path from
    "pdm_lite" for locating rgb_path - it must respect feature_cache_tag identically."""
    ds = _bare_dataset(tmp_path, feature_cache_tag="regnety_032")
    route_dir = tmp_path / "route_00"
    (route_dir / "rgbs").mkdir(parents=True)
    rgb_path = route_dir / "rgbs" / "00000.jpg"
    arr = np.random.randn(*FEAT_SHAPE).astype(np.float32)
    _write_feature_cache(rgb_path, ds.pixel_cache_tag(), "regnety_032", arr)
    ds.samples = [{"route_dir": str(route_dir), "speed": 5.0, "command": 3}]
    sample = ds[0]
    assert "vision_features" in sample
    assert sample["vision_features"].shape == FEAT_SHAPE


def test_dataset_instance_path_matches_the_standalone_helper(tmp_path):
    """build_feature_cache.py computes sidecar paths via the standalone `feature_cache_path`
    function so it never needs to construct a feature-loading dataset first; this pins that
    the dataset's own internal path resolution can never drift from it."""
    ds = _bare_dataset(tmp_path, feature_cache_tag="regnety_032", crop_bottom_frac=0.25)
    rgb_path = str(tmp_path / "frame.jpg")
    expected = feature_cache_path(rgb_path, ds.pixel_cache_tag(), "regnety_032")
    assert ds._feature_cache_path(rgb_path) == expected
