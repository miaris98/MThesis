"""Regression tests for the 2026-09-28 bug hunt on the CARLA training/eval path.

Each test pins one defect that ran without an error before:
  * a resumed run could overwrite best_model.pth with a worse epoch (threshold taken from the resumed epoch);
  * a resume whose keys did not match (torch.compile's "_orig_mod." prefix) trained random heads silently;
  * a failed checkpoint write only printed a thread traceback while training went on without checkpoints;
  * a heads-only checkpoint without frozen_backbone.pth evaluated on a random backbone (S-072);
  * the overlay cache key ignored route_points, so 4- and 20-point runs shared cached images;
  * a frame without its JPEG trained as an all-black image, and that black frame was cached for good;
  * WOR_TFPP_BRAKE_RULE applied to the qwen arm only.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.wor_checkpoint import CheckpointWriter, best_metric_so_far, load_trainable_state
from src.training.wor_dataset import WorldOnRailsDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


# ---- resume ------------------------------------------------------------------------------------------
def test_resume_threshold_is_the_stored_best_not_the_resumed_epoch(tmp_path):
    torch.save({"metrics": {"val_loss": 0.30}}, tmp_path / "best_model.pth")
    assert best_metric_so_far(str(tmp_path), "val_loss", {"metrics": {"val_loss": 0.50}}) == pytest.approx(0.30)


def test_resume_threshold_falls_back_to_the_resumed_epoch(tmp_path):
    assert best_metric_so_far(str(tmp_path), "val_loss", {"metrics": {"val_loss": 0.50}}) == pytest.approx(0.50)
    assert best_metric_so_far(str(tmp_path), "val_loss", {"metrics": {}}) == float("inf")


def test_resume_accepts_a_compiled_runs_keys_in_an_uncompiled_model():
    src, dst = _Tiny(), _Tiny()
    state = {"_orig_mod." + k: v for k, v in src.state_dict().items() if not k.startswith("encoder.")}
    load_trainable_state(dst, state, frozen_keys=["encoder.weight", "encoder.bias"], source="ckpt")
    assert torch.equal(dst.head.weight, src.head.weight)


def test_resume_refuses_a_checkpoint_that_leaves_trainable_tensors_untouched():
    with pytest.raises(RuntimeError, match="trainable tensors"):
        load_trainable_state(_Tiny(), {"something_else.weight": torch.zeros(2, 2)},
                             frozen_keys=["encoder.weight", "encoder.bias"], source="ckpt")


def test_failed_checkpoint_write_stops_training(tmp_path):
    writer = CheckpointWriter(_Tiny(), str(tmp_path), freeze_backbone=True)
    writer.save_async([({"x": 1}, str(tmp_path / "no_such_dir" / "latest_model.pth"))])
    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        writer.await_save()
    writer.await_save()  # reported once, not forever


# ---- evaluation loader -----------------------------------------------------------------------------
def test_heads_only_checkpoint_without_its_backbone_is_refused(tmp_path):
    from src.models.world_on_rails.wor_loader import load_wor_model
    ckpt = tmp_path / "model_epoch_015.pth"
    torch.save({"partial": True, "frozen_ref": "frozen_backbone.pth", "model": {}, "config": {}}, ckpt)
    with pytest.raises(FileNotFoundError, match="frozen_backbone.pth"):
        load_wor_model(str(ckpt), backbone_name="resnet18", pretrained_backbone=False)


# ---- dataset -----------------------------------------------------------------------------------------
def _ds(tmp_path, **kw):
    return WorldOnRailsDataset(data_dir=str(tmp_path / "absent"), img_size=(192, 512), **kw)


def test_overlay_cache_key_includes_route_points(tmp_path):
    a = _ds(tmp_path, route_overlay=True, route_points=4, crop_bottom_frac=0.25).pixel_cache_tag()
    b = _ds(tmp_path, route_overlay=True, route_points=20, crop_bottom_frac=0.25).pixel_cache_tag()
    assert a != b
    # Without an overlay the route never touches the pixels, so it must not split the cache.
    assert (_ds(tmp_path, route_points=4).pixel_cache_tag() == _ds(tmp_path, route_points=20).pixel_cache_tag())


def test_crop_is_keyed_by_value_and_the_canonical_names_are_kept(tmp_path):
    assert _ds(tmp_path).pixel_cache_tag() == "192x512"
    assert _ds(tmp_path, crop_bottom_frac=0.25).pixel_cache_tag() == "192x512c"
    assert _ds(tmp_path, crop_bottom_frac=0.3).pixel_cache_tag() != "192x512c"


def test_overlay_kwargs_change_the_cache_key(tmp_path):
    plain = _ds(tmp_path, route_overlay=True).pixel_cache_tag()
    thick = _ds(tmp_path, route_overlay=True, overlay_kwargs={"thickness": 9}).pixel_cache_tag()
    assert plain != thick


def _pdm_route(root: Path, n_frames: int, missing_rgb=()):
    from PIL import Image
    route = root / "Town01" / "route_0"
    (route / "measurements").mkdir(parents=True)
    (route / "rgb").mkdir()
    for i in range(n_frames):
        m = np.eye(4)
        m[0, 3] = float(i)
        meas = {"speed": 5.0, "command": 4, "ego_matrix": m.tolist(),
                "route": [[float(k), 0.0] for k in range(20)], "target_speed": 5.0}
        with gzip.open(route / "measurements" / f"{i:04d}.json.gz", "wt") as f:
            json.dump(meas, f)
        if i not in missing_rgb:
            Image.fromarray(np.full((32, 64, 3), 128, np.uint8)).save(route / "rgb" / f"{i:04d}.jpg")
    return route


def test_frames_without_an_image_are_skipped_not_trained_as_black(tmp_path):
    _pdm_route(tmp_path, n_frames=16, missing_rgb=(6,))
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=True)
    # Indexed frames are 5..8 (range(5, n - pred_len - 2)); frame 6 has no JPEG.
    assert len(ds.samples) == 3 and ds.missing_rgb == 1
    assert all(Path(s["rgb_path"]).exists() for s in ds.samples)


def test_black_fallback_frame_is_never_written_to_the_cache(tmp_path):
    ds = _ds(tmp_path, cache_decoded=True)
    missing = tmp_path / "gone.jpg"
    assert not ds._load_rgb(str(missing)).any()
    assert not list(tmp_path.glob("gone.jpg.*.npy"))


# ---- controller --------------------------------------------------------------------------------------
def test_tfpp_brake_rule_is_off_by_default_and_shared_by_both_arms(monkeypatch):
    from src.models.world_on_rails.pid_controller import tfpp_brake_override
    monkeypatch.delenv("WOR_TFPP_BRAKE_RULE", raising=False)
    assert tfpp_brake_override(0.5, 0.0, speed_mps=5.0, target_speed_kmh=0.0) == (0.5, 0.0)
    monkeypatch.setenv("WOR_TFPP_BRAKE_RULE", "1")
    assert tfpp_brake_override(0.5, 0.0, speed_mps=5.0, target_speed_kmh=0.0) == (0.0, 1.0)
    assert tfpp_brake_override(0.5, 0.0, speed_mps=5.0, target_speed_kmh=36.0) == (0.5, 0.0)
    assert tfpp_brake_override(0.5, 0.0, speed_mps=5.0, target_speed_kmh=None) == (0.5, 0.0)
    for path in ("src/models/world_on_rails/wor_policy.py", "src/models/world_on_rails/qwen_wor_policy.py"):
        assert "tfpp_brake_override(" in (REPO_ROOT / path).read_text(encoding="utf-8"), path
