"""The agent must drive a checkpoint with the preprocessing that checkpoint was trained under.

Training and evaluation now share camera.py's transform, but sharing the *function* is not
enough - it also has to be called with the same arguments. run_config.json is the carrier, and
these tests are what stop a checkpoint being driven at settings it never saw.
"""
import json
import os

import pytest

from src.agents.wor_agent import WorldOnRailsAgent
from src.config.camera import DEFAULT_IMG_SIZE, DEFAULT_CROP_BOTTOM_FRAC


class _StubAgent(WorldOnRailsAgent):
    """Exercises only the preprocessing-config logic, without building a network."""

    def __init__(self, checkpoint_path, **kw):
        self.img_size = tuple(kw.pop("img_size", DEFAULT_IMG_SIZE))
        self.crop_bottom_frac = float(kw.pop("crop_bottom_frac", DEFAULT_CROP_BOTTOM_FRAC))
        self.route_overlay = bool(kw.pop("route_overlay", False))
        if checkpoint_path:
            self._adopt_training_preprocessing(checkpoint_path)


def _write_run_config(tmp_path, **cfg):
    ckpt = tmp_path / "best_model.pth"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    (tmp_path / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(ckpt)


def test_adopts_img_size_crop_and_overlay_from_run_config(tmp_path):
    ckpt = _write_run_config(tmp_path, img_size="192x512", crop_bottom_frac=0.25,
                             route_overlay=1)
    a = _StubAgent(ckpt, img_size=(256, 256), crop_bottom_frac=0.0, route_overlay=False)
    assert a.img_size == (192, 512)
    assert a.crop_bottom_frac == 0.25
    assert a.route_overlay is True


def test_overlay_off_in_training_stays_off_at_eval(tmp_path):
    """The dangerous direction: a caller that defaults the overlay on would silently paint
    lines a checkpoint was never trained with."""
    ckpt = _write_run_config(tmp_path, img_size="256x256", crop_bottom_frac=0.0,
                             route_overlay=0)
    a = _StubAgent(ckpt, route_overlay=True)
    assert a.route_overlay is False


def test_legacy_checkpoint_without_run_config_keeps_defaults_and_warns(tmp_path, capsys):
    ckpt = tmp_path / "best_model.pth"
    ckpt.write_bytes(b"x")
    a = _StubAgent(str(ckpt), img_size=(256, 256), crop_bottom_frac=0.0)
    assert a.img_size == (256, 256)
    assert "No run_config.json" in capsys.readouterr().out


def test_list_form_img_size_is_accepted(tmp_path):
    """run_config.json records whatever train_wor.py received; tolerate both spellings."""
    ckpt = _write_run_config(tmp_path, img_size=[192, 512])
    assert _StubAgent(ckpt).img_size == (192, 512)


def test_corrupt_run_config_does_not_crash_the_agent(tmp_path, capsys):
    ckpt = tmp_path / "best_model.pth"
    ckpt.write_bytes(b"x")
    (tmp_path / "run_config.json").write_text("{not json", encoding="utf-8")
    a = _StubAgent(str(ckpt), img_size=(256, 256))
    assert a.img_size == (256, 256)
    assert "Could not read" in capsys.readouterr().out


def test_sensors_use_the_training_camera():
    """End-to-end on the real class attribute, not the stub."""
    from src.config.camera import camera_sensor_spec
    spec = camera_sensor_spec("rgb_front")
    cam = [s for s in [spec] if s["id"] == "rgb_front"][0]
    assert (cam["x"], cam["z"], cam["fov"]) == (-1.5, 2.0, 110)
    assert (cam["width"], cam["height"]) == (1024, 512)
