"""TODO A0: the WOR_BLANK_IMAGE switch must replace the camera frame and nothing else.

What it has to guarantee for the A0 reading to mean anything:
  * unset, the policy sees exactly the recorded frame (every earlier eval stays comparable);
  * `mean` / `black` remove every camera pixel but keep the route overlay, which is drawn after the blank;
  * `freeze` repeats the route's first frame and forgets it at destroy();
  * a typo fails loudly instead of quietly driving on the real camera.
"""
import numpy as np
import pytest

from src.agents import wor_agent
from src.agents.wor_agent import IMAGENET_MEAN_RGB, WorldOnRailsAgent, blank_image_mode_from_env
from src.config.camera import preprocess_rgb

H, W = 512, 1024  # the PDM-Lite source geometry the overlay projection is defined in
ROUTE = [(4.0, 0.0), (10.0, 0.0), (25.0, 0.0)]


class _FakeNet:
    def __init__(self):
        self.seen = []
        self.controller = self
        self.resets = 0

    def act(self, rgb, speed, command, device, route):
        self.seen.append(np.array(rgb, copy=True))
        return 0.0, 0.0, 0.0

    def reset(self):
        self.resets += 1


def _agent(monkeypatch, mode, overlay=False, size=(H, W)):
    if mode is None:
        monkeypatch.delenv("WOR_BLANK_IMAGE", raising=False)
    else:
        monkeypatch.setenv("WOR_BLANK_IMAGE", mode)
    a = WorldOnRailsAgent.__new__(WorldOnRailsAgent)  # no weights: only the inference path is exercised
    a.img_size, a.crop_bottom_frac, a.route_overlay = size, 0.0, overlay
    a.device, a.step_counter = "cpu", 0
    a.blank_image, a._frozen_rgb = blank_image_mode_from_env(), None
    a.net = _FakeNet()
    return a


def _frame(seed):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def _step(a, rgb, route=None):
    a.run_step({"rgb_front": (0, rgb), "speed": (0, {"speed": 3.0}), "command": 3, "route": route})
    return a.net.seen[-1]


def test_unset_passes_the_camera_frame_through_unchanged(monkeypatch):
    a = _agent(monkeypatch, None)
    rgb = _frame(0)
    assert np.array_equal(_step(a, rgb), rgb)


@pytest.mark.parametrize("mode, colour", [("mean", IMAGENET_MEAN_RGB), ("black", (0, 0, 0))])
def test_blank_modes_remove_every_camera_pixel(monkeypatch, mode, colour):
    a = _agent(monkeypatch, mode)
    seen = _step(a, _frame(1))
    assert seen.shape == (H, W, 3) and seen.dtype == np.uint8
    assert np.all(seen == np.array(colour, dtype=np.uint8))


def test_mean_is_zero_after_the_encoders_imagenet_normalisation():
    mean = np.array([0.485, 0.456, 0.406])
    assert np.max(np.abs(np.array(IMAGENET_MEAN_RGB) / 255.0 - mean)) < 0.5 / 255.0


def test_route_overlay_is_drawn_on_top_of_the_blank(monkeypatch):
    a = _agent(monkeypatch, "mean", overlay=True, size=(256, 512))
    seen = _step(a, _frame(2), route=ROUTE)
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    blank[...] = IMAGENET_MEAN_RGB
    expected = preprocess_rgb(blank, img_size=(256, 512), crop_bottom_frac=0.0, route_xy=ROUTE, overlay=True)
    assert np.array_equal(seen, expected)
    off_line = np.all(seen == np.array(IMAGENET_MEAN_RGB, dtype=np.uint8), axis=-1)
    assert 0 < (~off_line).sum() < 0.2 * off_line.size  # the line is there, everything else is the blank


def test_freeze_repeats_the_first_frame_until_destroy(monkeypatch):
    a = _agent(monkeypatch, "freeze")
    first, second = _frame(3), _frame(4)
    assert np.array_equal(_step(a, first), first)
    assert np.array_equal(_step(a, second), first)
    a.destroy()
    assert a.net.resets == 1
    assert np.array_equal(_step(a, second), second)  # the next route starts from its own first frame


def test_freeze_is_not_aliased_to_the_callers_buffer(monkeypatch):
    a = _agent(monkeypatch, "freeze")
    buf = _frame(5)
    keep = buf.copy()
    _step(a, buf)
    buf[...] = 0  # a sensor that reuses its buffer
    assert np.array_equal(_step(a, _frame(6)), keep)


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv("WOR_BLANK_IMAGE", "blnak")
    with pytest.raises(ValueError, match="WOR_BLANK_IMAGE"):
        blank_image_mode_from_env()


def test_mode_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("WOR_BLANK_IMAGE", " Mean ")
    assert blank_image_mode_from_env() == "mean"
    monkeypatch.setenv("WOR_BLANK_IMAGE", "")
    assert blank_image_mode_from_env() is None


def test_class_defaults_keep_run_step_total_without_init():
    assert WorldOnRailsAgent.blank_image is None and wor_agent.BLANK_IMAGE_MODES == ("mean", "black", "freeze")
