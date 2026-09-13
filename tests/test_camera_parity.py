"""Train/eval parity for the camera: one geometry, one preprocessing, both paths.

Challenge group 11.4 is seven instances of evaluation silently feeding the policy something
other than what training produced, none detectable from evaluation output. These tests are the
detector that was missing for the camera itself, which disagreed on four axes at once until
src/config/camera.py centralised it.
"""
import numpy as np
import pytest

from src.config.camera import (
    PDM_LITE_CAMERA, camera_sensor_spec, preprocess_rgb, vertical_fov_deg,
    DEFAULT_IMG_SIZE, DEFAULT_CROP_BOTTOM_FRAC)


def test_sensor_spec_matches_collection_geometry():
    """The agent must ask CARLA for the camera the dataset was rendered with.

    These exact numbers come from carla_garage team_code/config.py, which generated
    PDM_Lite_Carla_LB2. The dataset cannot be re-rendered, so the agent is what has to agree.
    """
    spec = camera_sensor_spec("rgb_front")
    assert spec["type"] == "sensor.camera.rgb"
    assert spec["id"] == "rgb_front"
    assert (spec["x"], spec["y"], spec["z"]) == (-1.5, 0.0, 2.0)
    assert (spec["roll"], spec["pitch"], spec["yaw"]) == (0.0, 0.0, 0.0)
    assert (spec["width"], spec["height"]) == (1024, 512)
    assert spec["fov"] == 110


def test_agent_sensor_spec_is_not_the_old_mismatched_camera():
    """Regression guard on the specific values that were wrong.

    The agent previously declared x=1.3, z=1.3, fov=100, 256x256 - 2.8 m forward, 0.7 m low and
    a different cone from the training data. If anyone reintroduces those, fail loudly.
    """
    spec = camera_sensor_spec()
    assert not (spec["x"] == 1.3 and spec["z"] == 1.3), "the pre-parity camera is back"
    assert spec["fov"] != 100, "the pre-parity FOV is back"
    assert (spec["width"], spec["height"]) != (256, 256), "the pre-parity resolution is back"


def test_vertical_fov_differs_from_horizontal_for_non_square_images():
    """CARLA's `fov` is horizontal only; the vertical cone follows the aspect ratio.

    This is the half of the old mismatch that reading sensor dicts alone would miss: 110 deg at
    2:1 is ~71 deg vertically, while 100 deg at 1:1 is ~100 deg - a ~29 deg difference in what
    the camera can see of the road, from two numbers that look comparable.
    """
    train_v = vertical_fov_deg(110, 1024, 512)
    old_eval_v = vertical_fov_deg(100, 256, 256)
    assert 70.0 < train_v < 73.0
    assert 99.0 < old_eval_v < 101.0
    assert abs(old_eval_v - train_v) > 25.0


def test_preprocess_crops_the_bonnet_and_preserves_aspect():
    src = np.zeros((512, 1024, 3), dtype=np.uint8)
    out = preprocess_rgb(src, img_size=(192, 512), crop_bottom_frac=0.25)
    assert out.shape == (192, 512, 3)
    # 1024x512 cropped to 1024x384 is 8:3; 512x192 is the same ratio, so geometry is preserved.
    assert abs((1024 / 384) - (512 / 192)) < 1e-6


def test_preprocess_actually_removes_the_bottom_strip():
    """Content check, not just a shape check: a shape-correct crop of the wrong region passes
    the assertion above while still feeding the network the wrong pixels."""
    src = np.zeros((512, 1024, 3), dtype=np.uint8)
    src[384:, :, 0] = 255  # paint only the region that should be cropped away
    out = preprocess_rgb(src, img_size=(192, 512), crop_bottom_frac=0.25)
    assert out[..., 0].max() == 0, "bottom strip survived the crop"


def test_preprocess_identical_for_pil_and_array_inputs():
    """The dataset passes PIL, the agent passes a numpy frame from CARLA.

    They must produce bit-identical arrays. 'Close enough' is exactly how the colour-order and
    speed-unit mismatches in 11.4 survived.
    """
    from PIL import Image
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    from_arr = preprocess_rgb(arr, img_size=(192, 512), crop_bottom_frac=0.25)
    from_pil = preprocess_rgb(Image.fromarray(arr, mode="RGB"),
                              img_size=(192, 512), crop_bottom_frac=0.25)
    assert np.array_equal(from_arr, from_pil)


def test_bgra_input_is_channel_reversed_not_alpha_dropped():
    """CARLA hands out BGRA. Dropping the 4th channel leaves BGR, which is 9.2a's bug."""
    bgra = np.zeros((512, 1024, 4), dtype=np.uint8)
    bgra[..., 0] = 10   # B
    bgra[..., 1] = 20   # G
    bgra[..., 2] = 30   # R
    bgra[..., 3] = 255  # A
    out = preprocess_rgb(bgra, img_size=(8, 16), crop_bottom_frac=0.0)
    r, g, b = out[0, 0]
    assert (int(r), int(g), int(b)) == (30, 20, 10), f"expected RGB(30,20,10), got {(r, g, b)}"


def test_defaults_are_aspect_preserving():
    """The shipped defaults must not reintroduce the squash."""
    h, w = DEFAULT_IMG_SIZE
    src_w = PDM_LITE_CAMERA["width"]
    src_h = PDM_LITE_CAMERA["height"] * (1.0 - DEFAULT_CROP_BOTTOM_FRAC)
    assert abs((w / h) - (src_w / src_h)) < 0.02, "default img_size distorts the source aspect"
