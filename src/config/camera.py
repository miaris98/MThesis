"""The one definition of this project's camera - geometry and preprocessing - for both paths.

WHY THIS EXISTS
---------------
Challenge group 11.4 is a catalogue of the same bug happening seven times: evaluation quietly
fed the policy something other than what training produced, every instance ran without error,
and none were detectable from the evaluation output. Its stated resolution was that "every
shared transformation between training and evaluation was moved to one definition used by both
paths, rather than being reimplemented in the evaluation script."

The camera itself had never been moved. It was declared twice - once implicitly by whatever
collected the PDM-Lite dataset, once explicitly in `WorldOnRailsAgent.sensors()` - and the two
declarations disagreed on four axes at once:

                      training data        evaluation agent
    position x        -1.5 m               +1.3 m
    height z           2.0 m                1.3 m
    horizontal FOV     110 deg              100 deg
    vertical FOV      ~71 deg              ~100 deg   (derived from fov and aspect)
    resolution         1024x512 (2:1)       256x256 (1:1)

So the policy learned an image-to-waypoint mapping from a camera mounted 1.5 m *behind* the ego
origin at 2.0 m height seeing a 110 deg x 71 deg cone, and at evaluation was handed a camera
2.8 m further forward, 0.7 m lower, seeing a 100 deg x 100 deg cone squeezed into a square. Every
object sits at a different image position and scale than the geometry the network was fitted to.
Nothing raises; the car just drives worse, and ~60% of this project's Leaderboard infractions are
collisions with vehicles.

The numbers below are the *data collection* geometry (carla_garage `team_code/config.py`:
`camera_pos`, `camera_rot_0`, `camera_width/height`, `camera_fov`), because that is the one of
the two that cannot be changed retroactively - the dataset is already rendered. The agent is
what moves.
"""
import math
from typing import Any, Dict, Optional, Sequence, Tuple

# Data-collection camera for autonomousvision/PDM_Lite_Carla_LB2. Do not "fix" these to match an
# agent - the dataset was rendered with exactly these and they are the ground truth.
PDM_LITE_CAMERA: Dict[str, float] = {
    "x": -1.5,
    "y": 0.0,
    "z": 2.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "width": 1024,
    "height": 512,
    "fov": 110,
}

# TransFuser++ crops the bottom quarter (512 -> 384) before its encoder: that strip is the ego's
# own bonnet, identical in every frame, so it carries no information. The CARLA-pretrained
# weights this project now uses were trained on that crop.
DEFAULT_CROP_BOTTOM_FRAC = 0.25

# Aspect-preserving default: 1024x512 cropped to 1024x384 is 8:3, and 512x192 is the same ratio
# at half scale. Cache size is what stops us simply using 1024x384 (see train_wor.py --img_size).
DEFAULT_IMG_SIZE: Tuple[int, int] = (192, 512)  # (H, W)


def vertical_fov_deg(fov_deg: float = None, width: int = None, height: int = None) -> float:
    """Vertical FOV implied by CARLA's horizontal `fov` and the image aspect ratio.

    CARLA's `fov` sensor attribute is *horizontal*. Two cameras with the same `fov` but different
    aspect ratios therefore see vertically different cones, which is the half of the mismatch
    above that is easy to miss by reading sensor dicts alone.
    """
    fov_deg = PDM_LITE_CAMERA["fov"] if fov_deg is None else fov_deg
    width = PDM_LITE_CAMERA["width"] if width is None else width
    height = PDM_LITE_CAMERA["height"] if height is None else height
    return 2.0 * math.degrees(math.atan(math.tan(math.radians(fov_deg) / 2.0) * height / width))


def camera_sensor_spec(sensor_id: str = "rgb_front") -> Dict[str, Any]:
    """The Leaderboard sensor dict that reproduces the data-collection camera.

    Used by the agent's `sensors()` so the simulator renders the geometry the network was
    trained on. The agent then applies `preprocess_rgb` to match the training-time crop/resize -
    requesting the right camera is only half of parity.
    """
    spec = {"type": "sensor.camera.rgb", "id": sensor_id}
    spec.update({k: v for k, v in PDM_LITE_CAMERA.items()})
    return spec


def preprocess_rgb(
    img,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    crop_bottom_frac: float = DEFAULT_CROP_BOTTOM_FRAC,
    route_xy=None,
    overlay: bool = False,
    overlay_kwargs: Optional[Dict[str, Any]] = None,
):
    """Optionally draw the route, crop the bonnet strip, then resize - one definition, both paths.

    Accepts a PIL image (dataset path, decoded from JPEG) or an HxWx3 uint8 array (agent path,
    straight from CARLA) and always returns an HxWx3 uint8 array. Both inputs travel the same
    PIL code path on purpose: two resize implementations that agree "closely enough" is how the
    colour-channel and speed-unit mismatches in 11.4 survived for as long as they did.

    `img_size` is (H, W). Callers should keep it on the source's post-crop aspect ratio; a square
    size re-imposes the horizontal squash this function exists to avoid.

    The overlay is drawn *before* the crop and resize because the projection is defined in the
    source camera's pixel geometry (see route_overlay.py); drawing after would need the geometry
    rescaled and is an easy way to end up with a line that no longer matches the road.
    """
    from PIL import Image
    import numpy as np

    if not isinstance(img, Image.Image):
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 4:  # BGRA from CARLA
            arr = arr[:, :, [2, 1, 0]]
        img = Image.fromarray(arr.astype("uint8"))

    if overlay and route_xy is not None and len(route_xy) >= 2:
        from src.config.route_overlay import draw_route_overlay
        img = Image.fromarray(draw_route_overlay(
            np.array(img, dtype="uint8"), route_xy, **(overlay_kwargs or {})))

    if crop_bottom_frac:
        w, h = img.size
        img = img.crop((0, 0, w, int(round(h * (1.0 - crop_bottom_frac)))))

    h, w = img_size
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img, dtype="uint8")


def describe(img_size: Optional[Tuple[int, int]] = None,
             crop_bottom_frac: float = DEFAULT_CROP_BOTTOM_FRAC) -> str:
    """One-line summary for run logs, so a run directory records the geometry it assumed."""
    img_size = img_size or DEFAULT_IMG_SIZE
    src_w, src_h = PDM_LITE_CAMERA["width"], PDM_LITE_CAMERA["height"]
    cropped_h = int(round(src_h * (1.0 - crop_bottom_frac)))
    return (f"camera @({PDM_LITE_CAMERA['x']}, {PDM_LITE_CAMERA['y']}, {PDM_LITE_CAMERA['z']}) "
            f"fov {PDM_LITE_CAMERA['fov']}deg h / {vertical_fov_deg():.1f}deg v | "
            f"{src_w}x{src_h} -> crop {src_w}x{cropped_h} -> resize {img_size[1]}x{img_size[0]}")
