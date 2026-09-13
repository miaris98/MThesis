"""Draw the planned route into the camera image, like a reversing camera's guide lines.

WHY THIS EXISTS
---------------
The route currently reaches the policy as `route_points` x/y pairs pushed through a small MLP
and concatenated into a 96-d state vector, which is then fused with the vision features *after*
the encoder. So the network is handed an abstract list of ego-frame coordinates and a picture,
and has to learn the correspondence between them - which pixel each route point projects to -
using a frozen encoder that was never trained to answer that question.

Rendering the route onto the ground plane in the image puts both in the same spatial frame
before the encoder sees either. The information content is unchanged; what changes is that the
correspondence is given rather than inferred. This is the same trick a car's reversing camera
uses on the driver: the trajectory is drawn where it will actually go.

HONEST CAVEATS
--------------
1. Painting pixels moves the image off the distribution the CARLA-pretrained encoder was trained
   on (TransFuser++ never saw overlaid lines). A thin, low-alpha polyline is a small
   perturbation, but it is a perturbation - this belongs in an A/B against the identical run
   without it, not switched on by assumption.
2. It only helps where the route is visible. Points behind the ego or outside the frustum are
   dropped; at a sharp junction much of the route can be off-frame, which is exactly where
   navigation matters most.
3. It must be applied identically in training and evaluation or it becomes another 11.4 entry.
   That is why it lives next to camera.py and is driven by the same geometry constants.

GEOMETRY
--------
CARLA's camera uses a standard pinhole model with square pixels and a principal point at the
image centre. The focal length follows from the horizontal FOV:

    f = (W / 2) / tan(fov_h / 2)

Ego coordinates in CARLA are x forward, y right, z up. The camera sits at `PDM_LITE_CAMERA`'s
(x, y, z) offset with no rotation, so a point is first shifted into camera-relative coordinates
and then projected.
"""
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from src.config.camera import PDM_LITE_CAMERA


def focal_length_px(width: Optional[int] = None, fov_deg: Optional[float] = None) -> float:
    """Pinhole focal length in pixels for CARLA's horizontal FOV convention."""
    width = PDM_LITE_CAMERA["width"] if width is None else width
    fov_deg = PDM_LITE_CAMERA["fov"] if fov_deg is None else fov_deg
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def project_ego_points(
    points_xy: Sequence[Sequence[float]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    fov_deg: Optional[float] = None,
    cam_xyz: Optional[Tuple[float, float, float]] = None,
    min_forward: float = 0.5,
) -> np.ndarray:
    """Project ego-frame ground points (x forward, y right, z=0) to pixel coordinates.

    Returns an (N, 2) float array of (u, v) for the points that are in front of the camera;
    points closer than `min_forward` metres are dropped, because the projection diverges as the
    forward distance approaches zero and would otherwise throw a handful of points to infinity
    and smear a line across the whole frame.
    """
    width = PDM_LITE_CAMERA["width"] if width is None else width
    height = PDM_LITE_CAMERA["height"] if height is None else height
    cam_x, cam_y, cam_z = cam_xyz if cam_xyz is not None else (
        PDM_LITE_CAMERA["x"], PDM_LITE_CAMERA["y"], PDM_LITE_CAMERA["z"])

    f = focal_length_px(width, fov_deg)
    cu, cv = width / 2.0, height / 2.0

    out = []
    for p in points_xy:
        # Ego -> camera-relative. The camera is mounted behind and above the ego origin, so a
        # route point 10 m ahead is 11.5 m ahead *of the camera* and 2 m below it.
        fwd = float(p[0]) - cam_x
        right = float(p[1]) - cam_y
        down = cam_z  # ground plane (z=0 in ego frame) is cam_z below the camera
        if fwd <= min_forward:
            continue
        u = cu + f * (right / fwd)
        v = cv + f * (down / fwd)
        out.append((u, v))
    return np.asarray(out, dtype=np.float32).reshape(-1, 2)


def draw_route_overlay(
    rgb: np.ndarray,
    route_xy: Sequence[Sequence[float]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    fov_deg: Optional[float] = None,
    cam_xyz: Optional[Tuple[float, float, float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.55,
    thickness: int = 5,
    lane_half_width_m: float = 0.0,
) -> np.ndarray:
    """Blend the projected route into `rgb` (HxWx3 uint8) and return a new array.

    `rgb` must be the *source-resolution* frame: the projection uses the camera's pixel
    geometry, so drawing has to happen before any crop/resize. `lane_half_width_m` > 0 draws two
    parallel rails at that lateral offset instead of a single centre line, which is closer to
    what a reversing camera shows and gives the encoder a width cue as well as a direction.
    """
    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    width = w if width is None else width
    height = h if height is None else height

    if lane_half_width_m > 0:
        lines = [
            [(float(p[0]), float(p[1]) - lane_half_width_m) for p in route_xy],
            [(float(p[0]), float(p[1]) + lane_half_width_m) for p in route_xy],
        ]
    else:
        lines = [[(float(p[0]), float(p[1])) for p in route_xy]]

    layer = Image.new("RGB", (w, h), (0, 0, 0))
    mask = Image.new("L", (w, h), 0)
    dl, dm = ImageDraw.Draw(layer), ImageDraw.Draw(mask)

    drew = False
    for line in lines:
        pts = project_ego_points(line, width, height, fov_deg, cam_xyz)
        if pts.shape[0] >= 2:
            xy = [(float(u), float(v)) for u, v in pts]
            dl.line(xy, fill=color, width=thickness)
            dm.line(xy, fill=int(round(255 * alpha)), width=thickness)
            drew = True

    if not drew:
        # Nothing projected in front of the camera - return the frame untouched rather than a
        # blend against an empty layer, so "no visible route" is exactly the original image.
        return rgb

    base = Image.fromarray(rgb.astype("uint8"))
    return np.array(Image.composite(layer, base, mask), dtype=np.uint8)
