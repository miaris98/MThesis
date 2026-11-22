"""TF++ camera-only (its `backbone = 'aim'` variant) on this project's sensor, resolution, frozen encoder and data (TODO A86, 2026-11-21).

User, 2026-11-21: "Lets go with A but lets also make sure we use the same sensors and resolution". What is ours and what is TF++'s:

  ours   the PDM-Lite front camera only (TF++'s agent attaches no LiDAR for 'aim'); the full 1024x512 frame resized to 288x768 by
         `src.config.camera.preprocess_rgb`, no crop; TF++'s image encoder FROZEN with the exact weights our policy uses (E15's
         frozen_backbone.pth = TF++'s `backbone.image_encoder.*`); our frame list, route split and recovery frames.
  TF++'s its decoder (6-layer transformer over the full 9x24 map, 10 path checkpoints + target-speed classes), its labels (computed
         by the functions of carla_garage's data.py, lifted from that file at import time, so the code is TF++'s own), its loss,
         optimizer (AdamW 3e-4, amsgrad, default weight decay 0.01), schedule (cosine warm restarts T0 1, x2: 31 epochs end a cycle),
         its 50% shifted-camera rule, its target point and its controller / agent logic at test time.
  off    colour augmentation (our policy has none), and the image-space auxiliary heads (depth / semantics / boxes / BEV): with the encoder
         frozen they cannot change what the planner sees, they only train themselves.

Training reads the frozen encoder's full stride-32 map from a pooled cache built with `--grid 9x24` (our 4x4 tokens are its
average-pool); the camera-only backbone's forward is replaced by a pass-through for that. The saved state dict is TF++'s full
LidarCenterNet (the frozen encoder's weights included), so carla_garage's agent loads it with strict=True.

carla_garage's modules import `carla`, `shapely` and CARLA's `agents` package at the top. The trainer runs without CARLA, so stand-ins are
installed for whichever of those is missing (none of their objects is used on the training path); on a CARLA box the real ones load.
"""
import ast
import os
import sys
import types
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM_CODE = os.path.join(REPO, "Carla-utils", "carla_garage", "team_code")
OUR_ENCODER_PREFIX = "encoder.encoder."        # E15's frozen_backbone.pth keys
TFPP_ENCODER_PREFIX = "backbone.image_encoder."


class _Stand(object):
    """Accepts any constructor call; only stands in for module-level names that the training path never uses."""

    def __init__(self, *a, **k):
        pass


def _stub(name: str, **attrs) -> None:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


def install_import_shims() -> Dict[str, bool]:
    """Stand-ins for carla / shapely / agents.navigation when they are missing. Returns which ones were stubbed."""
    stubbed = {}
    try:
        import carla  # noqa: F401
    except ImportError:
        _stub("carla", **{n: _Stand for n in ("Color", "Vector3D", "Location", "Rotation", "Transform", "VehicleControl", "Client")})
        stubbed["carla"] = True
    try:
        import shapely.geometry  # noqa: F401
    except ImportError:
        _stub("shapely")
        _stub("shapely.geometry", Polygon=_Stand, Point=_Stand)
        sys.modules["shapely"].geometry = sys.modules["shapely.geometry"]
        stubbed["shapely"] = True
    try:
        import agents.navigation.global_route_planner  # noqa: F401
    except ImportError:
        _stub("agents")
        _stub("agents.navigation")
        _stub("agents.navigation.global_route_planner", GlobalRoutePlanner=_Stand)
        stubbed["agents"] = True
    if TEAM_CODE not in sys.path:
        sys.path.insert(0, TEAM_CODE)
    return stubbed


@contextmanager
def _no_timm_download():
    """The camera-only backbone calls timm.create_model(pretrained=True); its weights are replaced by TF++'s at once, so never download."""
    import timm
    orig = timm.create_model

    def create(*a, **k):
        k["pretrained"] = False
        return orig(*a, **k)
    timm.create_model = create
    try:
        yield
    finally:
        timm.create_model = orig


def make_config(img_hw: Tuple[int, int] = (288, 768)):
    """carla_garage's GlobalConfig for the camera-only TF++ on our input (see the module docstring for every change)."""
    install_import_shims()
    from config import GlobalConfig  # carla_garage/team_code/config.py
    c = GlobalConfig()
    h, w = int(img_hw[0]), int(img_hw[1])
    # backbone and outputs: TF++'s own camera-only model, planning heads only
    c.backbone = "aim"
    c.image_architecture = "regnety_032"
    c.use_controller_input_prediction = True
    c.use_wp_gru = False
    c.transformer_decoder_join = True
    c.detect_boxes = 0
    c.use_semantic = False
    c.use_depth = False
    c.use_bev_semantic = False
    # input: the full frame resized to h x w (ours), no crop; the decoder's position table is sized by the stride-32 grid
    c.crop_image = False
    c.img_vert_anchors, c.img_horz_anchors = h // 32, w // 32
    c.tfpp_cam_img_hw = [h, w]
    # training recipe: TF++'s, minus colour augmentation (our policy has none); 31 epochs end the cosine cycle
    c.use_color_aug = 0
    c.augment = 1
    c.augment_percentage = 0.5
    c.epochs = 31
    return c


def build_model(config, encoder_state: Optional[Dict[str, torch.Tensor]] = None):
    """TF++'s LidarCenterNet for `config`, its image encoder loaded from `encoder_state` (E15's frozen_backbone.pth or a TF++ checkpoint)
    and frozen. Returns the model; raises if the encoder weights do not cover the encoder exactly."""
    install_import_shims()
    from model import LidarCenterNet  # carla_garage/team_code/model.py
    with _no_timm_download():
        net = LidarCenterNet(config)
    if encoder_state is not None:
        load_encoder(net, encoder_state)
    for p in net.backbone.image_encoder.parameters():
        p.requires_grad_(False)
    return net


def encoder_tensors(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """timm-named image-encoder tensors from either E15's frozen_backbone.pth (encoder.encoder.*) or a TF++ checkpoint
    (backbone.image_encoder.*)."""
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    out = {}
    for k, v in state.items():
        for p in (OUR_ENCODER_PREFIX, TFPP_ENCODER_PREFIX):
            if k.startswith(p):
                out[k[len(p):]] = v
    return out


def load_encoder(net, state: Dict[str, torch.Tensor]) -> None:
    enc = encoder_tensors(state)
    own = net.backbone.image_encoder.state_dict()
    missing = sorted(set(own) - set(enc))
    unexpected = sorted(set(enc) - set(own))
    if missing or unexpected:
        raise ValueError(f"encoder weights do not match TF++'s camera-only encoder: missing {missing[:4]} ({len(missing)}), "
                         f"unexpected {unexpected[:4]} ({len(unexpected)})")
    bad = [k for k in own if tuple(own[k].shape) != tuple(enc[k].shape)]
    if bad:
        raise ValueError(f"encoder tensor shapes differ: {bad[:4]}")
    net.backbone.image_encoder.load_state_dict(enc, strict=True)


def use_cached_maps(net) -> None:
    """Training on the cached frozen map: the camera-only backbone becomes a pass-through. The decoder's input path is unchanged:
    TF++'s model calls `fused_features, image_feature_grid = self.backbone(rgb)` and, with transformer_decoder_join, uses the grid."""
    def passthrough(image):
        return image, None
    net.backbone.forward = passthrough


# ---------------------------------------------------------------------------------------------------------------------------------
# labels: carla_garage's own data.py functions, lifted from the file so the code is TF++'s, without importing data.py (laspy, imgaug,
# sklearn at its top)
_LABEL_FUNCS = ("augment_route", "augment_target_point", "get_two_hot_encoding", "get_indices_speed_angle", "smooth_path",
                "iterative_line_interpolation")


class TfppLabels(object):
    """carla_garage data.py's label methods on a light object (self.config, self.target_speed_bins, self.angle_bins, as data.py sets)."""

    def __init__(self, config):
        self.config = config
        self.target_speed_bins = np.array(config.target_speed_bins)
        self.angle_bins = np.array(config.angle_bins)

    def __call__(self, meas: Dict, augmented: bool) -> Dict[str, np.ndarray]:
        """One frame's TF++ planning labels from its PDM-Lite measurement, as data.py's __getitem__ builds them."""
        import transfuser_utils as t_u
        c = self.config
        rot = float(meas["augmentation_rotation"]) if augmented else 0.0
        trans = float(meas["augmentation_translation"]) if augmented else 0.0
        route = meas["route"]
        if len(route) < c.num_route_points:
            route = np.array(route)
            route = np.vstack((route, np.tile(route[-1], (c.num_route_points - len(route), 1))))
        else:
            route = np.array(route[:c.num_route_points])
        route = self.augment_route(route, y_augmentation=trans, yaw_augmentation=rot)
        route = self.smooth_path(route) if c.smooth_route else route
        ts_idx, _ = self.get_indices_speed_angle(target_speed=meas["target_speed"], brake=meas["brake"], angle=meas["angle"])
        return {
            "checkpoint": np.asarray(route[:c.predict_checkpoint_len], dtype=np.float32),
            "target_point": np.asarray(self.augment_target_point(np.array(meas["target_point"]), y_augmentation=trans,
                                                                 yaw_augmentation=rot), dtype=np.float32),
            "target_speed_twohot": np.asarray(self.get_two_hot_encoding(meas["target_speed"], c.target_speeds, meas["brake"]),
                                              dtype=np.float32),
            "target_speed_index": np.int64(ts_idx),
            "speed": np.float32(meas["speed"]),
            "command": np.asarray(t_u.command_to_one_hot(meas["command"]), dtype=np.float32),
        }


def _lift_label_methods() -> None:
    install_import_shims()
    src_path = os.path.join(TEAM_CODE, "data.py")
    src = open(src_path, encoding="utf-8").read()
    found = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in _LABEL_FUNCS:
            found[node.name] = ast.get_source_segment(src, node)
    missing = [n for n in _LABEL_FUNCS if n not in found]
    if missing:
        raise RuntimeError(f"carla_garage data.py no longer defines {missing}")
    import transfuser_utils as t_u
    ns = {"np": np, "t_u": t_u}
    for name, code in found.items():
        exec(compile(code, src_path, "exec"), ns)   # TF++'s own source, unchanged
        setattr(TfppLabels, name, ns[name])


_LIFTED = False


def labels_for(config) -> TfppLabels:
    global _LIFTED
    if not _LIFTED:
        _lift_label_methods()
        _LIFTED = True
    return TfppLabels(config)


def planning_loss(net, out: Tuple, labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """TF++'s compute_loss with the planning labels only (the auxiliary heads are not built), weighted as its trainer does."""
    pred_wp, pred_target_speed, pred_checkpoint = out[0], out[1], out[2]
    c = net.config
    ts = labels["target_speed_twohot"] if c.use_twohot_target_speeds else labels["target_speed_index"]
    losses = net.compute_loss(pred_wp=pred_wp, pred_target_speed=pred_target_speed, pred_checkpoint=pred_checkpoint, pred_semantic=None,
                              pred_bev_semantic=None, pred_depth=None, pred_bounding_box=None, pred_wp_1=None, selected_path=None,
                              waypoint_label=None, target_speed_label=ts, checkpoint_label=labels["checkpoint"], semantic_label=None,
                              bev_semantic_label=None, depth_label=None, center_heatmap_label=None, wh_label=None, yaw_class_label=None,
                              yaw_res_label=None, offset_label=None, velocity_label=None, brake_target_label=None, pixel_weight_label=None,
                              avg_factor_label=None)
    total = sum(c.detailed_loss_weights[k] * v for k, v in losses.items())
    return {"total": total, **losses}


def crop_or_resize_like_ours(orig_crop):
    """For carla_garage's agent (scripts/eval/tfpp_cam_agent.py): its tick() turns the camera image into the network input with
    `t_u.crop_array(config, rgb)` (after its JPEG round trip and BGR -> RGB). For a camera-only config built here (it carries
    `tfpp_cam_img_hw`) that becomes OUR preprocessing, src.config.camera.preprocess_rgb: the full frame, no crop, PIL bilinear to (h, w),
    the same function our dataset and our agent use. Anything else (depth / semantics arrays, other configs) keeps TF++'s crop."""
    from src.config.camera import preprocess_rgb

    def crop_array(config, images_i):
        hw = getattr(config, "tfpp_cam_img_hw", None)
        if hw and getattr(images_i, "ndim", 0) == 3 and images_i.shape[-1] == 3:
            return preprocess_rgb(images_i, img_size=(int(hw[0]), int(hw[1])), crop_bottom_frac=0.0)
        return orig_crop(config, images_i)
    crop_array.tfpp_cam_patched = True
    return crop_array


def save_tfpp_dir(net, config, out_dir: str, epoch: int) -> str:
    """carla_garage's agent format: config.json (jsonpickle of the GlobalConfig) + model_<epoch>.pth (the full state dict, strict)."""
    import jsonpickle
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(jsonpickle.encode(config))
    path = os.path.join(out_dir, f"model_{epoch:04d}.pth")
    torch.save(net.state_dict(), path)
    return path
