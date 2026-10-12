"""PCLA-Compatible World on Rails (WoR) Autonomous Agent for CARLA.

Implements the standard autonomous agent interface (sensors, run_step) compatible
with the Pretrained CARLA Leaderboard Agents (PCLA) framework and CARLA Leaderboard.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import torch

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.models.world_on_rails.wor_loader import load_wor_model, download_pretrained_weights
from src.config.camera import (
    camera_sensor_spec, preprocess_rgb, DEFAULT_IMG_SIZE, DEFAULT_CROP_BOTTOM_FRAC)


class WorldOnRailsAgent:
    """
    World on Rails Autonomous Agent compatible with PCLA and CARLA PythonAPI.
    """
    # Declared at class level, not only assigned in __init__, so `run_step` is total: it reads
    # these on every frame, and an instance built through __new__ to skip weight loading (which
    # is how the unit tests exercise the inference path without a checkpoint) would otherwise
    # raise AttributeError from inside the driving loop. __init__ assigns all of them
    # unconditionally, so this masks nothing in real use - it only gives the attributes a
    # defined value before __init__ has run.
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE
    crop_bottom_frac: float = DEFAULT_CROP_BOTTOM_FRAC
    route_overlay: bool = False
    backbone_name: str = "resnet34"
    use_target_speed: bool = False
    route_points: int = 4

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "wor_nc",
        backbone_name: str = "resnet34",
        pretrained_backbone: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        policy_arch: str = "cnn",
        route_points: int = 4,
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        crop_bottom_frac: float = DEFAULT_CROP_BOTTOM_FRAC,
        route_overlay: bool = False
    ):
        self.device = device
        self.model_type = model_type
        # Must match what the checkpoint was trained under. A checkpoint trained at one
        # resolution, crop or overlay setting and driven at another is the same class of silent
        # mismatch as the camera geometry itself. train_wor.py writes run_config.json beside
        # every checkpoint precisely so this does not have to be remembered by hand, so read it
        # rather than trusting the caller's defaults.
        self.img_size = tuple(img_size)
        self.crop_bottom_frac = float(crop_bottom_frac)
        self.route_overlay = bool(route_overlay)
        self.backbone_name = backbone_name
        self.use_target_speed = False
        self.route_points = route_points
        if checkpoint_path:
            # Deliberately before the model is built: this can change the backbone family, and
            # constructing the wrong one first would only be detectable as a partial weight load.
            self._adopt_training_preprocessing(checkpoint_path)

        # Load or download pretrained weights
        if checkpoint_path is None:
            checkpoint_path = download_pretrained_weights(model_type=model_type)

        self.net = load_wor_model(
            checkpoint_path=checkpoint_path if checkpoint_path else None,
            backbone_name=self.backbone_name,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=True,
            device=self.device,
            policy_arch=policy_arch,
            # self.route_points, not the constructor's raw default: _adopt_training_
            # preprocessing above may have overwritten it from the checkpoint's own
            # run_config.json, and load_wor_model's own stamp-resolution is the backstop
            # if this ever disagrees with it, not the primary mechanism.
            route_points=self.route_points
        )
        self.net.eval()
        self.step_counter = 0

    def _adopt_training_preprocessing(self, checkpoint_path: str):
        """Take img_size / crop / overlay from the checkpoint's own run_config.json.

        The alternative is passing them in at every call site and hoping they match what the
        run used - which is how a checkpoint ends up driven at a resolution or with an overlay
        setting it never saw, producing a plausible-looking but meaningless driving score. Falls
        back to the constructor defaults (with a warning) when the file is absent, since older
        checkpoints predate it.
        """
        import json
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                                "run_config.json")
        if not os.path.exists(cfg_path):
            print(f"[Warning] No run_config.json beside {checkpoint_path}; using preprocessing "
                  f"defaults img_size={self.img_size}, crop={self.crop_bottom_frac}, "
                  f"overlay={self.route_overlay}. If this checkpoint was trained with different "
                  f"settings, its driving score will not mean what it appears to.")
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"[Warning] Could not read {cfg_path}: {exc}")
            return

        spec = cfg.get("img_size")
        if isinstance(spec, str) and "x" in spec.lower():
            h, w = spec.lower().split("x")
            self.img_size = (int(h), int(w))
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            self.img_size = (int(spec[0]), int(spec[1]))
        self.crop_bottom_frac = float(cfg.get("crop_bottom_frac", self.crop_bottom_frac))
        self.route_overlay = bool(cfg.get("route_overlay", self.route_overlay))
        # The backbone family has to come from here too. A regnety-trained checkpoint loaded
        # into a resnet34 skeleton matches almost nothing, and load_wor_model's strict=False
        # would report that as a partial load rather than an error - which is 11.10's "harness
        # ran the wrong model without saying so", reached by a different route.
        self.backbone_name = str(cfg.get("backbone", getattr(self, "backbone_name", "resnet34")))
        # Whether the checkpoint carries a target-speed head, so the architecture matches.
        self.use_target_speed = float(cfg.get("target_speed_loss_weight", 0.0)) > 0
        # The route length is part of the input contract, not just the architecture: the
        # policy was trained on exactly this many evenly-spaced points, so the evaluator has
        # to build the same shape. Left unadopted, every eval path used its own default of 4
        # against a 20-point checkpoint.
        self.route_points = int(cfg.get("route_points", getattr(self, "route_points", 4)))
        print(f"--> From run_config.json: backbone={self.backbone_name}, "
              f"img_size={self.img_size}, crop_bottom_frac={self.crop_bottom_frac}, "
              f"route_overlay={self.route_overlay}, target_speed={self.use_target_speed}, "
              f"route_points={self.route_points}")

    def sensors(self) -> List[Dict[str, Any]]:
        """
        Defines sensor setup required by the World on Rails agent.

        The camera comes from src/config/camera.py rather than being written out here. It used
        to be declared locally as (x=1.3, z=1.3, fov=100, 256x256), which disagreed with the
        geometry PDM-Lite was actually rendered at (x=-1.5, z=2.0, fov=110, 1024x512) on four
        axes at once - so the network was evaluated through a camera 2.8 m further forward,
        0.7 m lower and ~29 deg wider vertically than the one it was trained on. See that
        module's docstring; this is 11.4's "one definition used by both paths" applied to the
        one shared transformation that had never been centralised.
        """
        return [
            camera_sensor_spec("rgb_front"),
            {
                "type": "sensor.other.gnss",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "id": "gps"
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "id": "imu"
            },
            {
                "type": "sensor.speedometer",
                "id": "speed"
            }
        ]

    def run_step(
        self,
        input_data: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> Any:
        """
        Executes one navigation step given sensory inputs.
        
        Args:
            input_data: Dictionary containing sensor readings:
                - 'rgb_front': (Frame, np.ndarray uint8 RGB image)
                - 'speed': (Frame, dict with 'speed' in m/s, or a scalar in m/s)
                - 'command': Optional high-level command (1=Left, 2=Right, 3=Straight, 4=Follow)
        Returns:
            carla.VehicleControl or dict with (steer, throttle, brake)
        """
        self.step_counter += 1

        # 1. Extract RGB image
        if "rgb_front" in input_data:
            rgb_data = input_data["rgb_front"]
            rgb_img = rgb_data[1] if isinstance(rgb_data, (tuple, list)) else rgb_data
            if rgb_img.shape[-1] == 4:  # BGRA -> RGB
                rgb_img = rgb_img[:, :, [2, 1, 0]]
        elif "rgb" in input_data:
            rgb_img = input_data["rgb"]
        else:
            # Fallback dummy RGB
            rgb_img = np.zeros((*self.img_size, 3), dtype=np.uint8)

        # 2. Extract Speed (metres per second)
        #
        # m/s is the unit throughout: PDM-Lite's measurement files store it, so it is
        # what the network trained on, and `act()` converts to km/h for the PID. This
        # previously multiplied the dict form by 3.6 and passed bare floats through
        # untouched, so the two shapes meant different units and neither matched the
        # network's. CARLA's speedometer pseudo-sensor reports {'speed': m/s}.
        if "speed" in input_data:
            speed_data = input_data["speed"]
            speed_val = speed_data[1] if isinstance(speed_data, (tuple, list)) else speed_data
            if isinstance(speed_val, dict):
                speed_mps = float(speed_val.get("speed", 0.0))
            else:
                speed_mps = float(speed_val)
        else:
            speed_mps = 0.0

        # 3. Extract High-Level Navigation Command
        command = input_data.get("command", 2)  # Default: Follow Lane
        if isinstance(command, (tuple, list)):
            command = command[1] if len(command) > 1 else command[0]

        # 4. Ego-frame planned route: the policy's navigation intent. PDM-Lite's
        # command enum is LANEFOLLOW on every training frame, so the route is what
        # actually tells the policy which way to go at a junction. Without it the
        # policy gets a zero route and drives straight regardless of the turn.
        route = input_data.get("route")
        if isinstance(route, (tuple, list)) and len(route) == 2 and not isinstance(route[0], (list, tuple, np.ndarray)):
            route = route[1]  # (frame, data) sensor-style tuple

        # 5. Preprocess the frame exactly as training did.
        #
        # Deliberately after the route is parsed, not before: when the overlay is enabled the
        # route has to be drawn into the frame, so the image cannot be finalised until the route
        # is known. Requesting the right camera is only half of parity - the frames the network
        # trained on were cropped, optionally overlaid, and resized before it ever saw them, and
        # that has to be the same transform rather than an equivalent-looking reimplementation
        # (11.4). preprocess_rgb is what the dataset calls too.
        if rgb_img.shape[:2] != tuple(self.img_size) or self.route_overlay:
            rgb_img = preprocess_rgb(
                rgb_img, img_size=self.img_size, crop_bottom_frac=self.crop_bottom_frac,
                route_xy=route, overlay=self.route_overlay)

        # 6. Neural Network Forward Inference & PID Control
        steer, throttle, brake = self.net.act(
            rgb=rgb_img,
            speed=speed_mps,
            command=int(command),
            device=self.device,
            route=route
        )

        # 5. Return CARLA VehicleControl or Control Dict
        try:
            import carla
            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = False
            control.reverse = False
            return control
        except ImportError:
            return {
                "steer": float(steer),
                "throttle": float(throttle),
                "brake": float(brake)
            }

    def destroy(self):
        """Cleans up resources upon episode completion."""
        self.net.controller.reset()
