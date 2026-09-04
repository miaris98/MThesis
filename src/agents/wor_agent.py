"""PCLA-Compatible World on Rails (WoR) Autonomous Agent for CARLA.

Implements the standard autonomous agent interface (sensors, run_step) compatible
with the Pretrained CARLA Leaderboard Agents (PCLA) framework and CARLA Leaderboard.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.models.world_on_rails.wor_loader import load_wor_model, download_pretrained_weights


class WorldOnRailsAgent:
    """
    World on Rails Autonomous Agent compatible with PCLA and CARLA PythonAPI.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "wor_nc",
        backbone_name: str = "resnet34",
        pretrained_backbone: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_type = model_type
        
        # Load or download pretrained weights
        if checkpoint_path is None:
            checkpoint_path = download_pretrained_weights(model_type=model_type)

        self.net = load_wor_model(
            checkpoint_path=checkpoint_path if checkpoint_path else None,
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=True,
            device=self.device
        )
        self.net.eval()
        self.step_counter = 0

    def sensors(self) -> List[Dict[str, Any]]:
        """
        Defines sensor setup required by the World on Rails agent.
        """
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3, "y": 0.0, "z": 1.3,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": 256, "height": 256, "fov": 100,
                "id": "rgb_front"
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "id": "gps"
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0, "y": 0.0, "z": 0.0,
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
                - 'speed': (Frame, dict with 'speed' in m/s or scalar km/h)
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
                rgb_img = rgb_img[:, :, :3]
        elif "rgb" in input_data:
            rgb_img = input_data["rgb"]
        else:
            # Fallback dummy RGB
            rgb_img = np.zeros((256, 256, 3), dtype=np.uint8)

        # 2. Extract Speed (km/h)
        if "speed" in input_data:
            speed_data = input_data["speed"]
            speed_val = speed_data[1] if isinstance(speed_data, (tuple, list)) else speed_data
            if isinstance(speed_val, dict):
                speed_kmh = float(speed_val.get("speed", 0.0)) * 3.6
            else:
                speed_kmh = float(speed_val)
        else:
            speed_kmh = 0.0

        # 3. Extract High-Level Navigation Command
        command = input_data.get("command", 2)  # Default: Follow Lane
        if isinstance(command, (tuple, list)):
            command = command[1] if len(command) > 1 else command[0]

        # 4. Neural Network Forward Inference & PID Control
        steer, throttle, brake = self.net.act(
            rgb=rgb_img,
            speed=speed_kmh,
            command=int(command),
            device=self.device
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
