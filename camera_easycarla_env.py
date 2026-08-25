"""Facade entry point re-exporting CameraEasyCarlaEnv for backward compatibility."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.camera_easycarla_env import CameraEasyCarlaEnv
from src.envs.base_env import wait_for_carla_server
from src.envs.camera_sensor import CameraSensorManager
from src.envs.reward_calculator import RewardCalculator

__all__ = [
    "CameraEasyCarlaEnv",
    "wait_for_carla_server",
    "CameraSensorManager",
    "RewardCalculator"
]

if __name__ == "__main__":
    print("✓ CameraEasyCarlaEnv facade module loaded successfully!")
