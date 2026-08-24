"""Facade entry point re-exporting CarlaGymEnv for backward compatibility."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.carla_gym_env import CarlaGymEnv

__all__ = ["CarlaGymEnv"]

if __name__ == "__main__":
    print("✓ CarlaGymEnv facade module loaded successfully!")
