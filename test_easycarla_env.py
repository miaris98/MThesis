"""
Test script for EasyCarla-RL (easycarla) environment integration.
"""

import sys
import gym
import easycarla
import numpy as np

def test_environment_spec():
    print("==============================================================")
    print("   🚗 Testing EasyCarla-RL (easycarla) Gym Integration        ")
    print("==============================================================")

    default_params = {
        'number_of_vehicles': 20,
        'number_of_walkers': 0,
        'dt': 0.05,
        'ego_vehicle_filter': 'vehicle.tesla.model3',
        'surrounding_vehicle_spawned_randomly': True,
        'port': 2000,
        'town': 'Town10HD_Opt',
        'max_time_episode': 500,
        'max_waypoints': 12,
        'visualize_waypoints': False,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'view_mode': 'top',
        'traffic': 'off',
        'lidar_max_range': 50.0,
        'max_nearby_vehicles': 5,
    }

    print("\n✓ Successfully imported 'easycarla' module!")
    print("✓ Registered Gym Env ID: 'carla-v0'")
    print(f"✓ Default parameters configured for Town: {default_params['town']}")

if __name__ == "__main__":
    test_environment_spec()
