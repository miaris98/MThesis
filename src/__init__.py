"""Vision-based Deep Reinforcement Learning for Autonomous Driving in CARLA."""
import numpy as np

# NumPy 2.x backward-compatibility shim for TensorBoard and legacy libraries
try:
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'complex_'):
        np.complex_ = np.complex128
except Exception:
    pass

__version__ = "2.0.0"
