"""Put CARLA's Python client on the import path.

CARLA 0.9.x ships its client as an egg under ``$CARLA_ROOT/PythonAPI/carla/dist`` and its navigation agents
(``agents.navigation...``) under ``$CARLA_ROOT/PythonAPI/carla``. Six entry points used to copy the same block;
this is the one copy. ``CARLA_ROOT`` wins; otherwise ``$WORKSPACE/carla`` (the box layout, scripts/lib/common.sh).
On a machine without CARLA (this PC, CI) it changes nothing, so ``import carla`` fails there as before.
"""
import glob
import os
import sys


def carla_root() -> str:
    """The CARLA install this process should use."""
    return os.environ.get("CARLA_ROOT") or os.path.join(os.environ.get("WORKSPACE", "/workspace"), "carla")


def add_carla_to_path(with_agents: bool = True) -> str:
    """Prepend the client egg(s) and, with ``with_agents``, ``PythonAPI/carla``; returns the CARLA root used."""
    root = carla_root()
    dist = os.path.join(root, "PythonAPI", "carla", "dist")
    if os.path.isdir(dist):
        for egg in glob.glob(os.path.join(dist, "carla-*-py3*.egg")):
            if egg not in sys.path:
                sys.path.insert(0, egg)
        api = os.path.join(root, "PythonAPI", "carla")
        if with_agents and api not in sys.path:
            sys.path.insert(0, api)
    return root
