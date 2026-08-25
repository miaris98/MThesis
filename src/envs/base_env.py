"""Base CARLA environment helpers, server connection probing, and actor cleanup."""
import os
import sys
import glob
import time
from typing import Optional, List, Any

# Auto-add local CARLA PythonAPI egg package if present
carla_root = os.environ.get("CARLA_ROOT", "/workspace/carla")
carla_dist_path = os.path.join(carla_root, "PythonAPI", "carla", "dist")
if os.path.exists(carla_dist_path):
    eggs = glob.glob(os.path.join(carla_dist_path, "carla-*-py3*.egg"))
    for p in eggs:
        if p not in sys.path:
            sys.path.insert(0, p)
    if os.path.join(carla_root, "PythonAPI", "carla") not in sys.path:
        sys.path.insert(0, os.path.join(carla_root, "PythonAPI", "carla"))

# Auto-add local EasyCarla-RL package if present
easycarla_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Carla-utils", "EasyCarla-RL")
if os.path.exists(easycarla_path) and easycarla_path not in sys.path:
    sys.path.insert(0, easycarla_path)

try:
    import carla
except ImportError:
    carla = None


def wait_for_carla_server(port: int = 2000, max_wait: int = 60) -> bool:
    """Probe CARLA RPC server until world is loaded and ready for connections."""
    if carla is None:
        return False
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            c = carla.Client('127.0.0.1', port)
            c.set_timeout(4.0)
            ver = c.get_server_version()
            if ver:
                world = c.get_world()
                map_obj = world.get_map()
                spawn_pts = map_obj.get_spawn_points()
                if len(spawn_pts) > 0:
                    time.sleep(2.0)
                    return True
        except (Exception, BaseException):
            time.sleep(1.0)
    return False


def safe_clear_carla_actors(world: Any, client: Any, actor_filters: List[str]) -> None:
    """Safely stop controllers and batch-destroy actors without socket deadlocks."""
    if world is None:
        return
    for actor_filter in actor_filters:
        try:
            for actor in world.get_actors().filter(actor_filter):
                try:
                    if hasattr(actor, 'stop') and callable(actor.stop):
                        actor.stop()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        world.tick()
    except Exception:
        pass

    batch = []
    seen_ids = set()
    for actor_filter in actor_filters:
        try:
            for actor in world.get_actors().filter(actor_filter):
                if actor.id not in seen_ids:
                    seen_ids.add(actor.id)
                    batch.append(carla.command.DestroyActor(actor.id))
        except Exception:
            pass

    if batch and client is not None:
        try:
            client.apply_batch_sync(batch, False)
        except Exception:
            try:
                client.apply_batch(batch)
            except Exception:
                pass
