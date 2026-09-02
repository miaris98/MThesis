"""Camera sensor manager mounting 3 RGB cameras and zero-copy panorama buffer stitching."""
from typing import Dict, Optional, Any
import numpy as np

try:
    import carla
except ImportError:
    carla = None


class CameraSensorManager:
    """
    Manages 3 RGB Camera Sensors (Left -60°, Center 0°, Right +60°).
    Performs zero-copy memory copy directly into pre-allocated NumPy panorama buffer.
    """
    def __init__(self, img_width: int = 256, img_height: int = 256, sensor_tick: float = 0.05):
        self.img_width = img_width
        self.img_height = img_height
        self.sensor_tick = sensor_tick
        self.num_cameras = 3
        self.panorama_buffer = np.zeros((img_height, img_width * self.num_cameras, 3), dtype=np.uint8)
        self.sensors: Dict[str, Optional[Any]] = {"left": None, "center": None, "right": None}

    def setup_cameras(self, world: Any, ego_vehicle: Any) -> None:
        """Spawn and attach 3 RGB camera sensors to the ego vehicle."""
        if carla is None or world is None or ego_vehicle is None:
            return

        self.cleanup_cameras()
        bp_library = world.get_blueprint_library()
        cam_bp = bp_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.img_width))
        cam_bp.set_attribute('image_size_y', str(self.img_height))
        cam_bp.set_attribute('fov', '100')
        cam_bp.set_attribute('sensor_tick', str(self.sensor_tick))

        configs = {
            "left": (carla.Transform(carla.Location(x=1.3, y=0.0, z=1.4), carla.Rotation(pitch=0, yaw=-60, roll=0)), 0),
            "center": (carla.Transform(carla.Location(x=1.3, y=0.0, z=1.4), carla.Rotation(pitch=0, yaw=0, roll=0)), 1),
            "right": (carla.Transform(carla.Location(x=1.3, y=0.0, z=1.4), carla.Rotation(pitch=0, yaw=60, roll=0)), 2),
        }

        for cam_name, (transform, slot_idx) in configs.items():
            cam_sensor = world.spawn_actor(cam_bp, transform, attach_to=ego_vehicle)
            col_start = slot_idx * self.img_width
            col_end = (slot_idx + 1) * self.img_width
            
            def make_callback(start_c, end_c):
                def _callback(image):
                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = np.reshape(array, (self.img_height, self.img_width, 4))
                    # CARLA raw_data is BGRA. Convert to standard RGB [R, G, B]
                    self.panorama_buffer[:, start_c:end_c, :] = array[:, :, [2, 1, 0]]
                return _callback

            cam_sensor.listen(make_callback(col_start, col_end))
            self.sensors[cam_name] = cam_sensor

    def are_all_alive(self, ego_vehicle: Any) -> bool:
        """Check if all 3 camera sensors and the ego vehicle are alive and valid."""
        return (
            all(self.sensors.get(k) is not None and getattr(self.sensors[k], 'is_alive', False) for k in ["left", "center", "right"])
            and ego_vehicle is not None and getattr(ego_vehicle, 'is_alive', False)
        )

    def cleanup_cameras(self) -> None:
        """Stop listening and destroy all 3 camera sensor actors."""
        for k in ["left", "center", "right"]:
            sensor = self.sensors.get(k)
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
                self.sensors[k] = None
        self.panorama_buffer.fill(0)
