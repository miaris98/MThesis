"""Optional chase-camera video recording for a closed-loop route.

Kept out of `eval_wor_closed_loop.py` so the measurement path does not carry the
recording path's failure modes: a broken codec, a missing ffmpeg or a full disk must
never be able to invalidate a scored run. Recording is opt-in, every call is guarded,
and a recorder that fails degrades to no video rather than to a lost route.

The HUD itself is imported from `eval_wor.py` rather than reimplemented - it is the one
piece of that script worth keeping, and a second copy would drift.
"""
import os
import subprocess
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # recording is optional; scoring is not
    cv2 = None


class RouteVideoRecorder:
    """Records one route from a chase camera, with the telemetry HUD burned in.

    Frames are written with mp4v and re-encoded to H.264 on close, because mp4v output
    will not play in most browsers - the same two-step dance `eval_wor.py` documents.
    """

    def __init__(self, world, ego, path: str, route_id: str,
                 fps: int = 20, width: int = 1280, height: int = 720):
        self.path = path
        self.route_id = route_id
        self.width, self.height = width, height
        self.ok = cv2 is not None
        self.camera = None
        self._writer = None
        self._buf = {"data": None}
        self._raw_path = path.replace(".mp4", "_raw.mp4")

        if not self.ok:
            return

        try:
            import carla
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(width))
            bp.set_attribute("image_size_y", str(height))
            bp.set_attribute("fov", "90")
            self.camera = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0)),
                attach_to=ego,
            )
            # Native CARLA buffer is BGRA; drop alpha only and keep BGR, which is what
            # cv2 expects. Converting to RGB here and back would re-swap the channels -
            # the bug that produced inverted colours in the original recorder.
            self.camera.listen(lambda img: self._buf.update({
                "data": np.frombuffer(img.raw_data, dtype=np.uint8)
                          .reshape((height, width, 4))[:, :, :3]}))
            self._writer = cv2.VideoWriter(
                self._raw_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        except Exception as e:
            print(f"[WARNING] video recorder disabled for {route_id}: {e}")
            self.ok = False

    def actors(self) -> List:
        return [self.camera] if self.camera is not None else []

    def capture(self, *, speed_kmh: float, target_speed_kmh: float, steer: float,
                throttle: float, brake: float, step: int, max_steps: int,
                command_name: str, route_progress_pct: Optional[float],
                collisions: int, elapsed_s: float, driving_score: float) -> None:
        if not self.ok or self._writer is None or self._buf["data"] is None:
            return
        try:
            from eval_wor import draw_eval_hud
            frame = draw_eval_hud(
                frame=self._buf["data"].copy(),
                speed_kmh=speed_kmh, target_speed_kmh=target_speed_kmh,
                steer=steer, throttle=throttle, brake=brake,
                step=step, max_steps=max_steps, command_name=command_name,
                route_progress_pct=route_progress_pct, collisions=collisions,
                elapsed_s=elapsed_s,
            )
            cv2.putText(frame, f"{self.route_id}  DS={driving_score:.3f}",
                        (self.width - 380, self.height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            self._writer.write(frame)
        except Exception:
            # A recording failure mid-route must not end the route being scored.
            self.ok = False

    def close(self) -> Optional[str]:
        # The camera is intentionally left listening: the caller's cleanup stops every
        # sensor it owns, and stopping here as well makes CARLA log a spurious
        # "sensor wasn't listening" warning on the second call.
        if self._writer is None:
            return None
        try:
            self._writer.release()
        except Exception:
            return None
        if not os.path.exists(self._raw_path):
            return None
        try:
            # -crf 28 / veryfast: these are watched to see how the car drives, not
            # archived. Default settings produced ~67 MB for 75 s, which is a slow
            # download off a rented box for no visible benefit at this resolution.
            subprocess.run(
                ["ffmpeg", "-y", "-i", self._raw_path, "-vcodec", "libx264",
                 "-crf", "28", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-movflags", "faststart", self.path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
            if os.path.exists(self.path):
                os.remove(self._raw_path)
                return self.path
        except Exception:
            pass
        return self._raw_path  # unconverted, but still watchable locally
