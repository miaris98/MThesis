"""TF++ camera-only agent for Bench2Drive (TODO A86, 2026-11-21): carla_garage's own SensorAgent, unchanged except for the image geometry.

With `backbone = 'aim'` (the config written by scripts/training/train_tfpp_cam.py) carla_garage's agent already attaches no LiDAR (its
sensors(): LiDAR only for the other backbones), feeds the model an all-zero LiDAR grid that the camera-only backbone never reads, and
keeps its own target point, controller, stuck detector and stop-sign memory. The one change: its tick() crops the camera to 1024x384;
here the image becomes OUR input instead, the full 1024x512 frame resized to 288x768 by src.config.camera.preprocess_rgb
(src/baselines/tfpp_cam.py: crop_or_resize_like_ours), so TF++ and our policy see the same pixels.

Run (queue_A9c.sh CKE job flags): EVAL_AGENT=<repo>/scripts/eval/tfpp_cam_agent.py, EVAL_AGENT_CONFIG=<dir with config.json and
model_0031.pth>, EVAL_PYTHON=/workspace/venv_tfpp/bin/python. run_bench2drive.sh puts carla_garage/team_code first on PYTHONPATH.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.append(_REPO)   # after team_code: carla_garage's flat modules (config, model, ...) must win their names

import transfuser_utils as t_u  # noqa: E402  (carla_garage/team_code)
from sensor_agent import SensorAgent  # noqa: E402
from src.baselines.tfpp_cam import crop_or_resize_like_ours  # noqa: E402

if not getattr(t_u.crop_array, "tfpp_cam_patched", False):
    t_u.crop_array = crop_or_resize_like_ours(t_u.crop_array)


def get_entry_point():
    return "TfppCamAgent"


class TfppCamAgent(SensorAgent):

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        hw = getattr(self.config, "tfpp_cam_img_hw", None)
        if self.config.backbone != "aim" or not hw:
            raise RuntimeError(f"{path_to_conf_file} is not a TF++ camera-only model from train_tfpp_cam.py (backbone "
                               f"{self.config.backbone}, tfpp_cam_img_hw {hw})")
        print(f"[A86] TF++ camera-only: no LiDAR, image = full frame resized to {hw[0]}x{hw[1]} (our preprocess_rgb), "
              f"{self.model_count} model(s)", flush=True)
