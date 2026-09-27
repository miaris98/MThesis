"""Bench2Drive (Leaderboard 2.0) adapter for the ORIGINAL World on Rails agent (Chen et al., ICCV 2021),
as packaged by PCLA (`pcla_agents/wor/image_agent.py`; `wor_lb` = WoR's leaderboard model).

This is the thesis baseline, run as released - weights, sensors (3 wide RGB + 1 narrow RGB, speed,
GNSS), preprocessing, the GNSS waypointer and the control post-processing are all WoR's own; nothing is
tuned (memory: thesis-wor-baseline). The adapter only bridges interfaces: Bench2Drive's evaluator drives
a Leaderboard-2.0 `AutonomousAgent`, while PCLA's `ImageAgent` subclasses PCLA's copy of the
Leaderboard-1.0 base class.

Run through `run_bench2drive.sh` with
    EVAL_AGENT=<this file>
    EVAL_AGENT_CONFIG=$PCLA_ROOT/pcla_agents/wor_pretrained/leaderboard_weights/config_leaderboard.yaml
PCLA_ROOT (default /workspace/PCLA) must hold `leaderboard_codes/`, `pcla_functions/`, `pcla_agents/wor/`
and the downloaded `pcla_agents/wor_pretrained/`.
"""
import os
import sys

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

PCLA_ROOT = os.environ.get("PCLA_ROOT", "/workspace/PCLA")
# WoR's released config has log_wandb: True, which calls wandb.init() in setup and renders a debug video
# every step. Logging only - no effect on driving - and it needs a W&B account, so it is switched off.
os.environ.setdefault("WANDB_MODE", "disabled")
# pcla_agents/wor first: image_agent.py does `from utils import ...` / `from rails.models import ...`,
# and Bench2Drive's team_code has its own `utils` that would otherwise shadow WoR's.
for _p in (PCLA_ROOT, os.path.join(PCLA_ROOT, "pcla_agents", "wor")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from image_agent import ImageAgent  # noqa: E402  (PCLA's WoR agent)


def get_entry_point():
    return "WorOfficialAgent"


class WorOfficialAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        # The evaluator appends '+<save_name>' to the config string once per route without resetting
        # it (see bench2drive_agent.py's setup); the real path is everything before the first '+'.
        cfg = path_to_conf_file.split("+", 1)[0]
        self._inner = ImageAgent(cfg)
        self._inner.log_wandb = False  # skip the per-step debug-video rendering (see WANDB_MODE above)
        # ImageAgent.setup never initialises lane_changed (only destroy() and the non-lane-change branch
        # of run_step set it), so a route whose first command is a lane change would raise
        # AttributeError. Initialising it to the value run_step itself uses changes no behaviour.
        if not hasattr(self._inner, "lane_changed"):
            self._inner.lane_changed = None
        # Bench2Drive's evaluator calls set_global_plan() BEFORE setup(), so the plan may already be here.
        if getattr(self, "_pending_plan", None) is not None:
            self._inner.set_global_plan(*self._pending_plan)

    def sensors(self):
        return self._inner.sensors()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        # WoR's own (Leaderboard-1.0) downsampling of the plan, which its waypointer was built for.
        # Called before setup() by Bench2Drive's evaluator, so keep it for setup() when needed.
        self._pending_plan = (global_plan_gps, global_plan_world_coord)
        if hasattr(self, "_inner"):
            self._inner.set_global_plan(global_plan_gps, global_plan_world_coord)

    def run_step(self, input_data, timestamp):
        return self._inner.run_step(input_data, timestamp)

    def destroy(self):
        self._inner.destroy()
