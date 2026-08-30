"""Shared per-step telemetry and throughput reporting for the PPO and SAC trainers."""
import time
from typing import Any, Dict


class TelemetryMixin:
    """
    Per-step CSV logging and progress printing shared by trainers.

    Keeping one implementation means PPO and SAC runs land in the same CSV schema, which is
    what makes their telemetry directly comparable in MLflow and offline analysis. Consumers
    must define: csv_logger, global_step, episode_count, num_envs, cfg, last_sps, last_fps,
    last_progress_step, last_progress_time.
    """
    REWARD_FIELDS = ("r_progress", "r_lane", "r_light", "r_obstacle", "r_ttc",
                     "r_terminal", "lateral_dist")

    def log_telemetry_row(
        self,
        env_id: int,
        info: Dict[str, Any],
        action_np: Any,
        raw_reward: float,
        stored_reward: float,
        curriculum_factor: float,
        step_in_ep: int,
        speed_kmh: float,
        done: bool,
        hardware: Dict[str, Any],
        extra: Dict[str, Any] = None
    ) -> None:
        """Append one fully-populated telemetry row for a single environment step."""
        row = {
            "global_step": self.global_step, "env_id": env_id, "episode": self.episode_count,
            "step_in_ep": step_in_ep, "speed_kmh": round(float(speed_kmh), 2),
            "action_throttle": round(float(action_np[env_id, 0]), 3),
            "action_steer": round(float(action_np[env_id, 1]), 3),
            "action_brake": round(float(action_np[env_id, 2]), 3),
            "raw_reward": round(float(raw_reward), 4),
            "normalized_reward": round(float(stored_reward), 4),
            "curriculum_alpha": round(float(curriculum_factor), 2),
            **{k: round(float(info.get(k, 0.0)), 3) for k in self.REWARD_FIELDS},
            "heading_cos": round(float(info.get("heading_cos", 1.0)), 3),
            "sps": round(self.last_sps, 1) if self.last_sps else "",
            "fps": round(self.last_fps, 1) if self.last_fps else "",
            **hardware,
            "is_collision": info.get("is_collision", False),
            "is_off_road": info.get("is_off_road", False),
            "termination_reason": info.get("termination_reason", "") if done else ""
        }
        if extra:
            row.update(extra)
        self.csv_logger.log_step(row)

    def report_progress(self, suffix: str = "") -> None:
        """Print throughput once at least a full reporting interval of steps has elapsed."""
        if (self.global_step - self.last_progress_step) < max(20, self.num_envs * 10):
            return
        t_delta = max(1e-5, time.time() - (self.last_progress_time or time.time()))
        self.last_sps = (self.global_step - self.last_progress_step) / t_delta
        self.last_fps = self.last_sps * self.cfg.frame_skip
        pct = min(100.0, 100.0 * self.global_step / float(self.cfg.total_steps))
        print(f"[{time.strftime('%H:%M:%S')} | Step {self.global_step:05d}/{self.cfg.total_steps} "
              f"({pct:4.1f}%) | {self.last_sps:4.1f} Steps/s ({self.last_fps:4.1f} FPS) | "
              f"Episodes: {self.episode_count}{suffix}]", flush=True)
        self.last_progress_step = self.global_step
        self.last_progress_time = time.time()
