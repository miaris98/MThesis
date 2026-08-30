"""Step-by-step CSV telemetry recorder for offline analysis and trajectory evaluation."""
import os
import csv
from typing import Dict, Any


class CSVTelemetryLogger:
    """
    Step-by-step CSV telemetry recorder.
    Logs inputs, actions, rewards, sub-rewards, hardware metrics, and curriculum parameters.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.fieldnames = [
            "global_step", "env_id", "episode", "step_in_ep",
            "speed_kmh", "action_throttle", "action_steer", "action_brake",
            "raw_reward", "normalized_reward", "curriculum_alpha",
            "r_progress", "r_lane", "r_light", "r_obstacle", "r_ttc", "r_terminal",
            "lateral_dist", "heading_cos",
            "loss_policy", "loss_value", "loss_entropy", "loss_approx_kl", "loss_clip_fraction", "loss_explained_variance",
            "sps", "fps",
            "gpu_mem_used_mb", "gpu_mem_pct", "sys_cpu_pct", "sys_ram_used_gb",
            "is_collision", "is_off_road", "termination_reason"
        ]
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        file_exists = os.path.exists(filepath)
        self.file = open(filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log_step(self, row_dict: Dict[str, Any]) -> None:
        """Append a single step dictionary to the telemetry CSV file."""
        try:
            self.writer.writerow(row_dict)
        except Exception:
            pass

    def flush(self) -> None:
        """Flush the underlying file buffer to disk."""
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Safely flush and close the CSV telemetry file."""
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
