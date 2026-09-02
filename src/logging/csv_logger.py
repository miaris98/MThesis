"""Step-by-step CSV telemetry recorder for offline analysis and trajectory evaluation."""
import os
import csv
import time
from typing import Dict, Any, List, Optional


class CSVTelemetryLogger:
    """
    Step-by-step CSV telemetry recorder.
    Logs inputs, actions, rewards, sub-rewards, optimizer diagnostics, hardware metrics,
    and episode outcomes.

    One schema serves both PPO and SAC so their runs stay directly comparable in MLflow and
    offline analysis; each fills the columns its algorithm actually produces and leaves the
    rest blank. See TelemetryMixin.log_telemetry_row for the single writer.
    """

    #: Identity of the row.
    IDENTITY_FIELDS = ("global_step", "wall_time_s", "env_id", "episode", "step_in_ep")
    #: What the policy did and how fast the ego was going.
    ACTION_FIELDS = ("speed_kmh", "action_throttle", "action_steer", "action_brake")
    #: Reward, before and after normalization, plus its decomposition.
    REWARD_FIELDS = ("raw_reward", "normalized_reward", "curriculum_alpha", "reward_norm_std",
                     "r_progress", "r_lane", "r_light", "r_obstacle", "r_ttc", "r_terminal")
    #: Environment state the reward was computed from. All of these are already assembled by
    #: DrivingStateExtractor and passed through info; they are logged so a reward's behavior
    #: can be reconstructed offline without re-running the sim.
    STATE_FIELDS = ("lateral_dist", "heading_cos", "lane_width", "is_junction",
                    "is_at_red_light", "min_obs_dist", "ttc_seconds", "cost")
    #: Optimizer diagnostics. Constant across every row of a rollout (they describe the last
    #: update), which is what makes them joinable against the steps that produced them.
    OPTIM_FIELDS = ("loss_policy", "loss_value", "loss_entropy", "loss_total",
                    "loss_approx_kl", "loss_clip_fraction", "loss_explained_variance",
                    "grad_norm", "learning_rate",
                    "adv_mean", "adv_std", "return_mean", "return_std", "value_mean", "value_std",
                    "action_std_throttle", "action_std_steer", "action_std_brake",
                    "update_ms")
    #: SAC-only entropy temperature.
    SAC_FIELDS = ("sac_alpha", "sac_alpha_loss")
    #: Throughput and machine load.
    PERF_FIELDS = ("sps", "fps", "gpu_mem_used_mb", "gpu_mem_pct", "sys_cpu_pct", "sys_ram_used_gb")
    #: Termination flags, and episode aggregates written only on the row where done is True.
    OUTCOME_FIELDS = ("is_collision", "is_off_road", "is_stalled", "termination_reason",
                      "episode_return", "episode_length", "episode_avg_speed", "moving_avg_reward")

    FIELDNAMES: List[str] = list(
        IDENTITY_FIELDS + ACTION_FIELDS + REWARD_FIELDS + STATE_FIELDS
        + OPTIM_FIELDS + SAC_FIELDS + PERF_FIELDS + OUTCOME_FIELDS
    )

    def __init__(self, filepath: str, fieldnames: Optional[List[str]] = None):
        self.filepath = filepath
        self.fieldnames = list(fieldnames) if fieldnames else list(self.FIELDNAMES)
        self.rotated_from: Optional[str] = None
        self._warned = False
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # A file written under an older schema must not be appended to. DictWriter emits
        # values in the CURRENT fieldnames order regardless of what header is already on
        # disk, so every column after the first inserted field lands under the wrong name -
        # silently, and forever. Rotate the stale file aside and start a clean one.
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            if self._read_header(filepath) != self.fieldnames:
                self.rotated_from = self._rotate(filepath)

        file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        self.file = open(filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file, fieldnames=self.fieldnames, extrasaction="ignore", restval=""
        )
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    @staticmethod
    def _read_header(filepath: str) -> List[str]:
        """Return the column names already on disk, or [] if unreadable."""
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                return next(csv.reader(f), [])
        except Exception:
            return []

    @staticmethod
    def _rotate(filepath: str) -> Optional[str]:
        """Move a stale-schema CSV aside so a fresh one can be written. Returns the new path."""
        base, ext = os.path.splitext(filepath)
        target = f"{base}.{time.strftime('%Y%m%d-%H%M%S')}.old{ext or '.csv'}"
        try:
            os.replace(filepath, target)
            print(f"[CSVTelemetryLogger] Existing {os.path.basename(filepath)} was written under a "
                  f"different schema; moved to {os.path.basename(target)} and started a fresh file.",
                  flush=True)
            return target
        except Exception:
            return None

    def log_step(self, row_dict: Dict[str, Any]) -> None:
        """Append a single step dictionary to the telemetry CSV file."""
        try:
            self.writer.writerow(row_dict)
        except Exception as exc:
            # Report the first failure only. Silently dropping every row is how a schema
            # mismatch stayed invisible until someone opened the CSV and found gaps.
            if not self._warned:
                self._warned = True
                print(f"[CSVTelemetryLogger] Dropping telemetry rows: {exc}", flush=True)

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
