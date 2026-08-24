"""Video rendering and HUD overlay visualization engine for CARLA evaluations."""
from typing import Dict, Any, Optional
import numpy as np
import cv2


class VideoRenderer:
    """
    Renders evaluation video frames with telemetry cards, speed gauges, and policy control bars.
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 20):
        self.width = width
        self.height = height
        self.fps = fps

    @staticmethod
    def draw_hud(
        canvas: np.ndarray,
        x_offset: int,
        y_offset: int,
        panel_w: int,
        panel_h: int,
        speed_kmh: float,
        throttle: float,
        steer: float,
        brake: float,
        step: int,
        total_steps: int,
        episode_count: int,
        ep_reward: float,
        ep_avg_speed: float,
        backbone_name: str,
        is_at_red_light: bool = False,
        num_vehicles: int = 3,
        num_walkers: int = 10,
        status_str: str = "Active"
    ) -> None:
        """Draw HUD dashboard with telemetry cards and policy action bars."""
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (22, 27, 34), -1)
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (48, 54, 61), 2)

        # Header
        cv2.putText(canvas, "REAL-TIME TELEMETRY & POLICY CONTROLS", (x_offset + 25, y_offset + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (x_offset + 25, y_offset + 40), (x_offset + panel_w - 25, y_offset + 40), (65, 75, 90), 1)

        # Traffic light badge
        badge_y = y_offset + 60
        if is_at_red_light:
            cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (30, 41, 59), -1)
            cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (239, 68, 68), 2)
            cv2.circle(canvas, (x_offset + 45, badge_y + 17), 7, (239, 68, 68), -1)
            cv2.putText(canvas, "TRAFFIC LIGHT: RED (Legal Stop Active)", (x_offset + 65, badge_y + 23),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (239, 68, 68), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (30, 41, 59), -1)
            cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (34, 197, 94), 2)
            cv2.circle(canvas, (x_offset + 45, badge_y + 17), 7, (34, 197, 94), -1)
            cv2.putText(canvas, "TRAFFIC LIGHT: GREEN / OPEN ROAD", (x_offset + 65, badge_y + 23),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (34, 197, 94), 1, cv2.LINE_AA)

        # Speed Kinematics
        col1_x = x_offset + 25
        col1_y = badge_y + 55
        cv2.putText(canvas, "Kinematics:", (col1_x, col1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{speed_kmh:4.1f} km/h", (col1_x, col1_y + 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (16, 185, 129), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"NPCs: {num_vehicles} Cars | {num_walkers} Walkers", (col1_x, col1_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 130, 246), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Ep #{episode_count} | Reward: {ep_reward:+.1f}", (col1_x, col1_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 158, 11), 1, cv2.LINE_AA)

        # Controls Column
        mid_x = x_offset + 300
        cv2.line(canvas, (mid_x, badge_y + 50), (mid_x, y_offset + panel_h - 25), (55, 65, 81), 1)
        col2_x = mid_x + 25
        cv2.putText(canvas, f"Policy Controls ({backbone_name.upper()}):", (col2_x, col1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)

        bar_w = 180
        bar_h = 14
        bar_x = col2_x + 80

        # Throttle
        t_y = col1_y + 25
        cv2.putText(canvas, "Throttle:", (col2_x, t_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (209, 213, 219), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x, t_y), (bar_x + bar_w, t_y + bar_h), (55, 65, 81), -1)
        fill_t = int(bar_w * np.clip(throttle, 0.0, 1.0))
        cv2.rectangle(canvas, (bar_x, t_y), (bar_x + fill_t, t_y + bar_h), (34, 197, 94), -1)

        # Steer
        s_y = t_y + 30
        cv2.putText(canvas, "Steer:", (col2_x, s_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (209, 213, 219), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x, s_y), (bar_x + bar_w, s_y + bar_h), (55, 65, 81), -1)
        center_x = bar_x + bar_w // 2
        fill_s = int((bar_w // 2) * np.clip(steer, -1.0, 1.0))
        if fill_s >= 0:
            cv2.rectangle(canvas, (center_x, s_y), (center_x + fill_s, s_y + bar_h), (59, 130, 246), -1)
        else:
            cv2.rectangle(canvas, (center_x + fill_s, s_y), (center_x, s_y + bar_h), (59, 130, 246), -1)

        # Brake
        b_y = s_y + 30
        cv2.putText(canvas, "Brake:", (col2_x, b_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (209, 213, 219), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x, b_y), (bar_x + bar_w, b_y + bar_h), (55, 65, 81), -1)
        fill_b = int(bar_w * np.clip(brake, 0.0, 1.0))
        cv2.rectangle(canvas, (bar_x, b_y), (bar_x + fill_b, b_y + bar_h), (239, 68, 68), -1)
