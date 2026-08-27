"""Evaluation Studio HUD rendering and visual layout utilities."""
import cv2
import numpy as np


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
    status_str: str = "Active Driving"
) -> None:
    """Draw clean telemetry panel showing traffic lights, speed kinematics, and policy controls."""
    # Dark card background in BGR
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (22, 27, 34), -1)
    cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + panel_w, y_offset + panel_h), (48, 54, 61), 2)

    # 1. Section Header: Telemetry & Traffic State
    cv2.putText(canvas, "REAL-TIME TELEMETRY & POLICY CONTROLS", (x_offset + 25, y_offset + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (x_offset + 25, y_offset + 40), (x_offset + panel_w - 25, y_offset + 40), (65, 75, 90), 1)

    # Traffic Light Status Badge
    badge_y = y_offset + 60
    if is_at_red_light:
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (30, 41, 59), -1)
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (68, 68, 239), 2)
        cv2.circle(canvas, (x_offset + 45, badge_y + 17), 7, (68, 68, 239), -1)
        cv2.putText(canvas, "TRAFFIC LIGHT: RED (Legal Stop Active)", (x_offset + 65, badge_y + 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (68, 68, 239), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (30, 41, 59), -1)
        cv2.rectangle(canvas, (x_offset + 25, badge_y), (x_offset + panel_w - 25, badge_y + 35), (34, 197, 94), 2)
        cv2.circle(canvas, (x_offset + 45, badge_y + 17), 7, (34, 197, 94), -1)
        cv2.putText(canvas, "TRAFFIC LIGHT: GREEN / OPEN ROAD", (x_offset + 65, badge_y + 23),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (34, 197, 94), 1, cv2.LINE_AA)

    # 2. Left Column: Speed Kinematics
    col1_x = x_offset + 25
    col1_y = badge_y + 55
    cv2.putText(canvas, "Kinematics State:", (col1_x, col1_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, f"{speed_kmh:4.1f} km/h", (col1_x, col1_y + 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Normalized: {speed_kmh/50.0:.2f}", (col1_x + 5, col1_y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"NPC: {num_vehicles} Cars | {num_walkers} Walkers", (col1_x, col1_y + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (246, 130, 59), 1, cv2.LINE_AA)
    rew_color = (11, 158, 245) if ep_reward >= 0 else (68, 68, 239)
    cv2.putText(canvas, f"Ep #{episode_count} | Reward: {ep_reward:+.1f}", (col1_x, col1_y + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, rew_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Avg Speed: {ep_avg_speed:.1f} km/h", (col1_x, col1_y + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)

    # Divider
    mid_x = x_offset + 300
    cv2.line(canvas, (mid_x, badge_y + 50), (mid_x, y_offset + panel_h - 25), (55, 65, 81), 1)

    # 3. Right Column: Policy Controls
    col2_x = mid_x + 25
    cv2.putText(canvas, f"Model Output ({backbone_name.upper()}):", (col2_x, col1_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (209, 213, 219), 1, cv2.LINE_AA)

    bar_w, bar_h, bar_x = 200, 16, col2_x + 90

    # Throttle
    t_y = col1_y + 20
    cv2.putText(canvas, "Throttle:", (col2_x, t_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + bar_w, t_y + bar_h), (55, 65, 81), -1)
    fill_t = int(bar_w * np.clip(throttle, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, t_y), (bar_x + fill_t, t_y + bar_h), (34, 197, 94), -1)
    cv2.putText(canvas, f"{throttle*100:3.0f}%", (bar_x + bar_w + 10, t_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1, cv2.LINE_AA)

    # Steer
    s_y = t_y + 35
    cv2.putText(canvas, "Steer:", (col2_x, s_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, s_y), (bar_x + bar_w, s_y + bar_h), (55, 65, 81), -1)
    center_x = bar_x + bar_w // 2
    steer_px = int((bar_w // 2) * np.clip(steer, -1.0, 1.0))
    if steer_px >= 0:
        cv2.rectangle(canvas, (center_x, s_y), (center_x + steer_px, s_y + bar_h), (246, 130, 59), -1)
    else:
        cv2.rectangle(canvas, (center_x + steer_px, s_y), (center_x, s_y + bar_h), (22, 115, 249), -1)
    cv2.line(canvas, (center_x, s_y - 2), (center_x, s_y + bar_h + 2), (255, 255, 255), 2)
    cv2.putText(canvas, f"{steer:+.2f}", (bar_x + bar_w + 10, s_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (246, 130, 59), 1, cv2.LINE_AA)

    # Brake
    b_y = s_y + 35
    cv2.putText(canvas, "Brake:", (col2_x, b_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (209, 213, 219), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + bar_w, b_y + bar_h), (55, 65, 81), -1)
    fill_b = int(bar_w * np.clip(brake, 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, b_y), (bar_x + fill_b, b_y + bar_h), (68, 68, 239), -1)
    cv2.putText(canvas, f"{brake*100:3.0f}%", (bar_x + bar_w + 10, b_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (68, 68, 239), 1, cv2.LINE_AA)

    # Footer
    foot_y = y_offset + panel_h - 20
    cv2.line(canvas, (x_offset + 25, foot_y - 12), (x_offset + panel_w - 25, foot_y - 12), (65, 75, 90), 1)
    cv2.putText(canvas, f"Frame: {step:04d}/{total_steps} (20 FPS) | Status: {status_str}",
                (x_offset + 25, foot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (156, 163, 175), 1, cv2.LINE_AA)
