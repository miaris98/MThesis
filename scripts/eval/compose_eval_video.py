#!/usr/bin/env python3
"""Compose a Bench2Drive route recorded with B2D_VIDEO_DIR into an mp4 with a HUD on top (2026-11-05).

Input folder (written by scripts/eval/bench2drive_agent.py):
    NNNNNN.jpg   front camera frame, exactly as the policy received it (RGB, native size)
    ticks.jsonl  one line per tick: speed, command, route points (ego frame), predicted waypoints, target speed, control

Output: an mp4 with a 240 px HUD band above the camera frame:
    left    steer, throttle, brake bars and the numbers behind them, speed and target speed, command
    middle  bird's-eye view: route points the policy was given (blue), waypoints it predicted (red), ego at the bottom
    right   route tag (town and scenario), tick, time, ego pose

usage: compose_eval_video.py ROUTE_DIR OUT.mp4 [--every 2] [--fps 10]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

W, H_CAM, H_HUD = 1280, 720, 256
BG = (24, 24, 24)
TXT = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bar(img, x, y, w, h, frac, color, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(1.0, frac)))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(img, label, (x, y - 6), FONT, 0.55, TXT, 1, cv2.LINE_AA)


def _steer(img, x, y, w, h, steer):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    mid = x + w // 2
    px = int(mid + max(-1.0, min(1.0, steer)) * (w // 2))
    cv2.rectangle(img, (min(mid, px), y), (max(mid, px), y + h), (0, 200, 255), -1)
    cv2.line(img, (mid, y), (mid, y + h), (200, 200, 200), 1)
    cv2.putText(img, f"steer {steer:+.2f}", (x, y - 6), FONT, 0.55, TXT, 1, cv2.LINE_AA)


def _bev(img, x0, y0, size, route, wps, scale=6.0):
    """Ego frame: up is +x (forward), right is +y. Ego sits at the bottom centre."""
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (16, 16, 16), -1)
    cx, cy = x0 + size // 2, y0 + size - 22
    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    def px(p):
        return int(cx + float(p[1]) * scale), int(cy - float(p[0]) * scale)

    for pts, color in ((route, (255, 160, 60)), (wps, (60, 60, 255))):
        if pts is None:
            continue
        pts = [px(p) for p in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, color, -1)
    cv2.putText(img, "route in (blue)  waypoints out (red)", (x0 + 6, y0 + 16), FONT, 0.45, TXT, 1, cv2.LINE_AA)


def _text(img, x, y, lines, step=24):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + step * i), FONT, 0.55, TXT, 1, cv2.LINE_AA)


def _scenario(tag):
    """'RouteScenario_1825_rep0_Town12_ConstructionObstacleTwoWays_1_25_11_05_...' -> 'Town12  ConstructionObstacleTwoWays_1'."""
    tok = tag.split("_")
    for i, t in enumerate(tok):
        if t.startswith("Town") and i + 2 < len(tok):
            return f"{t}  {tok[i + 1]}_{tok[i + 2]}"
    return tag[:40]


def _hud(row, cam_note):
    hud = np.full((H_HUD, W, 3), BG, np.uint8)
    thr, brk, steer = row["throttle"], row["brake"], row["steer"]
    spd_kmh = row["speed_mps"] * 3.6
    tgt = row.get("target_kmh")

    _steer(hud, 20, 40, 380, 16, steer)
    _bar(hud, 20, 92, 380, 16, thr, (80, 220, 80), f"throttle {thr:.2f}")
    _bar(hud, 20, 144, 380, 16, brk, (60, 60, 230), f"brake {brk:.2f}")
    _text(hud, 20, 188, [
        f"speed {spd_kmh:5.1f} km/h",
        f"target {tgt:5.1f} km/h" if tgt is not None else "target     n/a",
        f"command {row['command']}",
    ])

    route = np.asarray(row["route_ego"], dtype=float) if row.get("route_ego") is not None else None
    wps = np.asarray(row["waypoints_pred"], dtype=float) if row.get("waypoints_pred") is not None else None
    _bev(hud, 430, 8, 224, route, wps)

    x, y, yaw = row["ego_xy"]
    _text(hud, 690, 40, [
        f"route  {_scenario(row['tag'])}",
        f"tick {row['n']}   t {row['t']:.1f} s",
        f"ego x {x:.1f}  y {y:.1f}  yaw {yaw:.0f}",
        f"act  steer {steer:+.2f}  thr {thr:.2f}  brk {brk:.2f}",
    ])
    cv2.putText(hud, cam_note, (690, 246), FONT, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return hud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route_dir")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2, help="use every Nth tick")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.route_dir, "ticks.jsonl")) if l.strip()]
    rows = rows[:: max(1, a.every)]
    if not rows:
        raise SystemExit("no ticks")
    first = cv2.imread(os.path.join(a.route_dir, f"{rows[0]['n']:06d}.jpg"))
    if first is None:
        raise SystemExit("missing frame")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H_HUD + H_CAM))
    note = "camera: the policy takes this frame at 288x768 (crop 0); shown here at 1280x720"
    for row in rows:
        img = cv2.imread(os.path.join(a.route_dir, f"{row['n']:06d}.jpg"))
        if img is None:
            continue
        cam = cv2.resize(img, (W, H_CAM), interpolation=cv2.INTER_AREA)
        writer.write(np.vstack([_hud(row, note), cam]))
    writer.release()
    print(f"wrote {a.out} ({len(rows)} frames at {a.fps} fps)")


if __name__ == "__main__":
    main()
