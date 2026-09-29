#!/usr/bin/env python3
"""How far does the flat-ground route overlay drift from the real road on slopes? (S-100)

`src/config/route_overlay.py` projects route points onto the ego's ground plane (z = 0). The car pitches with the
road under it, so the overlay's error at distance x ahead is the road's height deviation from the tangent plane at
the ego, projected: dv = f * dh / (x - cam_x), f from the camera's horizontal FOV. Uses the 2 m-spaced route
waypoints (with z) of a Leaderboard route file and reports the error per town group at 10/20/30 m.

On bench2drive220.xml (2026-09-28): p95 ~5 px at 30 m (one line-width at full resolution), Town12/13 no worse than
the training towns, so a slope-aware overlay was not needed. Re-run it for a new route set or camera.

    python scripts/analysis/route_overlay_slope_check.py [--xml path/to/routes.xml]
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config.camera import PDM_LITE_CAMERA  # noqa: E402

TRAIN_TOWNS = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"}
DEFAULT_XML = [Path(os.environ.get("GARAGE", "/workspace/carla_garage")) / "Bench2Drive/leaderboard/data/bench2drive220.xml",
               ROOT / "Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml"]


def town_group(town: str) -> str:
    if town in TRAIN_TOWNS:
        return "training towns"
    return "Town12/13" if town in ("Town12", "Town13") else "other unseen"


def overlay_errors(xml_text: str, f_px: float, cam_back_m: float, lookaheads=(10, 20, 30), grade_window_m=4.0):
    """-> (errors[group][x] = [px], grades[group] = [|grade|], routes[group] = n)."""
    errors = collections.defaultdict(lambda: collections.defaultdict(list))
    grades = collections.defaultdict(list)
    routes = collections.Counter()
    for m in re.finditer(r'<route id="(\d+)"[^>]*? town="(\w+)">(.*?)</route>', xml_text, re.S):
        _, town, body = m.groups()
        pts = np.array([[float(a), float(b), float(c)] for a, b, c in
                        re.findall(r'<position x="([-\d.]+)" y="([-\d.]+)" z="([-\d.]+)"', body)])
        if len(pts) < 10:
            continue
        grp = town_group(town)
        routes[grp] += 1
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
        z = pts[:, 2]
        for i in range(len(s)):
            w = (s >= s[i] - grade_window_m) & (s <= s[i] + grade_window_m)
            if w.sum() < 4:
                continue
            g = np.polyfit(s[w], z[w], 1)[0]  # local grade under the car (z is rounded to 0.1 m: fit, don't difference)
            grades[grp].append(abs(g))
            for x in lookaheads:
                j = np.searchsorted(s, s[i] + x)
                if j >= len(s):
                    continue
                dh = z[j] - z[i] - g * (s[j] - s[i])
                errors[grp][x].append(abs(f_px * dh / (s[j] - s[i] + cam_back_m)))
    return errors, grades, routes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, default=next((p for p in DEFAULT_XML if p.is_file()), None))
    args = ap.parse_args()
    if not args.xml or not args.xml.is_file():
        ap.error("route file not found; pass --xml")
    f_px = (PDM_LITE_CAMERA["width"] / 2) / math.tan(math.radians(PDM_LITE_CAMERA["fov"]) / 2)
    cam_back = -float(PDM_LITE_CAMERA["x"])  # camera sits behind the ego origin
    errors, grades, routes = overlay_errors(args.xml.read_text(), f_px, cam_back)
    print(f"{args.xml}\nf = {f_px:.1f} px at {PDM_LITE_CAMERA['width']} wide; routes with >= 10 waypoints: {dict(routes)}")
    for grp in ("training towns", "Town12/13", "other unseen"):
        if not grades[grp]:
            continue
        g = np.array(grades[grp]) * 100
        print(f"\n{grp}: |grade| median {np.median(g):.1f}%  p95 {np.percentile(g, 95):.1f}%  max {g.max():.1f}%")
        for x, e in sorted(errors[grp].items()):
            e = np.array(e)
            print(f"  {x:2d} m ahead: median {np.median(e):4.1f} px, p95 {np.percentile(e, 95):5.1f}, max {e.max():5.1f};"
                  f" >5 px {100 * (e > 5).mean():4.1f}% of positions (n={len(e)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
