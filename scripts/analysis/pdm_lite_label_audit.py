#!/usr/bin/env python3
"""Label audit on real PDM-Lite logs for TODO A21 / A37 / A38 / A42 / A45, read with HTTP range requests (no archive download).

For a few routes of one scenario archive on the Hub (autonomousvision/PDM_Lite_Carla_LB2) it reports

  * who sets the expert's speed (`speed_reduced_by_obj_type`): vehicle / walker / other, and at what distance;
  * whether that actor is inside the front camera's field of view (+-55 degrees, f = 358.5 px at 1024 px) and how many LiDAR points its box has
    (`num_points`, an occlusion proxy; the logs carry no camera visible-pixel count);
  * whether box ids persist from frame to frame (tracks), at the logged 4 Hz;
  * a 1-D counterfactual speed-safety label for in-lane lead vehicles: replace the ego's speed profile by a constant v_j for the next 3 s and
    re-integrate the gap to the lead actor from its *recorded* motion; the bin is unsafe when the gap falls below 0.5 m. Sanity check for A37:
    the bin nearest to the expert's own speed must be safe in (almost) every frame.

    py scripts/analysis/pdm_lite_label_audit.py Town12/data/HardBreakRoute.zip [--routes 6] [--every 1] [--out audit.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdm_lite_range_probe import BASE, HttpRangeFile  # noqa: E402

import zipfile  # noqa: E402

TARGET_SPEEDS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)  # TF++ bins, m/s
FOV_DEG, DT, HORIZON = 55.0, 0.25, 12                                          # 4 Hz frames, 3 s
A_DEC, A_ACC, HOLD = 7.0, 2.0, 3                                                        # m/s^2 braking / accelerating authority (PDM-Lite's comfortable braking is 3.7-8.7); the bin is held for HOLD frames
tl = threading.local()


def zipfile_for(url):
    if getattr(tl, "url", None) != url:
        tl.zf, tl.url = zipfile.ZipFile(HttpRangeFile(url)), url
    return tl.zf


def read_json(url, name):
    return json.loads(gzip.decompress(zipfile_for(url).read(name)))


def load_route(url, route, names_set, every):
    meas_names = sorted(n for n in names_set if n.startswith(route + "/measurements/") and n.endswith(".json.gz"))
    out = []
    for n in meas_names:
        b = n.replace("/measurements/", "/boxes/")
        out.append((n, b))
    return route, out


def fetch(url, pair):
    n, b = pair
    try:
        return read_json(url, n), read_json(url, b)
    except Exception as e:  # a frame without boxes
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path")
    ap.add_argument("--routes", type=int, default=6)
    ap.add_argument("--every", type=int, default=1, help="use every n-th logged frame (1 = all)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    url = BASE + a.zip_path
    zf = zipfile_for(url)
    names = zf.namelist()
    nameset = set(names)
    routes = sorted({m.group(1) for n in names if (m := re.match(r"(.+?/Route\d+_Rep\d+)/", n))})
    rnd = np.random.default_rng(0)
    pick = [routes[i] for i in sorted(rnd.choice(len(routes), size=min(a.routes, len(routes)), replace=False))]
    print(f"{a.zip_path}: {len(routes)} routes, auditing {len(pick)}", flush=True)

    stats = Counter()
    dists, npts_cause, gaps_unsafe = [], [], {v: [0, 0] for v in TARGET_SPEEDS}
    own_unsafe, own_floor, realised_gap = [0, 0], [0, 0], []
    id_persist = [0, 0]
    reduced_types = Counter()
    frames_per_route = []
    with ThreadPoolExecutor(a.threads) as ex:
        for route in pick:
            _, pairs = load_route(url, route, nameset, a.every)
            pairs = pairs[:: a.every]
            data = list(ex.map(lambda p: fetch(url, p), pairs))
            data = [d for d in data if d is not None]
            frames_per_route.append(len(data))
            for k, (m, boxes) in enumerate(data):
                stats["frames"] += 1
                stats["brake"] += bool(m.get("brake"))
                for h in ("vehicle_hazard", "walker_hazard", "light_hazard", "stop_sign_hazard"):
                    stats[h] += bool(m.get(h))
                stats["changed_route"] += bool(m.get("changed_route"))
                boxes = [b for b in boxes if "id" in b]     # the ego box carries no id
                byid = {b["id"]: b for b in boxes}
                if k + 1 < len(data):
                    nxt = {b["id"] for b in data[k + 1][1] if "id" in b}
                    id_persist[0] += sum(b["id"] in nxt for b in boxes)
                    id_persist[1] += len(boxes)
                typ = m.get("speed_reduced_by_obj_type")
                rid = m.get("speed_reduced_by_obj_id")
                if typ is None:
                    continue
                kind = "vehicle" if str(typ).startswith("vehicle") else "walker" if str(typ).startswith("walker") else str(typ).split(".")[0]
                reduced_types[kind] += 1
                stats["reduced"] += 1
                d = m.get("speed_reduced_by_obj_distance")
                if d is not None:
                    dists.append(float(d))
                b = byid.get(rid)
                if b is None:
                    stats["cause_no_box"] += 1
                    continue
                x, y = b["position"][0], b["position"][1]
                ang = math.degrees(math.atan2(y, x)) if x > 0 else 180.0
                stats["cause_in_fov"] += abs(ang) <= FOV_DEG
                stats["cause_behind_or_side"] += abs(ang) > FOV_DEG
                npts_cause.append(b.get("num_points", -1))
                # 1-D counterfactual for in-lane lead vehicles
                if kind == "vehicle" and 0 < x < 60 and abs(y) < 2.0 and k + HORIZON < len(data):
                    series = []
                    for s in range(1, HORIZON + 1):
                        bn = {bb["id"]: bb for bb in data[k + s][1] if "id" in bb}.get(rid)
                        if bn is None:
                            break
                        series.append((bn["position"][0] - bn["extent"][0] - 2.4, data[k + s][0]["speed"]))
                    if len(series) == HORIZON:
                        stats["lead_frames"] += 1
                        ego_v = [data[k + s][0]["speed"] for s in range(HORIZON)]

                        def min_gap(v_cmd):
                            """inevitable-collision criterion: the ego tracks v_cmd for HOLD frames (0.75 s), then brakes with A_DEC; the lead keeps its
                            recorded motion; returns the smallest bumper gap over the 3 s"""
                            v, gmin, shift = ego_v[0], 1e9, 0.0
                            for s, (g, _) in enumerate(series):
                                goal = v_cmd if s < HOLD else 0.0
                                v = v + float(np.clip(goal - v, -A_DEC * DT, A_ACC * DT))
                                shift += (ego_v[s] - v) * DT              # realised ego advance minus the counterfactual advance
                                gmin = min(gmin, g + shift)
                            return gmin
                        for v in TARGET_SPEEDS:
                            gaps_unsafe[v][0] += min_gap(v) < 0.5
                            gaps_unsafe[v][1] += 1
                        tgt = float(m.get("target_speed", m["speed"]))
                        vj = min(TARGET_SPEEDS, key=lambda t: abs(t - tgt))
                        vf = max([t for t in TARGET_SPEEDS if t <= tgt + 1e-6] or [0.0])     # the largest bin not above the commanded speed
                        own_unsafe[0] += min_gap(vj) < 0.5
                        own_floor[0] += min_gap(vf) < 0.5
                        own_unsafe[1] += 1
                        own_floor[1] += 1
                        realised_gap.append(min(g for g, _ in series))
    n = max(stats["frames"], 1)
    r = max(stats["reduced"], 1)
    res = {
        "archive": a.zip_path, "routes": len(pick), "frames": stats["frames"], "frames_per_route_mean": float(np.mean(frames_per_route)),
        "share_brake": stats["brake"] / n, "share_vehicle_hazard": stats["vehicle_hazard"] / n, "share_walker_hazard": stats["walker_hazard"] / n,
        "share_light_hazard": stats["light_hazard"] / n, "share_stop_sign_hazard": stats["stop_sign_hazard"] / n, "share_changed_route": stats["changed_route"] / n,
        "share_speed_reduced_by_actor": stats["reduced"] / n, "reduced_by": dict(reduced_types),
        "cause_distance_pct_<15_15-30_30-50_>50": [float(np.mean(np.array(dists) < 15)), float(np.mean((np.array(dists) >= 15) & (np.array(dists) < 30))),
                                                    float(np.mean((np.array(dists) >= 30) & (np.array(dists) < 50))), float(np.mean(np.array(dists) >= 50))] if dists else None,
        "cause_in_front_fov": stats["cause_in_fov"] / r, "cause_outside_fov": stats["cause_behind_or_side"] / r, "cause_without_box": stats["cause_no_box"] / r,
        "cause_num_points_<10": float(np.mean(np.array(npts_cause) < 10)) if npts_cause else None,
        "id_persists_to_next_frame": id_persist[0] / max(id_persist[1], 1),
        "lead_frames": stats["lead_frames"], "unsafe_share_by_bin": {f"{v:.1f}": gaps_unsafe[v][0] / max(gaps_unsafe[v][1], 1) for v in TARGET_SPEEDS},
        "expert_own_bin_unsafe_nearest": own_unsafe[0] / max(own_unsafe[1], 1), "expert_own_bin_unsafe_floor": own_floor[0] / max(own_floor[1], 1),
        "realised_min_gap_m_quantiles_5_25_50_75": [float(np.percentile(realised_gap, q)) for q in (5, 25, 50, 75)] if realised_gap else None,
        "realised_min_gap_below_0.5m_share": float(np.mean(np.array(realised_gap) < 0.5)) if realised_gap else None,
    }
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
