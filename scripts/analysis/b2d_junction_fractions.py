#!/usr/bin/env python3
"""Per-route junction completion fraction for Bench2Drive's Traffic_Signs ability, without a CARLA server.

`Bench2Drive/tools/ability_benchmark.py` scores a Traffic_Signs route as a success when the route completion passes
the first junction waypoint (+8 waypoints, "to ensure the ego pass the trigger volume") with no stop or red-light
infraction. It finds that waypoint by tracing the route's keypoints with GlobalRoutePlanner(map, 1.0) on a running
server. The map is the town's OpenDRIVE file, so this builds the same `carla.Map` client side from the .xodr that
ships with CARLA and runs the same trace. Output: JSON route id -> fraction, the `--junctions` input of a9_abilities.py.

Runs with the CARLA venv on an eval box (needs the carla module and PythonAPI/carla/agents), one process per town:

    /workspace/venv_carla/bin/python scripts/analysis/b2d_junction_fractions.py --carla-root /workspace/carla \
        --out /workspace/b2d_junctions.json
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GARAGE = ROOT / "Carla-utils/carla_garage"
if not GARAGE.exists():  # eval boxes keep carla_garage next to the repo
    GARAGE = Path(os.environ.get("GARAGE", "/workspace/carla_garage"))
XML = GARAGE / "Bench2Drive/leaderboard/data/bench2drive220.xml"
TOOL = GARAGE / "Bench2Drive/tools/ability_benchmark.py"


def traffic_sign_scenarios() -> set:
    tree = ast.parse(TOOL.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "Ability" for t in node.targets):
            return set(ast.literal_eval(node.value)["Traffic_Signs"])
    raise RuntimeError(f"no Ability table in {TOOL}")


def town_fractions(job):
    carla_root, town, routes = job  # routes: [(id, [(x, y, z), ...])]
    sys.path.insert(0, os.path.join(carla_root, "PythonAPI/carla"))
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    xodr = [p for p in glob.glob(os.path.join(carla_root, "CarlaUE4/Content/Carla/Maps/**/*.xodr"), recursive=True)
            if os.path.basename(p) == f"{town}.xodr"]
    if not xodr:
        return town, {}, f"no {town}.xodr under {carla_root}"
    grp = GlobalRoutePlanner(carla.Map(town, open(xodr[0]).read()), 1.0)
    out = {}
    for rid, pts in routes:
        locs = [carla.Location(*p) for p in pts]
        wps = [wp for a, b in zip(locs, locs[1:]) for wp, _ in grp.trace_route(a, b)]
        count = 0
        for wp in wps:  # same loop as the official tool: count includes the first junction waypoint
            count += 1
            if wp.is_junction:
                break
        out[rid] = (count + 8) / len(wps) if wps and wps[count - 1].is_junction else None
    return town, out, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "/workspace/carla"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ts = traffic_sign_scenarios()
    by_town = {}
    for r in ET.parse(XML).getroot().findall("route"):
        scen = r.find("scenarios").find("scenario").get("type")
        if scen not in ts:
            continue
        pts = [(float(p.get("x")), float(p.get("y")), float(p.get("z"))) for p in r.find("waypoints").findall("position")]
        by_town.setdefault(r.get("town"), []).append((int(r.get("id")), pts))
    print(f"{sum(map(len, by_town.values()))} Traffic_Signs routes in {len(by_town)} towns", flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=len(by_town)) as ex:
        for town, out, err in ex.map(town_fractions, [(a.carla_root, t, rs) for t, rs in sorted(by_town.items())]):
            print(f"{town}: {err or f'{len(out)} routes'}", flush=True)
            res.update(out)
    nojunc = sorted(r for r, v in res.items() if v is None)
    if nojunc:
        print("routes without a junction waypoint (the official tool raises here):", nojunc)
    json.dump({str(k): v for k, v in sorted(res.items()) if v is not None}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
