"""TODO A74 (2026-11-02): training routes for the maneuver RL, built from PDM-Lite's data-collection routes.

carla_garage's data/50x38_Town12 and data/50x36_Town13 hold one route per XML, 50 per scenario type, in the Leaderboard 2.0 format; they are not
the 220 Bench2Drive evaluation routes (leaderboard/data/bench2drive_split). This writes evaluator-sized batch files (one evaluator process runs
one batch, which also resets the leaderboard's memory growth) with obstacle scenarios at --obstacle_share (E15's weakest family: 44 DS on the
46 obstacle routes against ~84 for TF++), cycling through the routes in a seeded order, and holds out --val_per_type routes per scenario type
in val.xml for the learning curve on routes the head never trained on. manifest.json lists every batch's source files.

    python scripts/training/build_a74_routes.py --src /workspace/carla_garage/data --out /workspace/a74/routes --batches 200 --per_batch 25
"""
import argparse
import glob
import json
import os
import random
import re
import xml.etree.ElementTree as ET

OBSTACLE = re.compile(r"Accident|Obstacle|HazardAtSideLane|OpensDoor|Construction|InvadingTurn|ParkedObstacle")   # = the labels' dense set


def _scan(src):
    """{town: {scenario type: [route xml paths]}} over the per-town folders (50x38_Town12, 50x36_Town13, ...)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(src, "50x*_Town*"))):
        town = d.rsplit("_", 1)[-1]
        for t in sorted(os.listdir(d)):
            files = sorted(glob.glob(os.path.join(d, t, "*.xml")))
            if files:
                out.setdefault(town, {}).setdefault(t, []).extend(files)
    return out


def _write(path, sources):
    root = ET.Element("routes")
    for i, f in enumerate(sources):
        r = ET.parse(f).getroot().find("route")
        r.set("id", str(i))
        root.append(r)
    ET.ElementTree(root).write(path)


def build(src, out, batches, per_batch, obstacle_share=0.5, val_per_type=3, seed=0):
    rng = random.Random(seed)
    by_town = _scan(src)
    if not by_town:
        raise SystemExit(f"no route folders under {src} (expected 50x*_Town*/<scenario type>/*.xml)")
    val, pools = [], {}
    for town, by_type in sorted(by_town.items()):                  # one town per batch: a Town12 <-> Town13 switch is the slowest route setup
        obst, other = [], []
        for t, files in sorted(by_type.items()):
            files = files[:]
            rng.shuffle(files)
            val += files[:val_per_type]
            (obst if OBSTACLE.search(t) else other).extend(files[val_per_type:])
        rng.shuffle(obst)
        rng.shuffle(other)
        pools[town] = {"obstacle": obst, "other": other, "io": 0, "ix": 0}
    os.makedirs(out, exist_ok=True)
    n_obst = int(round(per_batch * obstacle_share))
    towns = sorted(pools)
    man = {"src": src, "seed": seed, "per_batch": per_batch, "obstacle_share": obstacle_share, "val": val, "batches": [],
           "pool": {"obstacle": sum(len(p["obstacle"]) for p in pools.values()), "other": sum(len(p["other"]) for p in pools.values())}}
    for b in range(batches):
        p = pools[towns[b % len(towns)]]
        pick = []
        for _ in range(n_obst):
            pick.append(p["obstacle"][p["io"] % len(p["obstacle"])]); p["io"] += 1
        for _ in range(per_batch - n_obst):
            pick.append(p["other"][p["ix"] % len(p["other"])]); p["ix"] += 1
        rng.shuffle(pick)
        name = f"batch_{b:04d}.xml"
        _write(os.path.join(out, name), pick)
        man["batches"].append({"file": name, "sources": pick})
    _write(os.path.join(out, "val.xml"), val)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--per_batch", type=int, default=25)
    ap.add_argument("--obstacle_share", type=float, default=0.5)
    ap.add_argument("--val_per_type", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = build(a.src, a.out, a.batches, a.per_batch, a.obstacle_share, a.val_per_type, a.seed)
    print(f"A74 routes: {len(m['batches'])} batches x {a.per_batch} ({m['pool']}), {len(m['val'])} held out -> {a.out}")


if __name__ == "__main__":
    main()
