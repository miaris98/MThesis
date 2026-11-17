"""Where were the objects we hit? Bench2Drive-220 Epic run 2026-11-05 (E15, vf_ctrl, vf_clean, vf_path), A44 collision clips.

For every vehicle / walker collision event (clip metadata, recorded at the first contact tick):
  zone      front / side / rear: the other actor's centre in our frame (rel_fwd > +2.5 m front, < -2.5 m rear, else side)
  vis_now   is any corner of the other actor's footprint inside our front camera's horizontal FOV (110 deg, camera 1.5 m behind the ego
            origin) at impact
  vis_1s / vis_2s  same, 1 s / 2 s before impact, both cars rolled back along their velocity at impact (constant-velocity approximation)
  mover     who closed the gap: ego (ego_closing > 0.5 m/s and >= other's), other, both, none (contact at ~zero closing speed)
Only clips of the record that counts for each route (summary/per_route_<arm>.csv label) are used, latest run per (label, route).
"""
import collections
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

CLIPS = sys.argv[1]
SUMMARY = sys.argv[2]
OUT = sys.argv[3]
ARMS = {"E15": "per_route_E15_epic.csv", "vf_ctrl": "per_route_baseline_vf_ctrl.csv", "vf_clean": "per_route_fix2_vf_clean.csv",
        "vf_path": "per_route_fix4_vf_path.csv"}
HALF_FOV = math.radians(55.0)
CAM_X = -1.5
EXT = {"vehicle": (2.4, 1.0), "walker": (0.3, 0.3)}

route_label = {}
scen = {}
for arm, f in ARMS.items():
    for r in csv.DictReader(open(os.path.join(SUMMARY, f))):
        route_label[(arm, r["route_id"])] = r["label"]
        scen[r["route_id"]] = r["scenario"].rsplit("_", 1)[0]


def arm_of(label):
    return "E15" if label.startswith("e15") else label.rsplit("_", 1)[0]


def visible(fwd, right, yaw_rel, ext):
    """any footprint corner of the other actor (centre fwd,right in our frame, heading yaw_rel) inside the camera's horizontal FOV"""
    ex, ey = ext
    c, s = math.cos(yaw_rel), math.sin(yaw_rel)
    for dx, dy in ((ex, ey), (ex, -ey), (-ex, ey), (-ex, -ey), (0, 0)):
        f = fwd + c * dx - s * dy - CAM_X
        r = right + s * dx + c * dy
        if f > 0.5 and abs(math.atan2(r, f)) < HALF_FOV:
            return True
    return False


# group clip files by (label, route, run timestamp); keep the latest run per (label, route)
runs = collections.defaultdict(lambda: collections.defaultdict(list))
pat = re.compile(r"RouteScenario_(\d+)_rep\d+_.*?_(\d\d_\d\d_\d\d_\d\d_\d\d)_h\d+_(vehicle|walker|layout|stuck)_t\d+\.npz$")
for p in glob.glob(os.path.join(CLIPS, "*", "clips", "*", "*.npz")):
    label = os.path.basename(os.path.dirname(p))
    m = pat.search(os.path.basename(p))
    if not m:
        continue
    rid, stamp, kind = m.groups()
    runs[(label, rid)][stamp].append((kind, p))

rows = []
for (label, rid), by_stamp in runs.items():
    arm = arm_of(label)
    if route_label.get((arm, rid)) != label:
        continue                                    # not the record that counts for this route
    stamp = max(by_stamp)
    for kind, p in by_stamp[stamp]:
        z = np.load(p, allow_pickle=True)
        e = json.loads(str(z["event"]))
        row = {"arm": arm, "route": rid, "scenario": scen.get(rid, "?"), "kind": kind, "file": os.path.basename(p)}
        if kind in ("vehicle", "walker"):
            fwd, right = float(e["other_rel_fwd"]), float(e["other_rel_right"])
            ex, ey, eyaw = e["ego_xy"]
            other = next((a for a in e.get("actors", []) if a["id"] == e["other_id"]), None)
            oyaw = math.radians(other["xy"][2]) if other else math.radians(eyaw)
            yaw_rel = oyaw - math.radians(eyaw)
            # velocities in the ego frame (ego moves +fwd; other along its heading); CARLA's y is "right" in the ego frame with yaw in degrees cw
            ve = np.array([float(e["ego_speed"]), 0.0])
            vo = float(e["other_speed"]) * np.array([math.cos(yaw_rel), math.sin(yaw_rel)])
            rel = np.array([fwd, right])
            u = rel / max(np.linalg.norm(rel), 1e-3)
            ego_cl, oth_cl = float(ve @ u), float(-(vo @ u))
            if kind == "walker":
                zone = "front" if fwd > 1.5 else ("rear" if fwd < -1.5 else "side")
            else:
                zone = "front" if fwd > 2.5 else ("rear" if fwd < -2.5 else "side")
            past = {}
            for dt in (1.0, 2.0):
                r = rel - (vo - ve) * dt         # relative position dt seconds earlier (constant velocities)
                past[dt] = visible(r[0], r[1], yaw_rel, EXT[kind])
            mover = ("both" if ego_cl > 0.5 and oth_cl > 0.5 else "ego" if ego_cl > 0.5 and ego_cl >= oth_cl else
                     "other" if oth_cl > 0.5 else "none")
            row.update(fwd=round(fwd, 2), right=round(right, 2), ego_speed=round(float(e["ego_speed"]), 2),
                       other_speed=round(float(e["other_speed"]), 2), ego_closing=round(ego_cl, 2), other_closing=round(oth_cl, 2),
                       zone=zone, vis_now=visible(fwd, right, yaw_rel, EXT[kind]), vis_1s=past[1.0], vis_2s=past[2.0], mover=mover,
                       other_type=e.get("other_type", ""), other_id=e.get("other_id"))
        rows.append(row)

with open(OUT, "w", newline="") as f:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)

col = [r for r in rows if r["kind"] in ("vehicle", "walker")]
print(f"events: {len(rows)} (vehicle {sum(r['kind']=='vehicle' for r in rows)}, walker {sum(r['kind']=='walker' for r in rows)}, "
      f"layout {sum(r['kind']=='layout' for r in rows)}, stuck {sum(r['kind']=='stuck' for r in rows)})")


def table(sel, title):
    n = len(sel)
    print(f"\n== {title}: {n} collisions")
    if not n:
        return
    for key in ("zone", "mover"):
        c = collections.Counter(r[key] for r in sel)
        print(f"  {key:6s} " + "  ".join(f"{k} {v} ({100*v/n:.0f}%)" for k, v in c.most_common()))
    seen = sum(r["vis_now"] or r["vis_1s"] or r["vis_2s"] for r in sel)
    print(f"  visible to the front camera at impact {sum(r['vis_now'] for r in sel)}, at impact or 1-2 s before {seen} ({100*seen/n:.0f}%); "
          f"never visible {n-seen} ({100*(n-seen)/n:.0f}%)")
    cz = collections.Counter((r["zone"], r["mover"]) for r in sel)
    print("  zone x mover: " + ", ".join(f"{z}/{m} {v}" for (z, m), v in sorted(cz.items(), key=lambda kv: -kv[1])))


for arm in ARMS:
    table([r for r in col if r["arm"] == arm], f"{arm}")
table(col, "ALL ARMS")
