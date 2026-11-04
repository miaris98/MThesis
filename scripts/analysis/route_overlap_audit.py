"""Which PDM-Lite training routes sit on a Bench2Drive-220 evaluation route? (user, 2026-11-04: "make sure we dont train directly on the
evaluation routes"). The obstacle archives exist only in Town12/13, the towns of 151 of the 220 evaluation routes, so the training set of an
obstacle arm has to be filtered before the closed-loop numbers mean anything.

A training route (carla_garage `data/50x3x_Town1x/<scenario>/<id>_<k>.xml`: start and end point plus one scenario trigger) is a leak when, in
the same town, its scenario trigger lies within `trigger_radius` of an evaluation scenario trigger of the same type, or within `path_radius`
of an evaluation route's waypoint polyline (any type: the network would have seen that stretch of road with an event on it).

    python scripts/analysis/route_overlap_audit.py --eval_xml Carla-utils/carla_garage/leaderboard/data/bench2drive220.xml \
        --train_dirs Carla-utils/carla_garage/data/50x36_Town13 Carla-utils/carla_garage/data/50x38_Town12 --out_json leak_routes.json
"""
import argparse
import glob
import gzip
import json
import math
import os
import xml.etree.ElementTree as ET


def point_to_polyline(p, line):
    """Euclidean distance (m, x/y only) from p to the polyline through the points of `line`."""
    best = math.inf
    for (ax, ay), (bx, by) in zip(line[:-1], line[1:]) if len(line) > 1 else [(line[0], line[0])]:
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        best = min(best, math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)))
    return best


def _route(el):
    wps = [(float(q.get("x")), float(q.get("y"))) for q in el.iter("position")]
    scen = []
    for s in el.iter("scenario"):
        tp = s.find("trigger_point")
        if tp is not None:
            scen.append((s.get("type"), float(tp.get("x")), float(tp.get("y"))))
    return {"id": el.get("id"), "town": el.get("town"), "waypoints": wps, "scenarios": scen}


def load_eval_routes(path):
    return [_route(r) for r in ET.parse(path).getroot().iter("route")]


def load_training_route(path):
    r = _route(next(ET.parse(path).getroot().iter("route")))
    r["file"] = str(path)
    return r


def flag_training_route(tr, eval_routes, trigger_radius=30.0, path_radius=10.0):
    same_type, near_path, ids = False, False, []
    for ev in eval_routes:
        if ev["town"] != tr["town"]:
            continue
        hit = False
        for typ, x, y in tr["scenarios"]:
            if any(et == typ and math.hypot(x - ex, y - ey) <= trigger_radius for et, ex, ey in ev["scenarios"]):
                same_type = hit = True
            if ev["waypoints"] and point_to_polyline((x, y), ev["waypoints"]) <= path_radius:
                near_path = hit = True
        if hit:
            ids.append(ev["id"])
    return {"same_type_near_eval_trigger": same_type, "near_eval_path": near_path, "leak": same_type or near_path, "eval_ids": ids}


def flag_trajectory(town, positions, eval_routes, path_radius=10.0, min_overlap_m=30.0):
    """Stricter check on a downloaded route folder: the metres of the driven trajectory (`positions`, x/y per frame) that lie within
    `path_radius` of each evaluation route in the same town. A leak is >= `min_overlap_m` along one evaluation route; crossing an
    evaluation road at a junction shares only a few metres and is not one."""
    overlap = {}
    for ev in eval_routes:
        if ev["town"] != town or not ev["waypoints"]:
            continue
        m = 0.0
        for p, q in zip(positions[:-1], positions[1:]):
            if point_to_polyline(p, ev["waypoints"]) <= path_radius and point_to_polyline(q, ev["waypoints"]) <= path_radius:
                m += math.hypot(q[0] - p[0], q[1] - p[1])
        if m > 0:
            overlap[ev["id"]] = m
    ids = sorted(k for k, v in overlap.items() if v >= min_overlap_m)
    return {"leak": bool(ids), "eval_ids": ids, "overlap_m": overlap}


def read_positions(route_dir, stride=2):
    """Ego x/y per frame from a PDM-Lite route folder (`measurements/*.json.gz`, key `pos_global`), every `stride`-th frame."""
    files = sorted(glob.glob(os.path.join(route_dir, "measurements", "*.json.gz")))[::stride]
    out = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            pg = json.load(fh)["pos_global"]
        out.append((float(pg[0]), float(pg[1])))
    return out


def audit_data_dir(data_dir, ev, path_radius, min_overlap_m):
    """Every downloaded Town12/13 route folder (its name starts with the town) -> trajectory flags."""
    rows = []
    for m in sorted(glob.glob(os.path.join(data_dir, "**", "measurements"), recursive=True)):
        rd = os.path.dirname(m)
        town = os.path.basename(rd).split("_")[0]
        if town not in {r["town"] for r in ev}:
            continue
        rows.append({"dir": rd, "town": town, **flag_trajectory(town, read_positions(rd), ev, path_radius, min_overlap_m)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_xml", required=True)
    ap.add_argument("--train_dirs", nargs="*", default=[], help="carla_garage route-file folders (trigger-level check)")
    ap.add_argument("--data_dir", default=None, help="downloaded PDM-Lite data (trajectory-level check on every route folder)")
    ap.add_argument("--trigger_radius", type=float, default=30.0)
    ap.add_argument("--path_radius", type=float, default=10.0)
    ap.add_argument("--min_overlap_m", type=float, default=30.0)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    ev = load_eval_routes(a.eval_xml)
    if a.data_dir:
        rows = audit_data_dir(a.data_dir, ev, a.path_radius, a.min_overlap_m)
        leak = [r for r in rows if r["leak"]]
        print(f"{len(ev)} evaluation routes, {len(rows)} Town12/13 route folders, {len(leak)} leak "
              f"(>= {a.min_overlap_m} m within {a.path_radius} m of one evaluation route)")
        if a.out_json:
            json.dump({"path_radius": a.path_radius, "min_overlap_m": a.min_overlap_m, "routes": rows}, open(a.out_json, "w"), indent=1)
        return
    rows, by_type = [], {}
    for d in a.train_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*", "*.xml"))):
            tr = load_training_route(f)
            fl = flag_training_route(tr, ev, a.trigger_radius, a.path_radius)
            typ = os.path.basename(os.path.dirname(f))
            key = (tr["town"], typ)
            n, k = by_type.get(key, (0, 0))
            by_type[key] = (n + 1, k + int(fl["leak"]))
            rows.append({"file": os.path.relpath(f, os.path.dirname(d)), "town": tr["town"], "type": typ,
                         "start": tr["waypoints"][0] if tr["waypoints"] else None, **fl})
    n_leak = sum(r["leak"] for r in rows)
    print(f"{len(ev)} evaluation routes, {len(rows)} training routes, {n_leak} flagged "
          f"(trigger radius {a.trigger_radius} m, path radius {a.path_radius} m)")
    for (town, typ), (n, k) in sorted(by_type.items()):
        if k:
            print(f"  {town} {typ}: {k} / {n}")
    if a.out_json:
        json.dump({"trigger_radius": a.trigger_radius, "path_radius": a.path_radius, "routes": rows}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
