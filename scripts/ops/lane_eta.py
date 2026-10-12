#!/usr/bin/env python3
"""Per-lane ETA for a CARLA evaluation box. usage: lane_eta.py w_xxx.txt
remaining routes of a lane = its list - driven (any restart segment) - excluded (a9_exclude.txt);
minutes per route = median gap between successive route start times (timestamp inside save_name) of that lane over its last 12 routes;
box ETA = slowest lane. Prints one summary line: LANE_ETA total=.. done=.. excluded=.. left=.. eta_min=.. worst_lane=.."""
import glob, json, re, sys, datetime, statistics, os
os.chdir("/workspace")
lanes = []
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) < 4 or not p[3].startswith("CKE:"):
        continue
    f = p[3].split(":")
    lanes.append((f[1], p[1], {int(x) for x in f[3].split(",")}))
excl = set()
if os.path.exists("a9_exclude.txt"):
    excl = {int(x) for x in open("a9_exclude.txt").read().split() if x.isdigit()}
now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
tot = done_n = ex_n = left_n = 0
worst = (0.0, "-"); rows = []
for label, port, routes in lanes:
    done, starts = set(), []
    for fn in glob.glob(f"bench2drive_out/{label}*_x.json"):
        if "abandoned" in fn or not re.fullmatch(rf"{re.escape(label)}r*_x\.json", os.path.basename(fn)):
            continue
        try:
            recs = json.load(open(fn))["_checkpoint"]["records"]
        except Exception:
            continue
        for r in recs:
            m = re.search(r"RouteScenario_(\d+)", r["route_id"])
            if m: done.add(int(m.group(1)))
            t = re.search(r"(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)$", r["save_name"])
            if t:
                mo, d, h, mi, s = map(int, t.groups())
                starts.append(datetime.datetime(2026, mo, d, h, mi, s))
    starts.sort()
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(starts, starts[1:]) if (b - a).total_seconds() < 3600]
    per = statistics.median(gaps[-12:]) if gaps else 12.0
    lane_done = routes & done; lane_ex = (routes - done) & excl; left = routes - done - excl
    tot += len(routes); done_n += len(lane_done); ex_n += len(lane_ex); left_n += len(left)
    last = max(starts) if starts else None
    # time since the last route started counts toward the route in progress
    eta = max(0.0, len(left) * per - ((now - last).total_seconds() / 60 if last and left else 0))
    rows.append((eta, port, len(left), per))
    if eta > worst[0]: worst = (eta, port)
rows.sort(reverse=True)
print(f"LANE_ETA total={tot} done={done_n} excluded={ex_n} left={left_n} eta_min={worst[0]:.0f} worst_lane={worst[1]}  "
      + " ".join(f"{p}:{l}left@{pp:.0f}m" for e, p, l, pp in rows[:3]))
