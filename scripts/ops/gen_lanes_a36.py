"""2026-10-05 CARLA session lanes: the A36 weight-space merges (and any other checkpoint list) on the lead-vehicle + obstacle routes of Bench2Drive-220
(the ~100 routes where E / J differ most, A52 stage-1 screen), balanced over lanes by the A9 wall time of each route.

    py scripts/ops/gen_lanes_a36.py ROUTE_DUR.json OUT_FILE BOX:GPUS:LANES_PER_GPU LABEL=CKPT_DIR/FILE [LABEL=...]
e.g. py scripts/ops/gen_lanes_a36.py route_dur.json lanes_B1.txt B1:4:3 a36m50=a36m50/model_epoch_020.pth a36soup=a36soup/model_epoch_020.pth
One line per lane 'BOX PORT GPU JOBS' for launch_lanes_nokill.sh; jobs in the given order (priority arms first); each (arm, lane) job is a CKL job."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from a3_failure_classes import LEAD_FAMILIES, OBSTACLE_FAMILIES, scenario_table  # noqa: E402

dur_f, out_f, spec, *arms = sys.argv[1:]
dur = {int(k): v for k, v in json.load(open(dur_f)).items()}
crash = {27515, 25854, 25896, 25955, 24816}
tab = scenario_table()
routes = sorted(r for r, v in tab.items() if (v["family"] in LEAD_FAMILIES or v["family"] in OBSTACLE_FAMILIES) and r not in crash)
print(len(routes), "lead + obstacle routes")
name, ng, lpg = spec.split(":")
lanes = [{"gpu": g, "port": 2000 + 400 * (g * int(lpg) + k), "R": [], "load": 0.0} for g in range(int(ng)) for k in range(int(lpg))]
med = sorted(dur.values())[len(dur) // 2]
for r in sorted(routes, key=lambda r: -dur.get(r, med)):
    L = min(lanes, key=lambda x: x["load"])
    L["R"].append(r); L["load"] += (dur.get(r, med) + 150) * len(arms)
lines = []
for L in lanes:
    rs = ",".join(map(str, sorted(L["R"])))
    jobs = " ".join(f"CKL:{a.split('=')[0]}_n{name}_{L['port']}:{a.split('=')[1]}:{rs}" for a in arms)
    lines.append(f"{name} {L['port']} {L['gpu']} {jobs}")
    print(f"port {L['port']} gpu {L['gpu']}: {len(L['R'])} routes, est {L['load'] / 3600:.2f} h")
open(out_f, "w", newline="\n").write("\n".join(lines) + "\n")
