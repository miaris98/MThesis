"""2026-10-04 CARLA session lanes (TODO_ACTIVE 'Order of work' item 1): the original WoR x2 repeats on the 60 repeat routes and arm J e20
seeds 0-1 on the other Bench2Drive-220 routes, balanced over lanes by the wall time each route took in A9 (E15 / WoR records).

    py scripts/ops/gen_lanes_0410.py ROUTE_DUR.json ROUTES_OBST.txt ROUTES_B2D20.txt OUT_DIR N1 N2 [N3 ...]
N1.. = lanes per GPU group; writes lanes_<box>.txt, one line per lane 'BOX PORT GPU JOBS' for launch_lanes_nokill.sh. Boxes are given as
name:gpus:lanes_per_gpu, e.g. B1:4:3 B2:4:3. Job order per lane: WoR a, J s0, WoR b, J s1 (so the priority work finishes first)."""
import json, re, sys
from pathlib import Path

dur_f, obst_f, r19_f, out_dir, *boxes = sys.argv[1:]
dur = {int(k): v for k, v in json.load(open(dur_f)).items()}
xml = Path(__file__).resolve().parents[2] / "Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml"
all_ids = sorted({int(i) for i in re.findall(r'<route[^>]*\bid="(\d+)"', xml.read_text())} - {27515})
crash = {25854, 25896, 25955, 24816}
obst = [int(x) for x in open(obst_f).read().strip().split(",")]
r19 = [int(x) for x in open(r19_f).read().strip().replace("\n", ",").split(",") if x]
rep60 = [r for r in dict.fromkeys(obst + r19) if r not in crash]
other = [r for r in all_ids if r not in set(rep60) and r not in crash]
print(len(all_ids), "routes;", len(rep60), "repeat routes;", len(other), "other routes")
med = sorted(dur.values())[len(dur) // 2]
OVERHEAD = 150  # s per route run (CARLA/town load, agent set-up), measured ~10 lane-min per route-run on 2026-10-03
wt_j = {r: dur.get(r, med) + OVERHEAD for r in other}
wt_w = {r: 2.7 * dur.get(r, med) + OVERHEAD for r in rep60}  # WoR drives ~2.7x slower per route than E in the A9 records
lanes = []
for spec in boxes:
    name, ng, lpg = spec.split(":")
    for g in range(int(ng)):
        for k in range(int(lpg)):
            lanes.append({"box": name, "gpu": g, "port": 2000 + 400 * (len(lanes) % 12) + 0, "W": [], "J": [], "load": 0.0})
# distinct ports per box
per_box = {}
for L in lanes:
    i = per_box.get(L["box"], 0); per_box[L["box"]] = i + 1; L["port"] = 2000 + 400 * i
items = sorted([("W", r, 2 * wt_w[r]) for r in rep60] + [("J", r, 2 * wt_j[r]) for r in other], key=lambda t: -t[2])
for kind, r, w in items:  # longest first onto the least-loaded lane
    L = min(lanes, key=lambda x: x["load"])
    L[kind].append(r); L["load"] += w
E_J = ["carla_armJ_ft_obst", "carla_armJ_ft_obst_s1"]
out = {}
for L in lanes:
    b, p = L["box"], L["port"]
    W, J = ",".join(map(str, sorted(L["W"]))), ",".join(map(str, sorted(L["J"])))
    jobs = []
    if W: jobs.append(f"WORL:r{b}a_{p}:{W}")
    if J: jobs.append(f"CKL:j20_n{b}_{p}:{E_J[0]}/model_epoch_020.pth:{J}")
    if W: jobs.append(f"WORL:r{b}b_{p}:{W}")
    if J: jobs.append(f"CKL:j1s20_n{b}_{p}:{E_J[1]}/model_epoch_020.pth:{J}")
    out.setdefault(b, []).append(f"{b} {p} {L['gpu']} " + " ".join(jobs))
    print(f"{b} port {p} gpu {L['gpu']}: {len(L['W'])} WoR + {len(L['J'])} J routes, est {L['load'] / 3600:.2f} h")
for b, ls in out.items():
    Path(out_dir, f"lanes_{b}.txt").write_text("\n".join(ls) + "\n", newline="\n") if False else open(Path(out_dir, f"lanes_{b}.txt"), "w", newline="\n").write("\n".join(ls) + "\n")
print("total est lane-hours", sum(L["load"] for L in lanes) / 3600, "-> wall at", len(lanes), "lanes", max(L["load"] for L in lanes) / 3600, "h")
