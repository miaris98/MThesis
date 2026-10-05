"""CARLA lane files for a box (S-126): which arms on which routes, lanes sized from the box, jobs balanced by the A9 per-route wall time.

    py scripts/ops/gen_lanes.py ROUTE_DUR.json OUT_FILE BOX:GPUS:VRAM_GB:CPUS [--routes lead_obstacle|all|FILE] [--rest-first] ARM [ARM ...]
    ARM = LABEL=CKPT_REL                         a plain checkpoint (CKL job):  a36m50=a36m50/model_epoch_020.pth
        | LABEL=CKPT_REL|K=V,K=V                  the same with environment flags (CKE job): j_a50=carla_armJ_ft_obst/model_epoch_020.pth|WOR_CONTACT_REFLEX=1,WOR_CREEP_CLAMP=1
                                                  (a ';' inside a value stands for ',': WOR_NEWSVENDOR=q0=0.3;T=8;qmax=0.7)

Lanes per GPU = floor(VRAM_GB / 5.5) (S-126: a lane needs ~5 GB of VRAM at 288x768, 8 lanes on a 24 GB card gave 40 of 59 routes as agent OOM failures), capped at CPUS / 5 in total
(CARLA needs ~5 cores per lane) and at 12 per box; fewer lanes if a lane would get fewer than 8 routes (job start-up and hand-over cost ~2 min per job).
Output: one line `BOX PORT GPU JOBS` per lane for scripts/ops/launch_lanes_nokill.sh (arms in the given order: priority arms first).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from a3_failure_classes import LEAD_FAMILIES, OBSTACLE_FAMILIES, scenario_table  # noqa: E402

CRASH = {27515, 25854, 25896, 25955, 24816}   # routes that kill the CARLA server on every attempt (S-095, S-119)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route_dur")
    ap.add_argument("out")
    ap.add_argument("box", help="NAME:GPUS:VRAM_GB:CPUS")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--routes", default="lead_obstacle", help="lead_obstacle (the ~100 stage-1 screen routes), all (the 219 drivable routes) or a file with comma-separated ids")
    ap.add_argument("--port0", type=int, default=2000)
    a = ap.parse_args()

    dur = {int(k): v for k, v in json.load(open(a.route_dur)).items()}
    tab = scenario_table()
    if a.routes == "lead_obstacle":
        routes = [r for r, v in tab.items() if v["family"] in LEAD_FAMILIES or v["family"] in OBSTACLE_FAMILIES]
    elif a.routes == "all":
        routes = list(tab)
    else:
        routes = [int(x) for x in open(a.routes).read().replace("\n", ",").split(",") if x.strip()]
    routes = sorted(r for r in routes if r not in CRASH)
    name, ng, vram, cpus = a.box.split(":")
    ng, vram, cpus = int(ng), float(vram), int(cpus)
    per_gpu = max(1, int(vram // 5.5))
    n_lanes = min(per_gpu * ng, max(1, cpus // 5), 12)
    n_lanes = max(1, min(n_lanes, len(routes) // 8))
    print(f"{len(routes)} routes, {len(a.arms)} arms; box {name}: {ng} GPU x {vram:g} GB -> {per_gpu} lanes per GPU by VRAM, {cpus} CPUs -> {max(1, cpus // 5)} by CPU: {n_lanes} lanes "
          f"({len(routes) / n_lanes:.1f} routes per job)")
    lanes = [{"gpu": i % ng, "port": a.port0 + 400 * i, "R": [], "load": 0.0} for i in range(n_lanes)]
    med = sorted(dur.values())[len(dur) // 2]
    for r in sorted(routes, key=lambda r: -dur.get(r, med)):
        L = min(lanes, key=lambda x: x["load"])
        L["R"].append(r)
        L["load"] += dur.get(r, med) + 150
    lines = []
    for L in lanes:
        rs = ",".join(map(str, sorted(L["R"])))
        jobs = []
        for arm in a.arms:
            lab, rest = arm.split("=", 1)
            ck, _, env = rest.partition("|")
            jobs.append(f"CKE:{lab}_n{name}_{L['port']}:{ck}:{rs}:{env}" if env else f"CKL:{lab}_n{name}_{L['port']}:{ck}:{rs}")
        lines.append(f"{name} {L['port']} {L['gpu']} " + " ".join(jobs))
        print(f"port {L['port']} gpu {L['gpu']}: {len(L['R'])} routes, est {L['load'] * len(a.arms) / 3600:.2f} h for {len(a.arms)} arms")
    open(a.out, "w", newline="\n").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
