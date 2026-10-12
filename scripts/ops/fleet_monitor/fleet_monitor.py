#!/usr/bin/env python3
"""10-minute fleet check: progress, errors, ETAs and checkpoint verification for the Vast boxes in boxes.txt.

    python fleet_monitor.py status     one line per box + ALERT lines + credit/burn   (run every 10 min)
    python fleet_monitor.py verify     box vs E: size/age of checkpoint_latest.pt and replay_latest.npz.gz per Atari box (run after a backup round)

Boxes: boxes.txt next to this file (name|kind|port|host|label). Atari label = run label, CARLA label = result-file prefix
(bench2drive_out/<label>*_x.json). Alerts (prefixed ALERT): unreachable box, trainer not running, log stale > 15 min, GPU < 40% under
training, evaluator process competing with training, Traceback / nan-guard / gn-stop / OOM lines, checkpoint older than 40 min,
CARLA routes never valid (infra failure: agent setup, simulator or agent crash, with no valid record anywhere in the
fleet) or crash-skipped routes, CARLA lanes dead while routes remain, credit below 3 h of burn.
CARLA "valid" = a record that is not one of those infra failures (TickRuntime is a driving outcome: the 4,000-tick cap, as in
check_leaderboard_results.py); the total is the union of valid route ids over all boxes, the latest driving record per route wins.
"""
import datetime, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAST = os.environ.get("VAST", r"C:\Users\miari\anaconda3\envs\graphtools\Scripts\vastai.exe")
DAY = os.environ.get("FLEET_DAY", datetime.date.today().strftime("%Y%m%d"))
RD = "MThesis/results/100k_benchmark/S058_ezv2_match"
SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]

PROBE = r'''
import json, glob, os, re, subprocess, sys, time
kind, label = sys.argv[1], sys.argv[2]
now = time.time(); out = {"kind": kind}
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
g = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits").split(",")
out["gpu"] = int(g[0]) if g and g[0].strip().isdigit() else None
out["vram"] = int(g[1]) if len(g) > 1 and g[1].strip().isdigit() else None
out["token"] = os.path.exists("/dev/shm/hf_token") and os.path.getsize("/dev/shm/hf_token") > 0
out["disk_free_gb"] = int(sh("df -BG /workspace | tail -1 | awk '{print $4}'").rstrip("G") or 0)
if kind == "atari":
    L = f"/workspace/{label}.log"
    lines = [l for l in open(L, errors="replace")] if os.path.exists(L) else []
    envl = [l for l in lines if l.startswith("env ")]
    out["log_age_min"] = round((now - os.path.getmtime(L)) / 60, 1) if lines else None
    if envl:
        m = re.match(r"env +(\d+)/100000 upd +(\d+).*train-life-ret\(20\) ([\d.\-]+).*\| ([\d.]+) min$", envl[-1].strip())
        if m: out.update(env=int(m[1]), upd=int(m[2]), ret=float(m[3]))
        pts = []
        for l in envl[-5:]:
            mm = re.match(r"env +(\d+).*\| ([\d.]+) min$", l.strip())
            if mm: pts.append((int(mm[1]), float(mm[2])))
        if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
            out["pace"] = round((pts[-1][1] - pts[0][1]) / ((pts[-1][0] - pts[0][0]) / 1000), 2)
        ups = []
        for l in envl[-5:]:
            mu = re.match(r"env +\d+/100000 upd +(\d+).*\| ([\d.]+) min$", l.strip())
            if mu: ups.append((int(mu[1]), float(mu[2])))
        if len(ups) >= 2 and ups[-1][0] > ups[0][0]:
            out["upd_pace"] = round((ups[-1][1] - ups[0][1]) / ((ups[-1][0] - ups[0][0]) / 1000), 2)
    txt = "".join(lines)
    out["traceback"] = txt.count("Traceback"); out["nan_guard"] = txt.count("[nan-guard]"); out["gn_stop"] = txt.count("[gn-stop]")
    out["oom"] = txt.count("out of memory")
    out["trainer"] = int(sh("pgrep -fc '[t]rain_ez_offpolicy'") or 0)
    out["evals"] = int(sh("pgrep -fc '[e]val_ez_checkpoints'") or 0)
    ck = f"/workspace/{'MThesis/results/100k_benchmark/S058_ezv2_match'}/{label}/checkpoints"
    for f in ("checkpoint_latest.pt", "replay_latest.npz", "replay_latest.npz.gz"):
        p = f"{ck}/{f}"
        if os.path.exists(p): out[f] = [os.path.getsize(p), round((now - os.path.getmtime(p)) / 60, 1)]
    ev = sh(f"grep -h '\\[EVAL' /workspace/{label}_eval.log {L} 2>/dev/null | tail -n1")
    mm = re.search(r"env(\d+) .*?: ([\d.]+) \+/- ([\d.]+) SE", ev)
    out["lasteval"] = f"env{mm[1]} {mm[2]}+-{mm[3]}" if mm else None
    sm = re.search(r"s(\d)_uniform", label)
    rs = f"/workspace/resume_s{sm.group(1)}.log" if sm else ""
    out["resume_log"] = sh(f"tail -n 2 {rs} 2>/dev/null | tr '\\n' ' ' | cut -c1-160") if rs else ""
else:
    INFRA = ("set up", "Simulation crashed", "Agent crashed")  # TickRuntime is a driving outcome (4,000-tick cap), not infra
    n = bad = 0; ds = 0.0; stamps = []; valid = {}; infra = set()
    for f in sorted(glob.glob(f"/workspace/bench2drive_out/{label}*_x.json"), key=os.path.getmtime):  # oldest first: a later driving record replaces an earlier one (merge_leaderboard_results.py)
        try: recs = json.load(open(f)).get("_checkpoint", {}).get("records", [])
        except Exception: continue
        for r in recs:
            rid = re.search(r"RouteScenario_(\d+)", r.get("route_id", ""))
            if rid is None: continue
            if any(b in r["status"] for b in INFRA): bad += 1; infra.add(int(rid.group(1))); continue
            n += 1; ds += r["scores"]["score_composed"]; valid[int(rid.group(1))] = r["scores"]["score_composed"]
            m = re.search(r"_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)$", r.get("save_name", ""))
            if m: stamps.append(tuple(int(x) for x in m.groups()))
    out.update(n=n, invalid=bad, ds=round(ds / max(n, 1), 1), stamps=sorted(stamps)[-40:], valid=valid, infra=sorted(infra))
    out["lanes"] = int(sh("pgrep -fc '[q]ueue_A9c.sh'") or 0) // 2
    out["carla"] = int(sh("pgrep -fc '[C]arlaUE4-Linux-Shipping'") or 0)
    out["crash_skips"] = int(sh("cat /workspace/lane_*.log 2>/dev/null | grep -c 'CRASH ROUTE'") or 0)
    out["provisioned"] = os.path.exists("/workspace/setup_carla.log") and "PROVISION_DONE" in open("/workspace/setup_carla.log", errors="replace").read()
    newest = sh("ls -t /workspace/bench2drive_out/*_x.json 2>/dev/null | head -1")
    out["newest_result_min"] = round((now - os.path.getmtime(newest)) / 60, 1) if newest else None
print(json.dumps(out))
'''


def ssh_json(name, port, host, kind, label):
    try:
        r = subprocess.run(SSH + ["-p", str(port), f"root@{host}", f"python3 - {kind} {label}"], input=PROBE,
                           capture_output=True, text=True, timeout=90)
        lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {"error": (r.stderr.strip().splitlines() or ["no output"])[-1][:100]}
    except Exception as ex:
        return {"error": repr(ex)[:100]}


def boxes():
    for ln in (HERE / "boxes.txt").read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            yield ln.split("|")


def credit_burn():
    try:
        d = json.loads(subprocess.run([VAST, "show", "instances", "--raw"], capture_output=True, text=True, timeout=60,
                                      env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout)
        u = json.loads(subprocess.run([VAST, "show", "user", "--raw"], capture_output=True, text=True, timeout=60,
                                      env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout)
        bad = [(i["id"], i["actual_status"]) for i in d if i["actual_status"] != "running"]
        return len(d), sum(i["dph_total"] for i in d), u.get("credit", 0), bad
    except Exception as ex:
        return None, None, None, [repr(ex)[:80]]


def carla_rate(stamps):
    """routes per hour from the save_name timestamps (UTC mm_dd_hh_mm_ss) of the last records."""
    if len(stamps) < 4: return None
    ts = [datetime.datetime(2026, *s) for s in stamps]
    span = (ts[-1] - ts[0]).total_seconds() / 3600
    return (len(ts) - 1) / span if span > 0 else None


def status():
    print(f"[{(datetime.datetime.utcnow() + datetime.timedelta(hours=3)).strftime('%H:%M')} Athens]")
    alerts, carla_tot = [], {}
    for name, kind, port, host, label in boxes():
        o = ssh_json(name, port, host, kind, label)
        if "error" in o:
            print(f"{name} {kind}: UNREACHABLE ({o['error']})"); alerts.append(f"{name} unreachable (retry 10-25 min before calling it destroyed)"); continue
        gpu = f"gpu {o['gpu']}% {o['vram']}MiB" if o.get("gpu") is not None else "gpu ?"
        if kind == "atari":
            cur, pace = o.get("env"), o.get("pace")
            if cur is None:
                wait = "waiting for HF token" if not o["token"] else (o["resume_log"] or "starting")
                state = "STARTING (trainer up, no progress line yet)" if o["trainer"] else "NOT RUNNING YET"
                print(f"{name} atari {label}: {state} ({wait}) | {gpu}");
                if o["token"] and not o["trainer"]: alerts.append(f"{name}: token present but no trainer running - check resume log: {o['resume_log']}")
                continue
            if cur >= 100000 and o.get("upd") is not None and o.get("upd_pace"):
                hrs = (120000 - o["upd"]) / 1000 * o["upd_pace"] / 60   # env is done: the trainer runs offline updates to 120k
                eta, done_at = f"{hrs:.1f}h to 120k upd", (datetime.datetime.utcnow() + datetime.timedelta(hours=3 + hrs)).strftime("%H:%M")
            else:
                eta = f"{(100000 - cur) / 1000 * pace / 60:.1f}h" if pace else "?"
                done_at = (datetime.datetime.utcnow() + datetime.timedelta(hours=3 + (100000 - cur) / 1000 * pace / 60)).strftime("%H:%M") if pace else "?"
            ck = o.get("checkpoint_latest.pt"); rp = o.get("replay_latest.npz")
            print(f"{name} atari {label}: env {cur}/100000 pace {pace} min/1k ETA {eta} (~{done_at} Athens) ret {o.get('ret')} lastEval {o.get('lasteval')} "
                  f"| {gpu} | ckpt age {ck[1] if ck else '?'} min | trainer {o['trainer']} evals {o['evals']} | errs tb {o['traceback']} nan {o['nan_guard']} gnstop {o['gn_stop']} oom {o['oom']}")
            if not o["trainer"] and cur < 100000: alerts.append(f"{name}: trainer not running at {cur}")
            if o.get("log_age_min") is not None and o["log_age_min"] > 15 and o["trainer"]: alerts.append(f"{name}: training log stale {o['log_age_min']} min")
            if o["trainer"] and o.get("gpu") is not None and o["gpu"] < 40: alerts.append(f"{name}: GPU {o['gpu']}% under training")
            if o["evals"]: alerts.append(f"{name}: {o['evals']} evaluator process(es) share the GPU with training")
            if o["traceback"] or o["nan_guard"] or o["gn_stop"] or o["oom"]: alerts.append(f"{name}: errors in log (tb {o['traceback']}, nan {o['nan_guard']}, gn-stop {o['gn_stop']}, oom {o['oom']})")
            if ck and ck[1] > 40: alerts.append(f"{name}: checkpoint_latest.pt is {ck[1]} min old (saves every ~25 min)")
            if o["disk_free_gb"] < 10: alerts.append(f"{name}: only {o['disk_free_gb']} GB disk free")
        else:
            rate = carla_rate(o.get("stamps", []))
            tot = carla_tot.setdefault(label, {"valid": {}, "infra": set(), "rate": 0.0})
            tot["valid"].update({int(k): v for k, v in o.get("valid", {}).items()}); tot["infra"].update(o.get("infra", [])); tot["rate"] += rate or 0.0
            print(f"{name} carla {label}: {o['n']} routes valid, {o['invalid']} infra-failed records, mean DS {o['ds']} | lanes {o['lanes']} carla {o['carla']} | {gpu} | "
                  f"crash-skips {o['crash_skips']} | rate {rate and round(rate, 1)}/h | provisioned {o['provisioned']} | last result {o['newest_result_min']} min ago")
            if not o["provisioned"]: print("   (CARLA stack not provisioned yet)"); continue
            if o["crash_skips"]: alerts.append(f"{name}: {o['crash_skips']} crash-skipped route(s): test per (agent, switches, rendering) before calling them broken")
            if o["lanes"] == 0: alerts.append(f"{name}: no CARLA lanes running")
            if o["newest_result_min"] is not None and o["newest_result_min"] > 45 and o["lanes"]: alerts.append(f"{name}: no new result for {o['newest_result_min']} min while lanes run (hung lanes?)")
    for label, t in carla_tot.items():
        valid = set(t["valid"]); never = sorted(t["infra"] - valid); todo = 220 - len(valid | t["infra"])
        mean = sum(t["valid"].values()) / max(len(valid), 1)
        eta = f", ETA {(220 - len(valid)) / t['rate']:.1f} h" if t["rate"] > 0 else ""
        print(f"CARLA {label}: {len(valid)}/220 valid (union over boxes, mean DS {mean:.1f}), {len(never)} infra-failed with no valid record {never}, {todo} not yet run, fleet rate {t['rate']:.1f} routes/h{eta}")
        if never: alerts.append(f"{len(never)} route(s) never valid (infra failure, no valid record anywhere): {never} - rerun at Epic")
    nbox, burn, credit, bad = credit_burn()
    if nbox is not None:
        print(f"boxes {nbox} burn ${burn:.2f}/h | credit ${credit:.2f} ({credit / burn:.1f} h of runway) | not running: {bad}")
        if credit / burn < 3: alerts.append(f"credit ${credit:.2f} covers only {credit / burn:.1f} h at ${burn:.2f}/h - top up")
        if bad: alerts.append(f"instances not running: {bad}")
    for a in alerts: print("ALERT", a)
    if not alerts: print("no alerts")


def verify():
    """box vs E: for the resume pair of every Atari seed (E: copy must exist, match the box size and be < 40 min older than the box file)."""
    import time
    for name, kind, port, host, label in boxes():
        if kind != "atari": continue
        o = ssh_json(name, port, host, kind, label)
        if "error" in o: print(f"VERIFY {name}: unreachable"); continue
        edir = Path(f"E:/MThesis_EXP/live_{DAY}_{name}/{RD}/{label}/checkpoints")
        for f, bf in (("checkpoint_latest.pt", "checkpoint_latest.pt"), ("replay_latest.npz.gz", "replay_latest.npz.gz")):
            b = o.get(bf); p = edir / f
            if not b: print(f"VERIFY {name} {f}: not on box yet"); continue
            if not p.exists(): print(f"VERIFY {name} {f}: MISSING on E: (box {b[0]} B)"); continue
            age = (time.time() - p.stat().st_mtime) / 60
            same = p.stat().st_size == b[0]
            print(f"VERIFY {name} {f}: box {b[0]} B E: {p.stat().st_size} B {'SAME' if same else 'DIFF (box file is newer or the copy is truncated)'}, E: copy {age:.0f} min old")


if __name__ == "__main__":
    {"status": status, "verify": verify}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
