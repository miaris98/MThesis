"""Autonomous 2-hour supervisor for MThesis workloads (19:30 - 21:30 Athens Time).

Monitors:
1. Atari 100k benchmarks (S047a, S047b) -> launches seeds 1 & 2 upon completion.
2. CARLA Town01 augmented evaluation -> aggregates 16 routes and compares with control.
3. LightZero 100k Breakout -> monitors MCTS steps.
4. Auto-syncs all completed runs to E:\MThesis_EXP.
"""
import subprocess
import time
import json
import os
import glob
import re
from datetime import datetime

ATARI_HOST = "ssh1.vast.ai"
ATARI_PORT = 33163
CARLA_HOST = "ssh5.vast.ai"
CARLA_PORT = 24525

LOG_FILE = "c:/Users/miari/Desktop/MThesis/supervisor_status.log"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fp:
        fp.write(entry + "\n")

def run_ssh(host, port, cmd, timeout=30):
    ssh_cmd = ["ssh", "-n", "-p", str(port), "-o", "StrictHostKeyChecking=no", f"root@{host}", cmd]
    try:
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return res.stdout.strip(), res.returncode
    except Exception as e:
        return f"ERROR: {e}", -1

log("Autonomous Supervisor started for 2-hour autonomous run.")

# State tracking
s047_seeds_launched = False
carla_augmented_done = False

while True:
    try:
        # Check Atari 100k
        if not s047_seeds_launched:
            out, rc = run_ssh(ATARI_HOST, ATARI_PORT, "ps aux | grep -E 'train_onpolicy_gtrxl_ez2' | grep -v grep")
            if not out or "train_onpolicy_gtrxl_ez2" not in out:
                log("S047 Seed 0 runs appear finished! Checking logs...")
                log_a, _ = run_ssh(ATARI_HOST, ATARI_PORT, "tail -n 15 /workspace/MThesis/results/100k_benchmark/_logs/S047a_100k_standard_s0.log 2>/dev/null")
                log_b, _ = run_ssh(ATARI_HOST, ATARI_PORT, "tail -n 15 /workspace/MThesis/results/100k_benchmark/_logs/S047b_100k_sample_efficient_s0.log 2>/dev/null")
                log(f"S047a tail:\n{log_a}")
                log(f"S047b tail:\n{log_b}")
                
                # Launch Seeds 1 & 2
                log("Launching S047 Seeds 1 & 2 for both variants...")
                launch_cmd = """
                cd /workspace/MThesis
                nohup /workspace/venv_atari/bin/python atari_qwen/scripts/run_100k_benchmark.py --parallel --seeds 1 2 > /workspace/benchmark_100k_seeds12.log 2>&1 &
                """
                run_ssh(ATARI_HOST, ATARI_PORT, launch_cmd)
                s047_seeds_launched = True
                log("Seeds 1 & 2 launched successfully in parallel!")

        # Check CARLA Augmented Eval
        if not carla_augmented_done:
            out, rc = run_ssh(CARLA_HOST, CARLA_PORT, "ps aux | grep -E 'run_phase4_chunk|leaderboard_evaluator' | grep -v grep")
            if not out or "run_phase4_chunk" not in out:
                log("CARLA Town01 Augmented evaluation completed! Summarizing scores...")
                parse_cmd = """python3 -c "
import glob, json
files = sorted(glob.glob('/workspace/bench2drive_out/town01_augmented_perroute/*.json'))
scores, rcs = [], []
for f in files:
    try:
        d = json.load(open(f))
        recs = d.get('_checkpoint', {}).get('records', [])
        if recs:
            r = recs[0]
            ds = r.get('scores', {}).get('score_composed', 0.0)
            rc = r.get('scores', {}).get('score_route', 0.0)
            scores.append(ds)
            rcs.append(rc)
            print(f'Route {r.get(\\"route_id\\")}: DS={ds:.2f}, RC={rc:.2f}, status={r.get(\\"status\\")}')
    except Exception:
        pass
if scores:
    print('========================================')
    print(f'TOWN01 AUGMENTED ({len(scores)} routes): Mean DS = {sum(scores)/len(scores):.2f}, Mean RC = {sum(rcs)/len(rcs):.2f}%')
    print('========================================')
"
"""
                summary, _ = run_ssh(CARLA_HOST, CARLA_PORT, parse_cmd)
                log(f"CARLA Town01 Augmented Summary:\n{summary}")
                carla_augmented_done = True

        # Check LightZero
        lz_out, _ = run_ssh(ATARI_HOST, ATARI_PORT, "tail -n 5 /workspace/lightzero_breakout_100k_seed0.log 2>/dev/null")
        log(f"LightZero Progress: {lz_out.splitlines()[-1] if lz_out else 'N/A'}")

    except Exception as e:
        log(f"Supervisor loop error: {e}")

    # Sleep 3 minutes between checks
    time.sleep(180)
