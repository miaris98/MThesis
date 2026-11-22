#!/usr/bin/env bash
# TODO A81-A83 drives (2026-11-14): rerun the infrastructure failures (Simulation crashed, Agent couldn't be set up = CUDA OOM at load)
# and any route a finished set never drove (queue_A9c.sh's crash skip can skip the wrong route), so every set ends with all 220 routes.
# Pass 1 on port 2010 once lane 1's own queue has ended: only sets that no lane is still driving. Pass 2 on port 2010 once every lane
# has ended: whatever is still missing. One rerun label per set per pass (<set>_rr<pass>), records next to the others; the analysis
# takes a route's rerun record over its failed one. Marks RERUNS_DONE in $WORKSPACE/arms/drive_status when both passes are through.
# Run on the box:  cd $WORKSPACE; setsid nohup bash $WORKSPACE/MThesis/scripts/ops/arms_rerun_sweep.sh > $WORKSPACE/arms/sweep.out 2>&1 < /dev/null &
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
S=$WORKSPACE/arms/drive_status; mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis
P=2010
busy() { pgrep -f "[q]ueue_A9c.sh $1 " > /dev/null; }
anybusy() { pgrep -f "[q]ueue_A9c.sh 20[0-3]0 " > /dev/null; }
plan() {  # PASS -> job strings for the sets that are finished and still have failed or missing routes
  python3 - "$1" <<'PY'
import glob, json, os, re, subprocess, sys
pas = sys.argv[1]
OUT = "/workspace/bench2drive_out"
lanes = [l.strip().split(",") for l in open("/workspace/lanes220.txt") if l.strip()]
INFRA = ("Simulation crashed", "couldn't be set up")
running = set(re.findall(r"b2d_guardian\.sh (\S+)_x ", subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout))
sets = {"vf_ctrl": "vf_ctrl/model_epoch_030.pth"}
sets.update({f"arm_{a}": f"arm_{a}/model_epoch_030.pth" for a in ("s4", "w2", "w4", "ea", "wr")})
for s, ck in sets.items():
    files = glob.glob(f"{OUT}/{s}_*_x.json")
    labels = {os.path.basename(f)[:-7] for f in files}
    if not files or any(l.startswith(s + "_") for l in running):
        continue                                   # not started, or a lane is still driving it
    if not all(any(l.startswith(f"{s}_L{k}") for l in labels) for k in (1, 2, 3)):
        continue                                   # some lane has not reached this set yet
    ok = set()
    try:
        for f in files:
            for r in json.load(open(f)).get("_checkpoint", {}).get("records", []):
                if not any(x in r["status"] for x in INFRA):
                    ok.add(r["route_id"].split("_")[1])
    except (ValueError, OSError):
        continue                                   # a file mid-write: leave this set for the next pass
    want = {r for lane in lanes for r in lane}
    redo = sorted(want - ok, key=int)
    if redo:
        print(f"CKE:{s}_rr{pas}:{ck}:{','.join(redo)}:CARLA_QUALITY=Epic")
PY
}
run_pass() {  # PASS
  JOBS=$(plan "$1" | tr '\n' ' ')
  if [ -z "${JOBS// /}" ]; then mark "RERUN_PASS_$1 nothing to rerun"; return; fi
  mark "RERUN_PASS_$1 port $P: $(echo $JOBS | grep -oE 'CKE:[a-z0-9_]+:[^:]+:[0-9,]+' | sed -E 's/CKE:([a-z0-9_]+):[^:]+:/\1 /' | tr '\n' ';')"
  bash scripts/eval/queue_A9c.sh $P 0 $JOBS > $WORKSPACE/evlogs/rerun_pass$1.out 2>&1 < /dev/null
  mark "RERUN_PASS_$1 done"
}
if [ "${1:-}" = plan ]; then plan test; exit 0; fi   # dry run: print what a pass would rerun now
mark "SWEEP_WAITING for lane 1 (2010) to end"
until ! busy 2010; do sleep 60; done
sleep 90; until ! busy 2010; do sleep 60; done
run_pass 1
mark "SWEEP_WAITING for every lane to end"
n=0; while [ $n -lt 3 ]; do if anybusy; then n=0; else n=$((n + 1)); fi; sleep 60; done
run_pass 2
mark "RERUNS_DONE"
