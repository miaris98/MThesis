#!/usr/bin/env bash
# One CARLA eval lane (2026-09-28): TODO_ACTIVE A12 (finish arm I) and A9 (full Bench2Drive, arm E e15 vs the original
# WoR). usage: queue_A9c.sh PORT GPU JOB...
#   JOB = I15 | I20            arm I e15/e20 on the 19 b2d20 routes (resumes from the copied partial JSON)
#       | E15:<k> | WOR:<k>    chunk k (1..10) of bench2drive220 minus 27515, every 10th route from position k-1
#       | E15L:<name>:<r1,r2,..>  arm E e15 on an explicit route list as a9_E15_<name> (retry of skipped routes)
#       | WORL:<name>:<r1,r2,..>  the original WoR on an explicit route list as a9_WOR_<name> (rebalanced lanes, S-099)
# Lanes on one box need distinct PORTs (TM port = PORT+6000); each lane clears its own rpc range before a launch (S-091).
# A lane relaunched with a new job list attaches to a guardian that is already running instead of duplicating it.
# Crash routes (S-095): a route that segfaults CARLA on every attempt (like 27515) is skipped after 4 launch attempts in
# a row at the same progress. It goes into /workspace/a9_exclude.txt (applied only to chunks that have not started, so a
# resumed run never changes its route list - S-021) and the rest of the chunk runs as <label>r. The merge pairs E and WoR
# on the routes both completed.
cd /workspace/MThesis
# the agent needs little CPU; unbounded torch threads took ~4 cores per evaluator next to CARLA
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PORT=$1; GPU=$2; shift 2; TM=$((PORT + 6000))
XML=/workspace/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml
R19="2050,2143,2286,2509,2664,3373,3457,3717,3936,23687,24340,24784,24795,25318,25975,26401,27532,28035,28198"
C=/workspace/checkpoints
WOR_CFG=/workspace/PCLA/pcla_agents/wor_pretrained/leaderboard_weights/config_leaderboard.yaml
OUT=/workspace/bench2drive_out
EXCL=/workspace/a9_exclude.txt

chunk() {  # k use_exclusions(0/1)
  python3 -c '
import os, re, sys
ids = sorted({int(i) for i in re.findall(r"<route[^>]*\bid=\"(\d+)\"", open(sys.argv[1]).read())} - {27515})
sel = ids[int(sys.argv[2]) - 1::10]
if sys.argv[3] == "1" and os.path.exists(sys.argv[4]):
    ex = {int(x) for x in open(sys.argv[4]).read().split()}
    sel = [i for i in sel if i not in ex]
print(",".join(map(str, sel)))' "$XML" "$1" "$2" "$EXCL"; }
routes_for() {  # label k -> the route list this label must keep using
  if [ -f $OUT/${1}_x.routes ]; then cat $OUT/${1}_x.routes
  elif [ -f $OUT/${1}_x.json ]; then chunk $2 0    # started before route files existed: its original list
  else chunk $2 1; fi
}
# `local c`: without it the recursion overwrote c and printed the deepest leaf instead of each child, so the
# evaluator was never killed, stayed alive as an orphan holding the ports, and its guardian never relaunched
# (S-099)
desc() { local c; for c in $(ps -eo pid=,ppid= | awk -v p=$1 '$2==p {print $1}'); do desc $c; echo $c; done; }
clear_ports() { for q in $(seq $(( $1 - 2 )) $(( $1 + 4 ))); do for pid in $(pgrep -f "CarlaUE4.*-carla-rpc-port=$q( |$)"); do kill -9 $pid 2>/dev/null; done; done; }
nrec() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['_checkpoint']['records']))" $OUT/${1}_x.json 2>/dev/null || echo 0; }
eval_job() {  # label ckpt routes
  local label=$1 ckpt=$2 routes=$3 n p3 na p X rest G
  if [ -f $OUT/${label}_x.skipped ]; then  # this chunk already hit a crash route: go straight to its continuation
    read -r nl nr < $OUT/${label}_x.skipped; eval_job $nl "$ckpt" "$nr"; return
  fi
  n=$(awk -F, '{print NF}' <<<"$routes")
  [ "$(nrec $label)" = "$n" ] && { echo "$(date) skip $label ($n done)"; return; }
  [ -f $OUT/${label}_x.routes ] || echo "$routes" > $OUT/${label}_x.routes
  if pgrep -f "b2d_guardian.sh ${label}_x " > /dev/null; then
    echo "$(date) attach $label (guardian already running)"
  else
    clear_ports $PORT; sleep 5
    echo "$(date) start $label ($n routes, port $PORT, GPU $GPU)"
    LABEL=$label CKPT=$ckpt GPU=$GPU ARMS="x:$PORT:$TM:$routes" bash scripts/eval/launch_b2d20.sh run || { echo "$(date) FAILED to launch $label"; return; }
    sleep 120
  fi
  while pgrep -f "b2d_guardian.sh ${label}_x " > /dev/null; do
    sleep 60
    p3=$(grep -a "launch attempt=" /workspace/guardian_logs/${label}_x.log 2>/dev/null | tail -4 | grep -o "progress=[0-9]*" | sort -u)
    na=$(grep -a -c "launch attempt=" /workspace/guardian_logs/${label}_x.log 2>/dev/null)
    if [ "${na:-0}" -ge 4 ] && [ -n "$p3" ] && [ "$(wc -l <<<"$p3")" = 1 ]; then
      p=${p3#progress=}
      read -r X rest < <(python3 -c 'import sys; r = sys.argv[1].split(","); p = int(sys.argv[2]); print(r[p] if p < len(r) else "", ",".join(r[p + 1:]))' "$routes" "$p")
      [ -n "$X" ] || continue
      for G in $(pgrep -f "b2d_guardian.sh ${label}_x "); do kill -9 $(desc $G) $G 2>/dev/null; done
      clear_ports $PORT
      echo "$X" >> $EXCL
      echo "$(date) CRASH ROUTE $X: $label stuck at progress $p for 4 attempts - skipped (added to $EXCL)"
      if [ -n "$rest" ]; then echo "${label}r $rest" > $OUT/${label}_x.skipped; eval_job ${label}r "$ckpt" "$rest"; fi
      return
    fi
  done
  echo "$(date) done $label: $(nrec $label)/$n records"
}

until grep -q PROVISION_DONE /workspace/setup_carla.log 2>/dev/null && [ -f "$XML" ]; do sleep 60; done
pgrep -f "bash /workspace/MThesis/scripts/eval/eval_watchdog.sh" > /dev/null || \
  (setsid nohup bash /workspace/MThesis/scripts/eval/eval_watchdog.sh >> /workspace/eval_watchdog.log 2>&1 < /dev/null &)
for job in "$@"; do
  case $job in
    I15) eval_job v2_armI_e15 $C/carla_armI_hires_color/model_epoch_015.pth "$R19" ;;
    I20) eval_job v2_armI_e20 $C/carla_armI_hires_color/model_epoch_020.pth "$R19" ;;
    E15:*) k=${job#*:}; L=a9_E15_c$(printf %02d $k); eval_job $L $C/carla_armE_aug1_hires/model_epoch_015.pth "$(routes_for $L $k)" ;;
    E15L:*) spec=${job#*:}; eval_job a9_E15_${spec%%:*} $C/carla_armE_aug1_hires/model_epoch_015.pth "${spec#*:}" ;;
    WORL:*) spec=${job#*:}
      ( export EVAL_AGENT=/workspace/MThesis/scripts/eval/wor_official_b2d_agent.py EVAL_AGENT_CONFIG=$WOR_CFG PCLA_ROOT=/workspace/PCLA ALLOW_NO_FROZEN_BACKBONE=1
        eval_job a9_WOR_${spec%%:*} $WOR_CFG "${spec#*:}" ) ;;
    WOR:*) k=${job#*:}; L=a9_WOR_c$(printf %02d $k)
      ( export EVAL_AGENT=/workspace/MThesis/scripts/eval/wor_official_b2d_agent.py EVAL_AGENT_CONFIG=$WOR_CFG PCLA_ROOT=/workspace/PCLA ALLOW_NO_FROZEN_BACKBONE=1
        eval_job $L $WOR_CFG "$(routes_for $L $k)" ) ;;
    *) echo "unknown job $job" ;;
  esac
done
echo "$(date) lane $PORT/GPU$GPU done: $*"
