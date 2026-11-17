#!/usr/bin/env bash
# A81 lanes, started exactly once (2026-11-12 fix: the driver and the supervisor both launched them, giving three copies per lane).
# Each lane runs the baseline (vf_ctrl) then the arm on the same obstacle routes, same host, CARLA Epic.
set -uo pipefail
cd /workspace/MThesis; export MTHESIS_ROOT=/workspace/MThesis
ARM=a81_smoe16; F="CARLA_QUALITY=Epic"; IDS=/workspace/obst_lanes.txt; S=/workspace/a81/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
if pgrep -f "[q]ueue_A9c.sh 20[0-2]0 " >/dev/null; then mark "LANES_REFUSED already running"; exit 1; fi
for k in 1 2 3; do
  L=$(sed -n ${k}p $IDS); P=$((1990 + 10 * k))
  setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "CKE:vfctrl_obst_L$k:vf_ctrl/model_epoch_030.pth:$L:$F" "CKE:a81_obst_L$k:$ARM/model_epoch_030.pth:$L:$F" \
    > /workspace/evlogs/lane$k.out 2>&1 < /dev/null &
done
mark "LANES_STARTED_ONCE baseline then $ARM"
