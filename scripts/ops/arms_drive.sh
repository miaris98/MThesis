#!/usr/bin/env bash
# TODO A81-A83 drives (2026-11-14): the six 220-route drives on ONE box: vf_ctrl (the paired baseline, same host and lanes), then the
# arms s4, w2, w4, ea, wr. Each of the three lanes runs its own queue of all six sets in that order on its own route list
# (/workspace/lanes220.txt line k), so a lane never idles waiting for the slowest lane of the previous set (the first version did:
# one set at a time, every hand-over waited for the slowest of three lanes). Pairing is unchanged: same host, lanes, routes and flags.
# Before any lane starts, every arm passes arms_box.sh loadcheck (its new experts are loaded and non-zero); a failing arm is left out
# of the queues and marked SKIP_ARM. Status -> /workspace/arms/drive_status.
# Run on the box after PROVISION_DONE is in /workspace/setup_carla.log (it waits for it):
#   cd /workspace; setsid nohup bash /workspace/MThesis/scripts/ops/arms_drive.sh > /workspace/arms/drive.out 2>&1 < /dev/null &
set -uo pipefail
mkdir -p /workspace/arms /workspace/evlogs; S=/workspace/arms/drive_status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
LANES=/workspace/lanes220.txt; F="CARLA_QUALITY=Epic"
if pgrep -f "[q]ueue_A9c.sh 20[0-3]0 " > /dev/null; then mark "REFUSED lanes already running"; exit 1; fi
until grep -q PROVISION_DONE /workspace/setup_carla.log 2>/dev/null; do sleep 60; done
mark "PROVISION_SEEN"
[ "$(wc -l < $LANES)" -ge 3 ] || { mark "NO_ROUTE_FILE"; exit 1; }
[ -f /workspace/checkpoints/vf_ctrl/model_epoch_030.pth ] && [ -f /workspace/checkpoints/vf_ctrl/frozen_backbone.pth ] || { mark "BASE_CKPT_MISSING"; exit 1; }
OK=""
for ARM in s4 w2 w4 ea wr; do
  if bash scripts/ops/arms_box.sh loadcheck "$ARM"; then OK="$OK $ARM"; mark "LOADCHECK_OK $ARM"; else mark "SKIP_ARM $ARM loadcheck failed"; fi
done
for k in 1 2 3; do
  L=$(sed -n ${k}p $LANES); P=$((2000 + 10 * k))
  JOBS="CKE:vf_ctrl_L$k:vf_ctrl/model_epoch_030.pth:$L:$F"
  for ARM in $OK; do JOBS="$JOBS CKE:arm_${ARM}_L$k:arm_$ARM/model_epoch_030.pth:$L:$F"; done
  setsid nohup bash scripts/eval/queue_A9c.sh $P 0 $JOBS > /workspace/evlogs/lane$k.out 2>&1 < /dev/null &
  mark "LANE_STARTED $k port $P sets: vf_ctrl$OK"
  [ $k -lt 3 ] && sleep 20   # stagger the CARLA boots (Vulkan init race)
done
sleep 120
while pgrep -f "[q]ueue_A9c.sh 20[0-3]0 " > /dev/null; do sleep 60; done
mark "ALL_DRIVES_DONE"
