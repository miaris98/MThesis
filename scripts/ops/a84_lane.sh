#!/usr/bin/env bash
# One A84 lane (started by a84_lanes.sh): lane J (CARLA port 2000 + 10 J, GPU 0) drives chunk K of /workspace/chunks8.txt with vf_clean, then
# a84_rb, then a84_dg with WOR_DANGER_GATE. An arm's turn waits for /workspace/checkpoints/a84_ARM/READY.
set -uo pipefail
J=${1:?lane}; K=${2:?chunk}; P=$(( 2000 + 10 * J ))
S=/workspace/a84/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
CH=$(sed -n ${K}p /workspace/chunks8.txt)
F="CARLA_QUALITY=Epic"
GATE="WOR_DANGER_GATE=theta=0.6;off=0.4;T=4;release=3;max_release=3"
run() {  # label ckpt_rel flags
  mark "LANE$J START $1 chunk $K"
  bash scripts/eval/queue_A9c.sh $P 0 "CKE:$1_c$K:$2:$CH:$3" > /workspace/evlogs/$1_c$K.out 2>&1
  mark "LANE$J END $1 chunk $K rc=$?"
}
run vf_clean vf_clean/model_epoch_030.pth "$F"
for arm in rb dg; do
  until [ -f /workspace/checkpoints/a84_$arm/READY ]; do sleep 60; done
  if [ $arm = dg ]; then run a84_dg_gate a84_dg/model_epoch_030.pth "$F,$GATE"; else run a84_$arm a84_$arm/model_epoch_030.pth "$F"; fi
done
mark "LANE$J ALL_DONE"
