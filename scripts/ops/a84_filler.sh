#!/usr/bin/env bash
# TODO A84 (2026-11-17), eval box A only: fill each lane's wait for the A84 checkpoints with vf_clean seed 1 on that lane's own chunk (a free,
# same-host, per-route-paired measure of training-seed noise). Each lane starts when its own "LANEj END vf_clean" mark appears; FILLER_DONE
# when all four finished. The arms' READY markers must only be written after FILLER_DONE (one queue per CARLA port).
# usage: bash scripts/ops/a84_filler.sh A|B CKPT_DIR      (e.g. A vf_clean_s1)
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
BOX=${1:?A or B}; CK=${2:?checkpoint dir}; OFF=0; [ "$BOX" = B ] && OFF=4
S=$WORKSPACE/a84/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis
F="CARLA_QUALITY=Epic"
fill() {
  local j=$1 k=$(( OFF + $1 )) P=$(( 2000 + 10 * $1 )) CH
  CH=$(sed -n ${k}p $WORKSPACE/chunks8.txt)
  until grep -q "LANE$j END vf_clean chunk $k" $S; do sleep 30; done
  mark "FILL$j START $CK chunk $k"
  bash scripts/eval/queue_A9c.sh $P 0 "CKE:${CK}_c$k:$CK/model_epoch_030.pth:$CH:$F" > $WORKSPACE/evlogs/${CK}_c$k.out 2>&1
  mark "FILL$j END $CK chunk $k rc=$?"
}
for j in 1 2 3 4; do fill $j & done
wait
mark "FILLER_DONE $CK"
