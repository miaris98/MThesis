#!/usr/bin/env bash
# TODO A84 control (2026-11-17): the dg arm WITHOUT the danger gate, on the same lane chunk as the lane's gated dg run. Same checkpoint, so the
# difference between a84_dg_gate and a84_dg_nogate is the gate alone. Starts only after the lane's own ALL_DONE mark (one queue per CARLA port).
# usage: bash scripts/ops/a84_lane_nogate.sh J K        (J = lane 1-4 on this box, K = the chunk number from $WORKSPACE/chunks8.txt)
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
J=${1:?lane}; K=${2:?chunk}; P=$(( 2000 + 10 * J ))
S=$WORKSPACE/a84/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis
CH=$(sed -n ${K}p $WORKSPACE/chunks8.txt)
until grep -q "LANE$J ALL_DONE" $S; do sleep 60; done
mark "NOGATE$J START a84_dg_nogate chunk $K"
bash scripts/eval/queue_A9c.sh $P 0 "CKE:a84_dg_nogate_c$K:a84_dg/model_epoch_030.pth:$CH:CARLA_QUALITY=Epic" > $WORKSPACE/evlogs/a84_dg_nogate_c$K.out 2>&1
mark "NOGATE$J END a84_dg_nogate chunk $K rc=$?"
