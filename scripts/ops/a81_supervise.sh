#!/usr/bin/env bash
# TODO A81 supervisor (2026-11-12, replaces the training window of `a81_box.sh all` for the 7-hour limit).
# Training ends at CUT_UTC (default 09:15 UTC). Closed-loop lanes run only if the arm reached epoch 30, the same 15-epoch schedule
# vf_ctrl trained with; a truncated arm is not comparable, so the run stops there and spends no evaluation time.
# The baseline (vf_ctrl) and the arm then run on the same three lanes, same host, same flags, one after the other per lane.
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
CUT_UTC=${CUT_UTC:-09:15}; S=$WORKSPACE/a81/status; ARM=a81_smoe16
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis; export PYTHONPATH=$WORKSPACE/MThesis MLFLOW_ALLOW_FILE_STORE=true
CUT_EPOCH=$(date -u -d "$(date -u +%Y-%m-%d) $CUT_UTC" +%s)
F="CARLA_QUALITY=Epic"
until grep -qE "CACHE_DONE|CACHE_FAILED" $S; do sleep 20; done
grep -q CACHE_DONE $S || { mark "STOP cache failed"; exit 1; }
mark "SUPERVISE_TRAIN_START cut_at=$CUT_UTC UTC"
A81_PY=main bash scripts/ops/a81_box.sh train & TP=$!
while kill -0 $TP 2>/dev/null; do
  if [ "$(date +%s)" -ge "$CUT_EPOCH" ]; then pkill -f "[t]rain_wor.py"; mark "TRAIN_CUT_AT $CUT_UTC"; break; fi
  sleep 30
done
wait $TP 2>/dev/null
LAST=$(ls -1 $WORKSPACE/checkpoints/$ARM/model_epoch_*.pth 2>/dev/null | sort -V | tail -1)
if [ ! -f $WORKSPACE/checkpoints/$ARM/model_epoch_030.pth ]; then
  mark "ARM_TRUNCATED last=$(basename "${LAST:-none}") (epoch 30 not reached): no closed-loop lanes; not comparable to vf_ctrl"
  exit 0
fi
mark "ARM_COMPLETE epoch 30"
until grep -q CARLA_PROV_END $WORKSPACE/vf/status 2>/dev/null; do sleep 30; done
A81_CKPT=$WORKSPACE/checkpoints/$ARM/model_epoch_030.pth bash scripts/ops/a81_box.sh loadcheck || { mark "STOP loadcheck failed"; exit 1; }
IDS=$WORKSPACE/obst_lanes.txt; mkdir -p $WORKSPACE/evlogs
for k in 1 2 3; do
  L=$(sed -n ${k}p $IDS); P=$((1990 + 10 * k))
  (setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "CKE:vfctrl_obst_L$k:vf_ctrl/model_epoch_030.pth:$L:$F" "CKE:a81_obst_L$k:$ARM/model_epoch_030.pth:$L:$F" \
    > $WORKSPACE/evlogs/lane$k.out 2>&1 < /dev/null &)
done
mark "LANES_STARTED baseline then $ARM"
while pgrep -f "[q]ueue_A9c.sh 20[0-2]0 " > /dev/null; do sleep 60; done
mark ALL_LANES_DONE
