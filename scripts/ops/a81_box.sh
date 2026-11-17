#!/usr/bin/env bash
# TODO A81 (2026-11-12): speed-head experts vs vf_ctrl on the 50 obstacle routes of Bench2Drive-220, CARLA Epic, one RTX 4090 box.
# The arm is vf_ctrl's exact recipe (E15 epoch 15 -> 30, leak-filtered data, pooled overlay cache, seed 0) plus --speed_moe_experts 16.
# vf_ctrl itself is not retrained: its logged checkpoint is the paired baseline, evaluated on this host with the same three lanes.
# usage: scripts/ops/a81_box.sh STEP         status lines -> /workspace/a81/status (setup/audit also write /workspace/vf/status)
#   setup      deps, data, TF++ weights, CARLA (vision_fix_box.sh setup; the data download is the long part)
#   base_lanes vf_ctrl on the 3 obstacle lanes (needs CARLA only, so it runs while the data downloads)
#   audit      drop Town12/13 route folders that sit on an evaluation route (vision_fix_box.sh audit)
#   cache      pooled 4x4 cache with the route overlay (the only cache vf_ctrl used)
#   train      the A81 arm (foreground)
#   loadcheck  the CARLA agent loads the arm, the experts are live, one step runs
#   arm_lanes  the arm on the same 3 lanes (refuses while any eval lane still runs)
#   all        setup (base_lanes starts as soon as CARLA is ready) -> audit -> cache -> train -> loadcheck -> wait for the baseline -> arm_lanes
# Files the deploy step (scripts/ops/a81_deploy.sh, run on the PC) puts on the box:
#   /workspace/checkpoints/E15/model_epoch_015.pth                       (E15 epoch 15, no optimizer state: the arm restarts AdamW anyway)
#   /workspace/checkpoints/vf_ctrl/{model_epoch_030.pth,frozen_backbone.pth,run_config.json}
#   /workspace/obst_lanes.txt  (3 lines: the 50 obstacle routes, the lists used on 2026-11-09 and 11-11)
#   /workspace/vf/bench2drive220.xml (for the leak audit)
set -uo pipefail
mkdir -p /workspace/a81; S=/workspace/a81/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis; export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python; PYC=/workspace/venv_carla/bin/python; [ -x $PYC ] || PYC=$PY
[ "${A81_PY:-}" = main ] && PYC=$PY   # steps that run while CARLA's venv is still being built must not pick up a half-installed venv
D=/workspace/dataset/wor_trajectories
TFPP=/workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
ARM=a81_smoe16
EXPERTS=${A81_EXPERTS:-16}; HIDDEN=${A81_HIDDEN:-32}
F="CARLA_QUALITY=Epic"
COMMON="--data_dir $D --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 30
 --resume_from /workspace/checkpoints/E15/model_epoch_015.pth --route_key route_original --weights_path $TFPP --freeze_backbone 1 --pretrained 1
 --route_points 4 --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4
 --lr_backbone 1e-4 --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --use_mlflow 1 --kill_stale 0 --seed 0 --experiment_name a81_speed_moe
 --route_overlay 1 --vision_grid 4 --num_workers 8"

lanes() {  # LABEL_PREFIX CKPT_UNDER_CHECKPOINTS
  local IDS=/workspace/obst_lanes.txt
  [ "$(wc -l < $IDS)" -ge 3 ] || { mark "LANES_NO_ROUTE_FILE"; return 1; }
  if pgrep -f "queue_A9c.sh 20[0-2]0 " > /dev/null; then mark "LANES_BUSY $1"; return 1; fi
  mkdir -p /workspace/evlogs
  for k in 1 2 3; do
    L=$(sed -n ${k}p $IDS); P=$((1990 + 10 * k))
    (setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "CKE:${1}_obst_L$k:$2:$L:$F" > /workspace/evlogs/${1}_L$k.out 2>&1 < /dev/null &)
  done
  mark "LANES_STARTED $1"
}
wait_lanes_idle() { while pgrep -f "queue_A9c.sh 20[0-2]0 " > /dev/null; do sleep 60; done; }

case $1 in
setup)
  mark SETUP_START; bash scripts/ops/vision_fix_box.sh setup; mark "SETUP_END $(tail -n 6 /workspace/vf/status | tr '\n' ' ')" ;;
base_lanes)
  until grep -q CARLA_PROV_END /workspace/vf/status 2>/dev/null; do sleep 30; done
  [ -f /workspace/checkpoints/vf_ctrl/model_epoch_030.pth ] && [ -f /workspace/checkpoints/vf_ctrl/frozen_backbone.pth ] || { mark BASE_CKPT_MISSING; exit 1; }
  lanes vfctrl vf_ctrl/model_epoch_030.pth ;;
audit)
  bash scripts/ops/vision_fix_box.sh audit; grep -q AUDIT_DONE /workspace/vf/status && mark "AUDIT_DONE $(grep AUDIT_DONE /workspace/vf/status | tail -1)" || { mark AUDIT_FAILED; exit 1; } ;;
cache)
  mkdir -p /workspace/pooled_ov1
  $PYC scripts/training/build_pooled_cache.py --data_dir $D --out_dir /workspace/pooled_ov1 --backbone regnety_032 --weights_path $TFPP \
    --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 1 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 \
    --device cuda:0 --shard 0 --num_shards 1 --num_workers $(( $(nproc) > 20 ? 16 : $(nproc) - 4 )) --batch_size 128 \
    > /workspace/a81/cache.log 2>&1 && ls /workspace/pooled_ov1/*.json > /dev/null && mark CACHE_DONE || { mark CACHE_FAILED; exit 1; } ;;
train)
  grep -q AUDIT_DONE /workspace/vf/status || { mark "TRAIN_REFUSED audit first"; exit 1; }
  P1=$(ls /workspace/pooled_ov1/*.json | head -1); P1=${P1%.json}
  mark TRAIN_START
  CUDA_VISIBLE_DEVICES=0 $PYC -u scripts/training/train_wor.py $COMMON --pooled_cache $P1 \
    --speed_moe_experts $EXPERTS --speed_moe_hidden $HIDDEN --speed_moe_top_k 2 --moe_aux_weight 0.01 \
    --run_label $ARM --save_dir /workspace/checkpoints/$ARM > /workspace/a81/$ARM.log 2>&1
  rc=$?; mark "TRAIN_END $ARM rc=$rc split=$(grep -aoE "'speed_moe_split': [0-9.na]+" /workspace/a81/$ARM.log | tail -1)"; exit $rc ;;
loadcheck)
  CKF=${A81_CKPT:-/workspace/checkpoints/$ARM/model_epoch_030.pth}
  $PYC - > /workspace/a81/loadcheck.log 2>&1 <<EOF && mark "LOADCHECK_OK $(tail -1 /workspace/a81/loadcheck.log)" || { mark LOADCHECK_FAILED; exit 1; }
import numpy as np, torch
from src.agents.wor_agent import WorldOnRailsAgent
a = WorldOnRailsAgent(checkpoint_path="$CKF", policy_arch="qwen30m", backbone_name="regnety_032")
assert a.net.speed_moe is not None and a.net.speed_moe.num_experts == $EXPERTS, "the agent rebuilt the model without the speed experts"
assert float(a.net.speed_moe.w2.abs().sum()) > 0, "speed experts are still at their zero init: training did not reach them"
x = np.random.default_rng(0).integers(0, 255, (512, 1024, 3), dtype=np.uint8)
r = np.array([[2, 0], [4, 0], [6, 0], [8, 0]], np.float32)
for v in (0.0, 6.0):
    c = a.run_step({"rgb_front": x, "speed": v, "command": 3, "route": r})
print(f"experts {a.net.speed_moe.num_experts} control {c.steer:.3f} {c.throttle:.3f} {c.brake:.3f}")
EOF
  ;;
arm_lanes)
  [ -f /workspace/checkpoints/$ARM/model_epoch_030.pth ] || { mark ARM_CKPT_MISSING; exit 1; }
  lanes a81 $ARM/model_epoch_030.pth ;;
all)
  # Hard time budget (user, 2026-11-12: the whole run must finish in 7 h of box time). From this launch:
  #   training is cut at A81_TRAIN_BUDGET_S (default 3 h) and the newest saved epoch is evaluated (save_freq 1);
  #   at A81_HARD_DEADLINE_S (default 6.3 h) every lane and the pipeline are killed and HARD_DEADLINE is marked.
  # GPU work (cache, training) runs while CARLA installs (CPU and network only); evaluation starts after training, so CARLA lanes
  # and the trainer never share the 24 GB. Each lane runs the baseline, then the arm, on the same routes (same host, same flags).
  T0=$(date +%s); TRAIN_END_AT=$((T0 + ${A81_TRAIN_BUDGET_S:-10800})); HARD_AT=$((T0 + ${A81_HARD_DEADLINE_S:-22680}))
  mark "ALL_START train_cut_at=$(date -u -d @$TRAIN_END_AT +%H:%M) hard_deadline=$(date -u -d @$HARD_AT +%H:%M) UTC"
  ( while [ $(date +%s) -lt $HARD_AT ]; do sleep 60; done
    mark "HARD_DEADLINE: killing lanes and pipeline"; pkill -f "[q]ueue_A9c.sh"; pkill -f "[b]2d_guardian"
    pkill -f "[l]eaderboard_evaluator"; pkill -f "[C]arlaUE4"; pkill -f "[t]rain_wor.py"; pkill -f "[v]ision_fix_box.sh"
    # last: this watchdog is itself a copy of the 'all' process and dies here
    pkill -f "^bash scripts/ops/a81_box.sh" ) &
  ( bash $0 setup ) &
  until grep -qE "DATA_DONE|DATA_FAILED" /workspace/vf/status 2>/dev/null; do sleep 30; done
  grep -q DATA_DONE /workspace/vf/status || { mark "STOP data download failed"; exit 1; }
  until grep -qE "TFPP_DONE|TFPP_FAILED" /workspace/vf/status && grep -q PIP_DONE /workspace/vf/status; do sleep 20; done
  grep -q TFPP_DONE /workspace/vf/status || { mark "STOP TF++ weights failed"; exit 1; }
  A81_PY=main bash $0 audit && A81_PY=main bash $0 cache || { mark "STOP audit or cache failed"; exit 1; }
  ( A81_PY=main bash $0 train ) & TP=$!
  while kill -0 $TP 2>/dev/null; do
    if [ $(date +%s) -ge $TRAIN_END_AT ]; then pkill -f "[t]rain_wor.py"; mark TRAIN_CUT_AT_BUDGET; break; fi
    sleep 60
  done
  CK=$(ls -1 /workspace/checkpoints/$ARM/model_epoch_*.pth 2>/dev/null | sort -V | tail -1)
  [ -n "$CK" ] || { mark "STOP no arm checkpoint"; exit 1; }
  mark "EVAL_CKPT $CK"
  until grep -q CARLA_PROV_END /workspace/vf/status 2>/dev/null; do sleep 30; done
  A81_CKPT=$CK bash $0 loadcheck || { mark "STOP loadcheck failed"; exit 1; }
  IDS=/workspace/obst_lanes.txt; mkdir -p /workspace/evlogs; REL=$ARM/$(basename $CK)
  for k in 1 2 3; do
    L=$(sed -n ${k}p $IDS); P=$((1990 + 10 * k))
    (setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "CKE:vfctrl_obst_L$k:vf_ctrl/model_epoch_030.pth:$L:$F" "CKE:a81_obst_L$k:$REL:$L:$F" \
      > /workspace/evlogs/lane$k.out 2>&1 < /dev/null &)
  done
  mark "LANES_STARTED baseline then $REL"
  wait_lanes_idle; mark ALL_LANES_DONE ;;
*) echo "unknown step $1"; exit 2 ;;
esac
