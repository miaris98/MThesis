#!/usr/bin/env bash
# V45 (2026-11-06, user approved): continue the two vision arms from epoch 30 to epoch 45 on ONE RTX 5090 box, same seed and schedule.
#   s2d   fix 3 (stride-16 space-to-depth 6x16 tokens, 20% lateral masking, live encoder)  resume: vf_s2d/model_epoch_030.pth
#   ctrl  the baseline (overlay, pooled 4x4 cache)                                          resume: vf_ctrl/model_epoch_030.pth
# Same recipe as scripts/ops/vision_fix_box.sh (arm J on the leak-filtered data); only --resume_from and --stop_epoch change.
# The pooled cache is built on the box for the overlay variant only (the ctrl arm reads it; the s2d arm uses the live encoder).
# usage: scripts/ops/v45_box.sh STEP      status lines -> $WORKSPACE/v45/status
#   audit   drop every Town12/13 route folder that sits on a Bench2Drive-220 evaluation route (moved, not deleted)
#   cache   pooled 4x4 cache with the overlay (one GPU)
#   train   s2d and ctrl, both on GPU 0, concurrently (the cached ctrl arm is light; check VRAM before the second starts)
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
mkdir -p $WORKSPACE/v45; S=$WORKSPACE/v45/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis; export PYTHONPATH=$WORKSPACE/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python
D=$WORKSPACE/dataset/wor_trajectories
TFPP=$WORKSPACE/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
COMMON="--data_dir $D --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 45
 --route_key route_original --weights_path $TFPP --freeze_backbone 1 --pretrained 1 --route_points 4
 --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4 --lr_backbone 1e-4
 --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --use_mlflow 1 --kill_stale 0 --seed 0 --experiment_name vision_fix"
case $1 in
audit)
  $PY scripts/analysis/route_overlap_audit.py --eval_xml $WORKSPACE/vf/bench2drive220.xml --data_dir $D --out_json $WORKSPACE/v45/leak_traj.json \
    > $WORKSPACE/v45/audit.log 2>&1 || { mark AUDIT_FAILED; exit 1; }
  $PY - >> $WORKSPACE/v45/audit.log 2>&1 <<'PY' || { mark AUDIT_MOVE_FAILED; exit 1; }
import json, os, shutil
D, X = "/workspace/dataset/wor_trajectories", "/workspace/dataset_excluded"
rows = json.load(open("/workspace/v45/leak_traj.json"))["routes"]
moved = 0
for r in rows:
    if r["leak"]:
        dst = os.path.join(X, os.path.relpath(r["dir"], D)); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(r["dir"], dst); moved += 1
print(f"moved {moved} of {len(rows)} route folders to {X}")
PY
  mark "AUDIT_DONE $(grep -oE 'moved [0-9]+ of [0-9]+' $WORKSPACE/v45/audit.log)" ;;
cache)
  mkdir -p $WORKSPACE/pooled_ov1
  $PY scripts/training/build_pooled_cache.py --data_dir $D --out_dir $WORKSPACE/pooled_ov1 --backbone regnety_032 --weights_path $TFPP \
    --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 1 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 \
    --device cuda:0 --shard 0 --num_shards 1 --num_workers 16 --batch_size 128 > $WORKSPACE/v45/cache_ov1.log 2>&1 \
    && mark CACHE_OV1_DONE || mark CACHE_OV1_FAILED ;;
train)
  grep -q AUDIT_DONE $S || { echo "audit first"; exit 1; }
  S2D="--route_overlay 1 --token_mode s2d --s2d_patch 3 --token_mask 0.2 --vision_grid 6x16 --encoder_channels_last 1 --num_workers 24"
  P1=$(ls $WORKSPACE/pooled_ov1/*.json | head -1); P1=${P1%.json}
  CTRL="--route_overlay 1 --vision_grid 4 --num_workers 8 --pooled_cache $P1"
  run() {  # LABEL RESUME extra-args (one run, foreground)
    CUDA_VISIBLE_DEVICES=0 $PY -u scripts/training/train_wor.py $COMMON --resume_from $2 $3 --run_label $1 --save_dir $WORKSPACE/checkpoints/$1 \
      > $WORKSPACE/v45/$1.log 2>&1; mark "TRAIN_END $1 rc=$?"
  }
  ( run v45_s2d $WORKSPACE/checkpoints/vf_s2d/model_epoch_030.pth "$S2D" ) &
  sleep 60
  ( run v45_ctrl $WORKSPACE/checkpoints/vf_ctrl/model_epoch_030.pth "$CTRL" ) &
  wait ;;
esac
