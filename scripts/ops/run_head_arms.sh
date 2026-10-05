#!/usr/bin/env bash
# TODO A59 head-only arms on the pooled cache (S-126): arm J's recipe (fine-tune of arm E e15 on J's data, epochs 16-20 of a 50-epoch schedule, frozen backbone) read from the cache,
# one change per arm, one arm per call. ~2-4 min per epoch of 427k frames (projection from the real-data step time), so ~15 min per arm.
#
# usage: run_head_arms.sh GPU LABEL SEED [extra train_wor.py flags ...]
#   run_head_arms.sh 0 jc_s0 0                                                   # the control: J's recipe on the cache (must reproduce arm J's closed loop within noise)
#   run_head_arms.sh 1 a39_o05_u0125 0 --speed_lambda_over 0.5 --speed_lambda_under 0.125
#   run_head_arms.sh 2 tsw1 0 --target_speed_loss_weight 1.0
# Resulting checkpoints: /workspace/checkpoints/<LABEL>/model_epoch_020.pth (+ frozen_backbone.pth, run_config.json) = what the lane jobs `CKL:` / `CKE:` take.
cd /workspace/MThesis
GPU=$1; LABEL=$2; SEED=$3; shift 3
export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true
PREFIX=$(ls /workspace/pooled/*.json | head -1); PREFIX=${PREFIX%.json}
TSW=0.2; for a in "$@"; do :; done   # --target_speed_loss_weight in "$@" overrides the 0.2 below (argparse keeps the last occurrence)
CUDA_VISIBLE_DEVICES=$GPU /workspace/venv_carla/bin/python -u scripts/training/train_wor.py \
  --data_dir /workspace/dataset/wor_trajectories --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 20 \
  --resume_from /workspace/checkpoints/carla_armE_aug1_hires/model_epoch_015.pth --route_key route_original \
  --weights_path /workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth --freeze_backbone 1 --pretrained 1 --route_overlay 1 --route_points 4 \
  --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight $TSW --wp_loss_weight 1.0 --lr_heads 3e-4 --lr_backbone 1e-4 --vision_grid 4 \
  --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --num_workers 8 --use_mlflow 1 --kill_stale 0 --seed $SEED --pooled_cache "$PREFIX" \
  --run_label $LABEL --save_dir /workspace/checkpoints/$LABEL --experiment_name A59_head_arms "$@" > /workspace/$LABEL.log 2>&1
echo "$(date -u) $LABEL rc=$?" >> /workspace/head_arms.done
