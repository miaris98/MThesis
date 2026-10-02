#!/usr/bin/env bash
# CARLA box day 2026-10-02 (S-113, TODO A14): more seeds of arm J on arm J's exact recipe (only --seed changes).
# Seed 0 = carla_armJ_ft_obst (+17.5 vs E e15 on obstacle routes), seed 1 = carla_armJ_ft_obst_s1 (+4.1 e18 / +9.2 e20, S-112).
# usage: launch_armJ_seeds.sh SEED_GPU0 [SEED_GPU1]   e.g. launch_armJ_seeds.sh 2 3
# Needs the same inputs as launch_armJ2_armK.sh (arm J data, TF++ weights, carla_armE_aug1_hires e15 + frozen_backbone).
cd /workspace/MThesis
COMMON="--data_dir /workspace/dataset/wor_trajectories --backbone regnety_032 --policy_arch qwen30m --img_size 288x768
 --batch_size 256 --epochs 50 --stop_epoch 20 --resume_from /workspace/checkpoints/carla_armE_aug1_hires/model_epoch_015.pth
 --route_key route_original --weights_path /workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
 --freeze_backbone 1 --pretrained 1 --route_overlay 1 --route_points 4 --use_augmented_camera 1 --lateral_loss_weight 3.0
 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4 --lr_backbone 1e-4 --vision_grid 4 --crop_bottom_frac 0.0
 --save_freq 1 --val_every 5 --num_workers 16 --use_mlflow 1 --kill_stale 0"
run() {  # GPU LABEL extra-args
  ( CUDA_VISIBLE_DEVICES=$1 setsid nohup /venv/main/bin/python -u scripts/training/train_wor.py $COMMON $3 \
      --run_label $2 --save_dir /workspace/checkpoints/$2 > /workspace/$2.log 2>&1 < /dev/null & )
  sleep 5
}
G=0
for S in "$@"; do run $G carla_armJ_ft_obst_s$S "--seed $S"; G=$((G + 1)); done
sleep 240
for S in "$@"; do L=carla_armJ_ft_obst_s$S; echo "== $L"; grep -aE "Resumed|Validation split|Error|Traceback" /workspace/$L.log | tail -3; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
