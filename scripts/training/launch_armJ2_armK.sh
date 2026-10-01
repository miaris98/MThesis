#!/usr/bin/env bash
# CARLA box day (S-111 plan): two fine-tunes of arm E e15 on arm J's data (6 towns + Town12/13 obstacle archives,
# route_original), one per GPU, each on arm J's exact recipe (run_config of carla_armJ_ft_obst) with ONE change:
#   GPU 0  carla_armJ_ft_obst_s1   --seed 1          is arm J's +15 on obstacle routes one lucky run? (second seed)
#   GPU 1  carla_armK_swerve10     --swerve_frac 0.1 TODO A16: swerve frames drawn as 10% of each epoch (arm J ~natural)
# Needs: /workspace/dataset/wor_trajectories (download_pdm_lite.py with arm J's --archive-regex), the TF++ weights,
# and checkpoints/carla_armE_aug1_hires/{model_epoch_015,frozen_backbone}.pth + run_config.json.
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
run 0 carla_armJ_ft_obst_s1 "--seed 1"
run 1 carla_armK_swerve10 "--seed 0 --swerve_frac 0.1"
sleep 240
for L in carla_armJ_ft_obst_s1 carla_armK_swerve10; do echo "== $L"; grep -aE "Resumed|Validation split|Swerve sampler|Error|Traceback" /workspace/$L.log | tail -4; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
