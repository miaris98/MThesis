#!/usr/bin/env bash
# Trains the three closed-loop arms at one --seed, replicating the seed-0 configuration
# exactly (read back from wor_cl_arms/*/run_config.json) so the resulting checkpoints are
# drawn from the same distribution as the ones behind 13.31.
#
# usage: run_seed_replication_arms.sh <seed>
#
# WHY: 13.31's closed-loop comparison rests on a single training seed, while the open-loop
# result it follows from was replicated at seeds 0/1/2. Until these exist, the driving claim
# is n=1 in training seed and a reader can fairly discount it.
#
# --split_seed stays pinned at 0 while --seed varies: the held-out route split must be
# identical across seeds, otherwise the replication measures the split as well as the seed
# (the methodological rule from 13.17).
#
# Training is GPU-bound (99-100% GPU, ~1.1s data wait per 130s epoch) while closed-loop
# evaluation is CPU-bound, so this is intended to run *alongside* an evaluation rather than
# instead of one - see challenges_05 5.9.
set -x
SEED="${1:?usage: run_seed_replication_arms.sh <seed>}"
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=/workspace/checkpoints/wor_cl_arms
DATA=/workspace/dataset/wor_trajectories

COMMON=(--data_dir "$DATA" --backbone resnet34 --pretrained 1 --freeze_backbone 1
        --epochs 15 --batch_size 32 --lr_backbone 1e-4 --lr_heads 3e-4 --num_workers 3
        --cache_decoded 1 --wp_loss_weight 1.0 --q_loss_weight 0.0 --lateral_loss_weight 3.0
        --route_points 4 --grad_clip 5.0 --warmup_frac 0.05 --decay_gates_and_norms 0
        --val_split 0.15 --split_seed 0 --num_folds 1 --fold 0 --seed "$SEED"
        --experiment_name WoR_ClosedLoop_Arms --use_mlflow 1)

run_arm () {
  local label="$1" arch="$2" heading="$3" curv="$4" port="$5"
  /venv/main/bin/python train_wor.py "${COMMON[@]}" \
    --policy_arch "$arch" --vision_grid 4 --pool_vision 0 \
    --heading_loss_weight "$heading" --curvature_loss_weight "$curv" \
    --mlflow_port "$port" \
    --run_label "$label" --save_dir "$OUT/$label" \
    > "/workspace/train_${label}.log" 2>&1 &
  # Stagger the launches: three processes creating the same MLflow experiment at the same
  # instant raced once already, and the loser fell back to TensorBoard. Harmless, but it
  # costs nothing to avoid.
  sleep 8
}

run_arm "cnn_s${SEED}"              cnn     0.0 0.0 $((10310 + SEED * 10))
run_arm "qwen30m_baseline_s${SEED}" qwen30m 0.0 0.0 $((10311 + SEED * 10))
run_arm "qwen30m_geom_s${SEED}"     qwen30m 0.5 0.2 $((10312 + SEED * 10))

wait
echo "SEED_${SEED}_ARMS_DONE"
