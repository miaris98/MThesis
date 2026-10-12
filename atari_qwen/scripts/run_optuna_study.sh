#!/usr/bin/env bash
# Launch Optuna hyperparameter tuning with background Hugging Face sync
set -e

ENV_ID=${1:-"BreakoutNoFrameskip-v4"}
N_TRIALS=${2:-15}
NUM_ENVS=${3:-16}
STEPS_PER_TRIAL=${4:-400000}

echo "================================================================"
echo "    Starting Qwen Atari Optuna Study & HF Sync Pipeline        "
echo "================================================================"
echo "  Environment:       $ENV_ID"
echo "  Trials:            $N_TRIALS"
echo "  Parallel Envs:     $NUM_ENVS"
echo "  Steps per Trial:   $STEPS_PER_TRIAL"
echo "  Persistent DB:     results/atari_qwen/optuna.db"
echo "================================================================"

if [ -d "/venv/main" ]; then
    source /venv/main/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results/atari_qwen

# 1. Kill any existing sync loops
pkill -f hf_sync.py || true

# 2. Launch recurring Hugging Face sync (every 5 mins) in background
echo "--> Launching background Hugging Face Hub sync loop (300s)..."
nohup python atari_qwen/training/hf_sync.py \
    --local-dir results/atari_qwen \
    --repo-id Miaris/mthesis-atari-qwen \
    --loop-seconds 300 \
    --game-id "$ENV_ID" \
    --preset optuna > results/atari_qwen/hf_sync.log 2>&1 &
echo "✓ HF Sync loop PID: $!"

# 3. Launch Optuna study
echo "--> Launching Optuna optimization..."
python atari_qwen/training/optimize_optuna.py \
    --env-id "$ENV_ID" \
    --n-trials "$N_TRIALS" \
    --timesteps-per-trial "$STEPS_PER_TRIAL" \
    --num-envs "$NUM_ENVS" \
    --storage sqlite:///results/atari_qwen/optuna.db \
    --study-name "qwen_${ENV_ID}_tuning" 2>&1 | tee -a results/atari_qwen/optuna.log
