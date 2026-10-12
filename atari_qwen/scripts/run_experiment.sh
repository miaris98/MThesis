#!/usr/bin/env bash
# Quick launcher for Atari Qwen training experiments
set -e

ENV_ID=${1:-"BreakoutNoFrameskip-v4"}
PRESET=${2:-"tiny"}
NUM_ENVS=${3:-16}
TOTAL_STEPS=${4:-10000000}

echo "Launching Atari Qwen PPO Training..."
echo "  Environment:   $ENV_ID"
echo "  Qwen Preset:   $PRESET"
echo "  Parallel Envs: $NUM_ENVS"
echo "  Total Steps:   $TOTAL_STEPS"

if [ -d "/venv/main" ]; then
    source /venv/main/bin/activate
fi

python atari_qwen/training/train_ppo.py \
    --env-id "$ENV_ID" \
    --preset "$PRESET" \
    --num-envs "$NUM_ENVS" \
    --total-timesteps "$TOTAL_STEPS" \
    --learning-rate 2.5e-4 \
    --exp-name "atari_qwen_${PRESET}"
