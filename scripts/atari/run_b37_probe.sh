#!/usr/bin/env bash
# TODO B37 step 1 (S-118/S-125): a 2,000-step probe per mixer setting, no scores: does the GTrXL mixer's q/k/v move (|dW|/|W0| > 5% by 2k updates)?
# One setting per GPU; the audit line `[B37 audit] upd N: |dW|/|W0| attn_out .. qkv .. ffn .. gate_w .. | grad ..` is printed every 250 updates (and goes to MLflow as audit/*).
#
# usage: run_b37_probe.sh GPU NAME [train_ez_offpolicy.py flags ...]
#   run_b37_probe.sh 0 adamw1e-4 --mixer-optimizer adamw --mixer-lr 1e-4
#   run_b37_probe.sh 3 adamw3e-4_open --mixer-optimizer adamw --mixer-lr 3e-4 --mixer-bg-init 0
cd /workspace/MThesis
GPU=$1; NAME=$2; shift 2
L=/workspace/MThesis/results/100k_benchmark/S058_ezv2_match; mkdir -p $L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true
LABEL=B37probe_${NAME}
mkdir -p $L/$LABEL/checkpoints
CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
  --trunk gtrxl --norm batch --priority-alpha 0 --seed 0 --mixer-weight-decay "${MIXER_WD:-0.05}" \
  --total-steps ${TOTAL_STEPS:-2000} --schedule-steps 100000 --eval-interval 100000 --eval-episodes 1 --eval-mode deferred \
  --graph-search --audit-every 250 --run-label $LABEL --log-dir $L "$@" >> /workspace/$LABEL.log 2>&1
echo "$(date -u) $LABEL rc=$?" >> /workspace/b37_probe.done
