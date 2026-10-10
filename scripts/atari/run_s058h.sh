#!/usr/bin/env bash
# Atari-100k Breakout, the S058h recipe (GTrXL, uniform replay, 4 envs, 16 sims, batch 256, lr 0.2 schedule over 100k) for one seed on one GPU,
# resumable: reruns continue from checkpoint_latest.pt. CUDA-graphed search (bit-identical to eager) and a deferred evaluator that scores every
# env{N} snapshot on the training protocol (10 episodes). The paper-protocol eval of the final checkpoint is a separate step (eval_ez_checkpoints.py
# --episodes 60 --sims 64 --env-steps 100000 --graph).
# usage: run_s058h.sh GPU SEED [TOTAL_STEPS]          label S058h_gtrxl_100k_s<SEED>_uniform (same label as the 2026-09-28/29 runs, so a restored folder resumes)
cd /workspace/MThesis
GPU=$1; SEED=$2; TOTAL=${3:-100000}
L=/workspace/MThesis/results/100k_benchmark/S058_ezv2_match; mkdir -p $L
LABEL=S058h_gtrxl_100k_s${SEED}_uniform; mkdir -p $L/$LABEL/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true
CUDA_VISIBLE_DEVICES=$GPU setsid nohup /venv/main/bin/python -u atari_qwen/training/eval_ez_checkpoints.py $L/$LABEL --watch --graph >> /workspace/${LABEL}_eval.log 2>&1 < /dev/null &
for try in 1 2 3; do
  CK=$L/$LABEL/checkpoints/checkpoint_latest.pt; RES=""; [ -f $CK ] && RES="--resume-from $CK"
  echo "$(date -u) start $LABEL try $try ${RES:-fresh}" >> /workspace/$LABEL.log
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
    --trunk gtrxl --norm batch --priority-alpha 0 --seed $SEED --total-steps $TOTAL --schedule-steps 100000 \
    --eval-interval 10000 --eval-episodes 10 --eval-mode deferred --graph-search --run-label $LABEL --log-dir $L $RES >> /workspace/$LABEL.log 2>&1
  rc=$?; echo "$(date -u) $LABEL exited rc=$rc" >> /workspace/$LABEL.log; [ $rc -eq 0 ] && break; sleep 30
done
