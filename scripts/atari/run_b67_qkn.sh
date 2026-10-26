#!/usr/bin/env bash
# TODO B67 (2026-10-26, user go: "lets go with that", QK-norm): the CARLA recipe for the mixer (--mixer-optimizer adamw_carla, AdamW 3e-4,
# warm-up then cosine) WITH Qwen3-style QK-norm in every mixer block (--mixer-qk-norm). Without QK-norm the same recipe collapsed the
# attention by update 8,004 (B66 b66_fix_s1: entropy 0.002, largest logit 9,249; training return at random level by env 19k).
# No B62 flags, so it differs from b66_fix_s1 only in the QK-norm. One run; under NVIDIA MPS when other runs share the GPU.
#   SIZE small = the 2x128 mixer of the B66 screens; SIZE 30m = CARLA's qwen30m shape (the claim arm, one run per 32 GB card).
# The edited block must also be rerun on CARLA (E15 with --qk_norm 1, user rule 2026-10-26) before any Atari number is a claim.
# Training only (--eval-mode deferred); score afterwards with scripts/atari/final_evals.sh (paper protocol + sticky 0.25).
# usage: run_b67_qkn.sh SEED [SIZE]     results: /workspace/b66/out/b67_qkn_<size>_s<seed>, logs: /workspace/b66/logs/<label>.log
cd /workspace/MThesis
L=/workspace/b66/out; G=/workspace/b66/logs; mkdir -p $L $G
S=${1:?seed}; SIZE=${2:-small}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps_pipe /tmp/mps_log
pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
ARM="--trunk qwen --mixer-size $SIZE --mixer-out-init-std 0 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0 --mixer-optimizer adamw_carla --mixer-lr 3e-4 --mixer-grad-clip 5.0 --mixer-qk-norm"
label=b67_qkn_${SIZE}_s$S
mkdir -p $L/$label/checkpoints
for try in 1 2 3 4 5; do
  res=""; [ -f $L/$label/checkpoints/checkpoint_latest.pt ] && res="--resume-from $L/$label/checkpoints/checkpoint_latest.pt"
  echo "$(date -u +%H:%M) start $label try $try ${res:-fresh}" >> $G/$label.log
  /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py $ARM --seed $S \
    --total-steps 100000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
    --graph-search --run-label $label --log-dir $L $res >> $G/$label.log 2>&1
  rc=$?; echo "$(date -u +%H:%M) $label exited rc=$rc" >> $G/$label.log
  [ $rc -eq 0 ] && break; sleep 60
done
echo "$(date -u +%H:%M) B67 DONE $label" > $G/done_$label.marker
