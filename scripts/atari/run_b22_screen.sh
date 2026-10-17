#!/usr/bin/env bash
# TODO B22 step 1 (2026-10-17): SimNorm on the latent, 10k screen. Same 100k-schedule recipe as the S058 10k snapshots (uniform replay,
# 4 envs, 16 sims, batch 256), stopped at 10,000 env steps. SIMNORM=0 is the control; SIMNORM=8 is TD-MPC2's group size.
# usage: run_b22_screen.sh GPU TRUNK SEED SIMNORM       (TRUNK = resnet | gtrxl)
# label: B22_<trunk>_10k_s<seed>_sn<simnorm>; results under /workspace/b22/out/<label>
cd /workspace/MThesis
GPU=$1; TRUNK=$2; SEED=$3; SIMNORM=$4
LABEL=B22_${TRUNK}_10k_s${SEED}_sn${SIMNORM}
L=/workspace/b22/out; mkdir -p $L/$LABEL/checkpoints /workspace/b22/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis
for try in 1 2 3; do
  CK=$L/$LABEL/checkpoints/checkpoint_latest.pt; RES=""; [ -f $CK ] && RES="--resume-from $CK"
  echo "$(date -u) start $LABEL try $try ${RES:-fresh}" >> /workspace/b22/logs/$LABEL.log
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
    --trunk $TRUNK --norm batch --priority-alpha 0 --seed $SEED --simnorm $SIMNORM \
    --total-steps 10000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
    --graph-search --run-label $LABEL --log-dir $L $RES >> /workspace/b22/logs/$LABEL.log 2>&1
  rc=$?; echo "$(date -u) $LABEL exited rc=$rc" >> /workspace/b22/logs/$LABEL.log; [ $rc -eq 0 ] && break; sleep 30
done
echo SCREEN_DONE $LABEL
