#!/usr/bin/env bash
# TODO B66 step 2 (2026-10-26): the claim arm. CARLA's own Qwen network (--mixer-size 30m = qwen30m: 8 blocks, width 512, 8 heads, FFN 1536)
# as the EZ-V2 mixer, trained with CARLA's recipe (--mixer-optimizer adamw_carla), no B62 flags. One run per 32 GB card (a 30m run uses
# about 20.5 GB; B61 2026-10-23), so no MPS. About 33-35 h per 100k seed on an RTX 5090 (B65 step 1).
# Training only (--eval-mode deferred); score afterwards with eval_ez_checkpoints.py --updates 120000 --episodes 60 --sims 16 --protocol paper_ep60_sim16.
# Resumes from checkpoint_latest.pt + replay_latest.npz of the same label (saved together every 2,500 updates, incl. the mixer AdamW state).
# usage: run_b66_30m.sh SEED [GPU]      results: /workspace/b66/out/b66_30m_s<seed>, logs: /workspace/b66/logs/b66_30m_s<seed>.log
cd /workspace/MThesis
L=/workspace/b66/out; G=/workspace/b66/logs; mkdir -p $L $G
S=${1:?seed}; GPU=${2:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis CUDA_VISIBLE_DEVICES=$GPU
ARM="--trunk qwen --mixer-size 30m --mixer-out-init-std 0 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0 --mixer-optimizer adamw_carla --mixer-lr 3e-4 --mixer-grad-clip 5.0"
label=b66_30m_s$S
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
echo "$(date -u +%H:%M) B66 30m DONE seed $S" > $G/done_30m_s$S.marker
