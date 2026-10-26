#!/usr/bin/env bash
# TODO B66 (2026-10-26, user go): attribute the Qwen-vs-EZ-V2 gap. Two Qwen-small 100k runs on one GPU under NVIDIA MPS, both WITHOUT the
# B62 flags (no policy-target smoothing, no stagnation exploration), so they differ from the EZ-V2 champions only in the mixer:
#   b66_fix_s<seed>  the mixer trained with the CARLA recipe of the same Qwen network (--mixer-optimizer adamw_carla)
#   b66_sgd_s<seed>  the mixer trained with EZ-V2's SGD, as in the 2026-10-25 qnowd runs (qnowd minus the B62 flags)
# Against qnowd seed 1 (SGD + B62 flags, final 238.2) this separates the optimiser effect from the B62-flag effect.
# Training only (--eval-mode deferred); scores afterwards from eval_ez_checkpoints.py --updates 120000 (paper protocol).
# Resumes from checkpoint_latest.pt + replay_latest.npz of the same label (saved together every 2,500 updates; the mixer AdamW
# state is in the checkpoint since 2026-10-26).
# usage: run_b66_attrib.sh SEED      results: /workspace/b66/out/<label>, logs: /workspace/b66/logs/<label>.log
cd /workspace/MThesis
L=/workspace/b66/out; G=/workspace/b66/logs; mkdir -p $L $G
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps_pipe /tmp/mps_log
pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
BASE="--trunk qwen --mixer-size small --mixer-out-init-std 0 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0"
FIX="$BASE --mixer-optimizer adamw_carla --mixer-lr 3e-4 --mixer-grad-clip 5.0"
SGD="$BASE --mixer-weight-decay 0"
run_one() {  # label arm-flags seed
  local label=$1 arm=$2 seed=$3
  mkdir -p $L/$label/checkpoints
  for try in 1 2 3 4 5; do
    local res=""; [ -f $L/$label/checkpoints/checkpoint_latest.pt ] && res="--resume-from $L/$label/checkpoints/checkpoint_latest.pt"
    echo "$(date -u +%H:%M) start $label try $try ${res:-fresh}" >> $G/$label.log
    /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py $arm --seed $seed \
      --total-steps 100000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
      --graph-search --run-label $label --log-dir $L $res >> $G/$label.log 2>&1
    local rc=$?; echo "$(date -u +%H:%M) $label exited rc=$rc" >> $G/$label.log
    [ $rc -eq 0 ] && break; sleep 60
  done
}
S=${1:-1}
run_one b66_fix_s$S "$FIX" $S & sleep 5
run_one b66_sgd_s$S "$SGD" $S &
wait
echo "$(date -u +%H:%M) B66 DONE seed $S" > $G/done_s$S.marker
