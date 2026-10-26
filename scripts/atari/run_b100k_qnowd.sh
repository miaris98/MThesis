#!/usr/bin/env bash
# TODO B100K (2026-10-24, user go): three seeds of the qnowd recipe (Qwen-small mixer, mixer weight decay 0, B62 anti-collapse
# fixes) on the full 100k schedule (100,000 online + 20,000 offline updates), three processes on one RTX 5090 under NVIDIA MPS
# (B64: 0.31 s per update each vs 0.40 s without MPS, 0.145 s alone). Training only: a checkpoint every 10k env steps
# (--eval-mode deferred); the scores come afterwards from eval_loop_stats.py (screen) and eval_ez_checkpoints.py (paper protocol).
# Resumes from checkpoint_latest.pt + replay_latest.npz of the same label (the pair is saved together every 2,500 updates).
# usage: run_b100k_qnowd.sh SEEDS...          results: /workspace/b100k/out/<label>, logs: /workspace/b100k/logs/<label>.log
cd /workspace/MThesis
L=/workspace/b100k/out; G=/workspace/b100k/logs; mkdir -p $L $G
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps_pipe /tmp/mps_log
pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
ARM="--trunk qwen --mixer-size small --mixer-out-init-std 0 --mixer-weight-decay 0 --stagnation-explore 300 --stagnation-burst 30 \
  --policy-target-smoothing 0.05 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0"
run_one() {  # seed
  local seed=$1 label=b100k_qnowd_s$1
  mkdir -p $L/$label/checkpoints
  for try in 1 2 3 4 5; do
    local res=""; [ -f $L/$label/checkpoints/checkpoint_latest.pt ] && res="--resume-from $L/$label/checkpoints/checkpoint_latest.pt"
    echo "$(date -u +%H:%M) start $label try $try ${res:-fresh}" >> $G/$label.log
    /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py $ARM --seed $seed \
      --total-steps 100000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
      --graph-search --run-label $label --log-dir $L $res >> $G/$label.log 2>&1
    local rc=$?; echo "$(date -u +%H:%M) $label exited rc=$rc" >> $G/$label.log
    [ $rc -eq 0 ] && break; sleep 60
  done
}
for s in "$@"; do run_one $s & sleep 5; done
wait
echo "$(date -u +%H:%M) B100K DONE $*" > $G/done.marker
