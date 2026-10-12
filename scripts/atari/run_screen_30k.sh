#!/usr/bin/env bash
# Atari trunk screen at 30k env steps (TODO B35, S-111 plan B): GTrXL vs ResNet, 8 seeds each.
# A 30k run is an exact prefix of the 100k run with the same seed (schedules follow --schedule-steps 100k), so the
# env30000 evals of the earlier 100k runs count as screen points (GTrXL s0, s3; ResNet s0-s3) and are not rerun.
# Training uses --eval-mode deferred; one eval_ez_checkpoints.py --watch per run scores its checkpoints in parallel.
# usage: run_screen_30k.sh GPU TRUNK:SEED [TRUNK:SEED ...]   (runs the listed runs one after another on that GPU)
# smoke test: TOTAL_STEPS=3000 EVAL_INTERVAL=1500 LABEL_SUFFIX=_smoke run_screen_30k.sh 0 resnet:9
cd /workspace/MThesis
GPU=$1; shift
TOTAL_STEPS=${TOTAL_STEPS:-30000}; EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
L=/workspace/MThesis/results/100k_benchmark/S058_ezv2_match; mkdir -p $L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
for spec in "$@"; do
  trunk=${spec%%:*}; seed=${spec#*:}
  case $trunk in gtrxl) p=S058h;; resnet) p=S058i;; *) echo "bad trunk $trunk"; exit 1;; esac
  LABEL=${p}_${trunk}_30k_s${seed}_uniform${LABEL_SUFFIX:-}
  mkdir -p $L/$LABEL/checkpoints
  echo "$(date -u) start $LABEL on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/eval_ez_checkpoints.py $L/$LABEL --watch \
    >> /workspace/${LABEL}_eval.log 2>&1 &
  EVPID=$!
  for try in 1 2 3; do
    CK=$L/$LABEL/checkpoints/checkpoint_latest.pt; RES=""; [ -f $CK ] && RES="--resume-from $CK"
    CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
      --trunk $trunk --norm batch --priority-alpha 0 --seed $seed \
      --total-steps $TOTAL_STEPS --schedule-steps 100000 --eval-interval $EVAL_INTERVAL --eval-episodes 10 --eval-mode deferred \
      --run-label $LABEL --log-dir $L $RES >> /workspace/$LABEL.log 2>&1
    rc=$?; echo "$(date -u) $LABEL exited rc=$rc"; [ $rc -eq 0 ] && break; sleep 30
  done
  # the evaluator exits by itself once the final checkpoint is scored; the next run's training starts now
done
wait
echo "$(date -u) screen lane GPU $GPU done: $*"
