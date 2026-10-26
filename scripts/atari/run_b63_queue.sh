#!/usr/bin/env bash
# TODO B63 (2026-10-24): why does the Qwen mixer not learn? Three arms at 10k, same recipe except the mixer, all with the B62 collapse fixes
# (stagnation exploration 300/30, policy-target smoothing 0.05) and the mixer diagnostics (--audit-every 1000):
#   gtrxl : GTrXL mixer, the original recipe (SGD, weight decay 1e-4 everywhere, zero-init mixer output)
#   qdef  : Qwen-small mixer, the same recipe (SGD, weight decay 1e-4 on the mixer too, zero-init output)
#   qnowd : Qwen-small mixer, the same recipe but mixer weight decay 0 (S-116: wd shrinks the mixer under SGD)
# usage: run_b63_queue.sh SEEDS...      (runs the three arms of each seed in parallel on GPU 0, one seed after another)
# results: /workspace/b63/out/<label>, logs: /workspace/b63/logs/<label>.log, DONE marker: /workspace/b63/logs/queue.done
cd /workspace/MThesis
L=/workspace/b63/out; G=/workspace/b63/logs; mkdir -p $L $G
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis CUDA_VISIBLE_DEVICES=0
FIX="--stagnation-explore 300 --stagnation-burst 30 --policy-target-smoothing 0.05 --audit-every 1000"
arm_args() {  # arm -> trainer flags
  case $1 in
    gtrxl) echo "--trunk gtrxl" ;;
    qdef)  echo "--trunk qwen --mixer-size small --mixer-out-init-std 0" ;;
    qnowd) echo "--trunk qwen --mixer-size small --mixer-out-init-std 0 --mixer-weight-decay 0" ;;
  esac
}
run_one() {  # arm seed
  local arm=$1 seed=$2 label=b63_${1}_10k_s$2
  mkdir -p $L/$label/checkpoints
  for try in 1 2 3; do
    local res=""; [ -f $L/$label/checkpoints/checkpoint_latest.pt ] && res="--resume-from $L/$label/checkpoints/checkpoint_latest.pt"
    echo "$(date -u) start $label try $try ${res:-fresh}" >> $G/$label.log
    /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py $(arm_args $arm) $FIX \
      --norm batch --priority-alpha 0 --seed $seed --simnorm 0 \
      --total-steps 10000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
      --graph-search --run-label $label --log-dir $L $res >> $G/$label.log 2>&1
    local rc=$?; echo "$(date -u) $label exited rc=$rc" >> $G/$label.log
    [ $rc -eq 0 ] && break; sleep 30
  done
  # 10-game screen with the loop-stat protocol (8,000-step cap), same as the B61/B62 screens
  local ck=$L/$label/checkpoints/checkpoint_env10000_upd10000.pt
  [ -f $ck ] && /venv/main/bin/python -u atari_qwen/training/eval_loop_stats.py $ck --episodes 10 --sims 16 --max-steps 8000 --out $L/$label/eval >> $G/$label.eval.log 2>&1
}
for seed in "$@"; do
  for arm in gtrxl qdef qnowd; do run_one $arm $seed & sleep 5; done
  wait
  echo "$(date -u) seed $seed done" >> $G/queue.log
done
echo "$(date -u) ALL DONE $*" > $G/queue.done
