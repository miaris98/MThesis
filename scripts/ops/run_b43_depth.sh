#!/usr/bin/env bash
# TODO B43 step 1 (S-125): score the 16 restored B35 env30000 checkpoints at 64 simulations with the imagined depth capped at D (30 episodes, no sticky, graph search),
# one evaluator per checkpoint, round-robin over the GPUs, at most NPAR at a time. Needs `restore_b35_checkpoints.py` first (box with /workspace/MThesis).
#
# usage: run_b43_depth.sh "3 5 8" NPAR [SIMS]          (D = 0 means unlimited: the B24 reference at the same protocol)
cd /workspace/MThesis
DEPTHS=${1:-"3 5 8"}; NPAR=${2:-4}; SIMS=${3:-64}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MLFLOW_ALLOW_FILE_STORE=true
mkdir -p /workspace/logs_eval
NG=$(nvidia-smi -L | wc -l); i=0
for D in $DEPTHS; do
  DF=""; [ "$D" != 0 ] && DF="--max-depth $D"
  for d in results/100k_benchmark/S058_ezv2_match/S058*; do
    r=$(basename $d)
    while [ "$(pgrep -fc '[e]val_ez_checkpoints.py')" -ge "$NPAR" ]; do sleep 20; done
    CUDA_VISIBLE_DEVICES=$((i % NG)) setsid nohup /venv/main/bin/python atari_qwen/training/eval_ez_checkpoints.py $d \
      --episodes 30 --sims $SIMS --env-steps 30000 --graph $DF --protocol ep30_sim${SIMS}_d${D} > /workspace/logs_eval/${r}_sim${SIMS}_d${D}.log 2>&1 < /dev/null &
    i=$((i + 1)); sleep 3
  done
done
wait
echo "$(date -u) B43 depth sweep done: depths $DEPTHS"
