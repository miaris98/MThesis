#!/usr/bin/env bash
# Box side, 2026-10-11, user-approved: official paper-protocol Atari eval of the FINAL checkpoint (120k updates) of one seed.
# Protocol: 60 episodes (seeds 10000+i), 16 simulations (EZ-V2 code), CUDA-graph search, no sticky actions, no FIRE reset, 27k cap.
# Waits until the seed's trainer has exited and the newest checkpoint is at >= 119000 updates, then starts the eval with nohup.
# Log: /workspace/eval16_final_s<SEED>.log (GPU name, start time, eval output).
# usage: bash official_final_eval_watch.sh SEED
SEED=$1
RUN=/workspace/MThesis/results/100k_benchmark/S058_ezv2_match/S058h_gtrxl_100k_s${SEED}_uniform
LOG=/workspace/eval16_final_s${SEED}.log
cd /workspace/MThesis || exit 1
while true; do
  UPD=$(ls "$RUN/checkpoints" 2>/dev/null | grep -E '^checkpoint_env[0-9]+_upd[0-9]+\.pt$' | sed -E 's/.*_upd([0-9]+)\.pt/\1/' | sort -n | tail -n 1)
  if [ -n "$UPD" ] && [ "$UPD" -ge 119000 ] && ! pgrep -f "train_ez_offpolicy.py.*--seed $SEED " > /dev/null; then
    break
  fi
  sleep 120
done
{
  echo "$(date -u +%H:%M) UTC  seed $SEED  final checkpoint at upd $UPD"
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
} > "$LOG"
nohup /venv/main/bin/python -u atari_qwen/training/eval_ez_checkpoints.py "$RUN" --episodes 60 --sims 16 --graph --protocol paper60_s16 --updates "$UPD" >> "$LOG" 2>&1 < /dev/null &
echo "$(date -u +%H:%M) UTC  eval launched, pid $!" >> "$LOG"
