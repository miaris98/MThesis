#!/usr/bin/env bash
# Gate 3 (-> 50k env steps) + Gate 2 companions for the off-policy MCTS trainer (S-053/S-054).
#
# usage: scripts/setup/launch_gate3.sh <resume|scratch> <name> <cores> <sims> <seed> <total_steps> [extra trainer args...]
#   resume  : <name> is a Gate 2 results dir; resumes from its newest checkpoint_latest.pt, which
#             restores model, target net, optimizer, RNG, step AND replay_latest.npz beside it.
#   scratch : fresh run into results/100k_benchmark/<name>.
# env: GPU (default 0)
set -euo pipefail
MODE=$1; NAME=$2; CORES=$3; SIMS=$4; SEED=$5; TOTAL=$6; shift 6
cd /workspace/MThesis
PY=/venv/main/bin/python
T=atari_qwen/training/train_mcts_offpolicy.py
GPU="${GPU:-0}"
COMMON="--env-id BreakoutNoFrameskip-v4 --total-steps $TOTAL --num-envs 8 --eval-interval 5000 \
  --eval-episodes 20 --ckpt-interval 2500 --min-replay-size 1000 \
  --num-simulations $SIMS --eval-simulations $SIMS --seed $SEED"
if [ "$MODE" = resume ]; then
  CK=$(ls -t results/100k_benchmark/$NAME/*/checkpoints/checkpoint_latest.pt | head -1)
  [ -f "$(dirname "$CK")/replay_latest.npz" ] || { echo "FATAL: no replay buffer beside $CK"; exit 1; }
  OUT="${NAME}_to${TOTAL}"
  EXTRA="--resume-from $CK"
else
  OUT="$NAME"; EXTRA=""
fi
CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORES" nohup $PY $T $COMMON $EXTRA "$@" \
  --log-dir "results/100k_benchmark/$OUT" </dev/null > "/workspace/$OUT.log" 2>&1 &
echo "launched $OUT (GPU $GPU, cores $CORES, $SIMS sims, seed $SEED, total $TOTAL) pid $! ${EXTRA}"
