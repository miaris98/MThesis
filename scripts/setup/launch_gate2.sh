#!/usr/bin/env bash
# Gate 2 (20k env steps) for the fixed off-policy MCTS trainer (S-053). Run on the box.
#
#   (a) 16 collection sims, seed 42 - resumed from Gate 1 checkpoint_latest.pt (step 5,000)
#   (b) 32 collection sims, seed 42 - resumed from the same Gate 1 checkpoint
#   (c) 32 collection sims, seed 1  - from scratch: an independent seed cannot be resumed from
#       seed 42's weights. Replaces the old policy-only "control", which cannot learn by
#       construction (its policy target is its own output).
#
# Resume from checkpoint_LATEST (never _best), and never pass --start-step: the checkpoint's own
# step is authoritative. Gate 1 predates replay-buffer checkpointing, so (a)/(b) restart with a
# fresh warm-up buffer - logged loudly by the trainer; Gate 2 -> 3 resumes will carry the buffer.
#
# Pass criteria (decided before results): at 20k, MCTS score beats the harness random baseline
# (~0.8) by > 2 SE AND beats the raw policy; proj_std not collapsing to ~0; reward recall > 0.
set -euo pipefail
cd /workspace/MThesis
PY=/venv/main/bin/python
T=atari_qwen/training/train_mcts_offpolicy.py
# Several gate1 run dirs exist (aborted starts); take the newest one that actually has a checkpoint.
GATE1_CKPT=$(ls -t results/100k_benchmark/gate1_poc_16sims/*/checkpoints/checkpoint_latest.pt | head -1)
STEP=$($PY -c "import torch;print(torch.load('$GATE1_CKPT',map_location='cpu',weights_only=False)['step'])")
echo "Gate 1 checkpoint: $GATE1_CKPT (step $STEP)"
[ "$STEP" -ge 4990 ] || { echo "FATAL: expected Gate 1 latest at ~5000, got $STEP"; exit 1; }

COMMON="--env-id BreakoutNoFrameskip-v4 --total-steps 20000 --num-envs 8 --eval-interval 5000 \
  --eval-episodes 20 --ckpt-interval 2500 --min-replay-size 1000"

launch() {  # name cores sims seed [resume]
  local name=$1 cores=$2 sims=$3 seed=$4 resume=${5:-}
  local extra=""; [ -n "$resume" ] && extra="--resume-from $resume"
  taskset -c "$cores" nohup $PY $T $COMMON --num-simulations "$sims" --eval-simulations "$sims" \
    --seed "$seed" --log-dir "results/100k_benchmark/$name" $extra \
    </dev/null > "/workspace/$name.log" 2>&1 &
  echo "launched $name (cores $cores, $sims sims, seed $seed) pid $!"
}
launch S053a_gate2_sim16_s42 16-20 16 42 "$GATE1_CKPT"
launch S053b_gate2_sim32_s42 21-25 32 42 "$GATE1_CKPT"
launch S053c_gate2_sim32_s1  26-31 32 1
