#!/usr/bin/env bash
# Extra Gate 2 seeds (from scratch, 20k env steps) so the 16-vs-32-sim decision rests on more than
# one seed. Companion to launch_gate2.sh (S053a/b/c); same trainer, same eval protocol.
#
# env: GPU (default 0), CORES_BASE (default 0) - each run gets 8 dedicated cores.
set -euo pipefail
cd /workspace/MThesis
PY=/venv/main/bin/python
T=atari_qwen/training/train_mcts_offpolicy.py
GPU="${GPU:-0}"; B="${CORES_BASE:-0}"
COMMON="--env-id BreakoutNoFrameskip-v4 --total-steps 20000 --num-envs 8 --eval-interval 5000 \
  --eval-episodes 20 --ckpt-interval 2500 --min-replay-size 1000"
i=0
for spec in "S053d_gate2_sim32_s2:32:2" "S053e_gate2_sim16_s1:16:1" "S053f_gate2_sim16_s2:16:2"; do
  IFS=: read -r name sims seed <<<"$spec"
  cores="$((B + i * 8))-$((B + i * 8 + 7))"
  CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$cores" nohup $PY $T $COMMON \
    --num-simulations "$sims" --eval-simulations "$sims" --seed "$seed" \
    --log-dir "results/100k_benchmark/$name" </dev/null > "/workspace/$name.log" 2>&1 &
  echo "launched $name (GPU $GPU, cores $cores, $sims sims, seed $seed) pid $!"
  i=$((i + 1))
done
