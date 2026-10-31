#!/bin/bash
# Box side (TODO B66/B67, 2026-10-26): wait for a training done marker, then score the final checkpoint (update 120,000) of each listed run
# that has one, all evaluations at once under MPS (evals scale ~2.8x for 4 processes, 2026-10-12 bench):
#   paper_ep60_sim16      the paper protocol: 60 games, 16 simulations, no sticky actions, up to 30 no-op starts (the claim column)
#   sticky025_ep60_sim16  the same with sticky actions p=0.25 (robustness column, S-138; ~21-27 distinct games instead of ~4)
# The paper command equals the 2026-10-25 qnowd finals except --updates 120000 instead of --env-steps 100000 (which also matched 98,004).
# A run without a final checkpoint (stopped early) is skipped and named in the log.
# usage: final_evals.sh MARKER LABEL [LABEL ...]    logs: /workspace/b66/logs/eval_final_{paper,sticky}_<label>.log, marker evals_done_<first label>
cd /workspace/MThesis
export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
G=${G:-/workspace/b66/logs}; O=${O:-/workspace/b66/out}   # env overrides G, O, GPU (B69, 2026-10-27)
MARKER=$1; shift
until [ -f "$MARKER" ]; do sleep 60; done
echo "$(date -u +%H:%M) $(basename $MARKER) seen, starting the finals"
for l in "$@"; do
  if ! ls $O/$l/checkpoints/checkpoint_env*_upd120000.pt >/dev/null 2>&1; then echo "$(date -u +%H:%M) $l has no final checkpoint, skipped"; continue; fi
  /venv/main/bin/python -u atari_qwen/training/eval_ez_checkpoints.py $O/$l \
    --updates 120000 --episodes 60 --sims 16 --protocol paper_ep60_sim16 > $G/eval_final_paper_$l.log 2>&1 < /dev/null &
  sleep 20
  /venv/main/bin/python -u atari_qwen/training/eval_ez_checkpoints.py $O/$l \
    --updates 120000 --episodes 60 --sims 16 --sticky 0.25 --protocol sticky025_ep60_sim16 > $G/eval_final_sticky_$l.log 2>&1 < /dev/null &
  sleep 20
done
sleep 120; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/eval VRAM: /'
wait
echo "$(date -u +%H:%M) EVALS DONE $*" | tee $G/evals_done_$1.marker
