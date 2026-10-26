#!/usr/bin/env bash
# TODO B64 (2026-10-24): where a 100k Qwen-small run (B63 arm qnowd) spends its time, and which execution changes make it faster
# without changing what it computes. Short probes on one GPU, nothing else running on it:
#   p1_plain    unshared rate, no diagnostics (the reference)
#   p2_split    --profile-targets (sub-stage ms per update, syncs the GPU) + a 20-update torch.profiler trace from update 1,500
#   p3_compile  --compile-model (torch.compile reduce-overhead on the online model; the reanalysis search stays a CUDA graph)
#   e1/e2       eval_loop_stats eager vs --graph on the B63 qnowd seed-1 checkpoint (10 games, 3,000-step cap): time, and the
#               per-game scores must be identical
#   s3 / s3_mps three p1 copies at once (seeds 1-3) without and with NVIDIA MPS: aggregate rate
# A probe is 4,000 env steps: 2,000 warm-up transitions, ~2,000 updates while collecting, ~2,000 after; the rate is read from the
# env-3000 and env-4000 log lines (1,000 updates while collecting).
# usage: run_b64_speed.sh CODE_DIR [WAIT_FILE]   (CODE_DIR = a repo copy with the B64 diagnostics; waits for WAIT_FILE if given)
# results: /workspace/b64/logs/summary.log (one line per probe), per-probe logs next to it
set -u
C=$1; WAIT=${2:-}
O=/workspace/b64/out; G=/workspace/b64/logs; mkdir -p $O $G
if [ -n "$WAIT" ]; then while [ ! -f "$WAIT" ]; do sleep 60; done; fi
cd $C
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=$C CUDA_VISIBLE_DEVICES=0
PY=/venv/main/bin/python
ARM="--trunk qwen --mixer-size small --mixer-out-init-std 0 --mixer-weight-decay 0 --stagnation-explore 300 --stagnation-burst 30
     --policy-target-smoothing 0.05 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0 --total-steps 4000
     --schedule-steps 100000 --eval-interval 100000 --eval-mode deferred --graph-search --no-mlflow"
rate() {  # log -> "min@3000 min@4000 => s/update"
  local a b; a=$(grep -E '^env +3000/' $1 | grep -oE '[0-9.]+ min$' | cut -d' ' -f1); b=$(grep -E '^env +4000/' $1 | grep -oE '[0-9.]+ min$' | cut -d' ' -f1)
  [ -n "$a" ] && [ -n "$b" ] && awk -v a=$a -v b=$b 'BEGIN { printf "%.1f->%.1f min, %.3f s/update", a, b, (b - a) * 60 / 1000 }'
}
probe() {  # label seed extra-flags...
  local l=$1 s=$2; shift 2
  $PY -u atari_qwen/training/train_ez_offpolicy.py $ARM --seed $s --run-label $l --log-dir $O "$@" > $G/$l.log 2>&1
  local rc=$?
  echo "$(date -u +%H:%M) $l rc=$rc $(rate $G/$l.log)" >> $G/summary.log
}
echo "$(date -u +%H:%M) B64 start on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $G/summary.log
probe p1_plain 1
probe p2_split 1 --profile-targets --profile-at 1500
grep -E '^\[B64 split\]' $G/p2_split.log | tail -2 >> $G/summary.log
probe p3_compile 1 --compile-model
grep -E 'compiled|torch.compile failed' $G/p3_compile.log | head -1 >> $G/summary.log

CK=/workspace/b63/out/b63_qnowd_10k_s1/checkpoints/checkpoint_env10000_upd10000.pt
for m in eager graph; do
  X=""; [ $m = graph ] && X="--graph"
  $PY -u atari_qwen/training/eval_loop_stats.py $CK --episodes 10 --sims 16 --max-steps 3000 --out $O/eval_$m $X > $G/e_$m.log 2>&1
  rc=$?
  echo "$(date -u +%H:%M) eval_$m rc=$rc $(grep -oE '"minutes": [0-9.]+' $G/e_$m.log | tail -1)" >> $G/summary.log
done
$PY - $O/eval_eager $O/eval_graph >> $G/summary.log 2>&1 <<'EOF'
import json, sys, glob
a, b = (json.load(open(glob.glob(d + "/*_loops.json")[0]))["per_game"] for d in sys.argv[1:3])
print("eval parity:", "IDENTICAL" if a == b else f"DIFFERENT eager {[g['score'] for g in a]} graph {[g['score'] for g in b]}")
EOF

for mode in s3 s3_mps; do
  if [ $mode = s3_mps ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log; mkdir -p /tmp/mps_pipe /tmp/mps_log
    nvidia-cuda-mps-control -d && echo "$(date -u +%H:%M) MPS started" >> $G/summary.log || { echo "MPS_FAILED" >> $G/summary.log; break; }
  fi
  for s in 1 2 3; do probe ${mode}_s$s $s & sleep 5; done
  wait
  [ $mode = s3_mps ] && echo quit | nvidia-cuda-mps-control
done
echo "$(date -u +%H:%M) B64_DONE" >> $G/summary.log
