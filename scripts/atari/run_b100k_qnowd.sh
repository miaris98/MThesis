#!/usr/bin/env bash
# TODO B100K (2026-10-24, user go): three seeds of the qnowd recipe (Qwen-small mixer, mixer weight decay 0, B62 anti-collapse
# fixes) on the full 100k schedule (100,000 online + 20,000 offline updates), three processes on one RTX 5090 under NVIDIA MPS
# (B64: 0.31 s per update each vs 0.40 s without MPS, 0.145 s alone). Training only: a checkpoint every 10k env steps
# (--eval-mode deferred); the scores come afterwards from eval_loop_stats.py (screen) and eval_ez_checkpoints.py (paper protocol).
# Resumes from checkpoint_latest.pt + replay_latest.npz of the same label (the pair is saved together every 2,500 updates).
# usage: run_b100k_qnowd.sh SEEDS...          results: $WORKSPACE/b100k/out/<label>, logs: $WORKSPACE/b100k/logs/<label>.log
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/ez_train.sh"   # ez_env, ez_mps, ez_train_resumable (paths: common.sh)
L=$WORKSPACE/b100k/out; G=$WORKSPACE/b100k/logs; mkdir -p $L $G
ez_env 0; ez_mps
ARM="--trunk qwen --mixer-size small --mixer-out-init-std 0 --mixer-weight-decay 0 --stagnation-explore 300 --stagnation-burst 30 \
  --policy-target-smoothing 0.05 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0"
for s in "$@"; do ez_train_resumable b100k_qnowd_s$s $L $G $ARM --seed $s --graph-search & sleep 5; done
wait
echo "$(date -u +%H:%M) B100K DONE $*" > $G/done.marker
