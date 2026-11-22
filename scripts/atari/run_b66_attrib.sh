#!/usr/bin/env bash
# TODO B66 (2026-10-26, user go): attribute the Qwen-vs-EZ-V2 gap. Two Qwen-small 100k runs on one GPU under NVIDIA MPS, both WITHOUT the
# B62 flags (no policy-target smoothing, no stagnation exploration), so they differ from the EZ-V2 champions only in the mixer:
#   b66_fix_s<seed>  the mixer trained with the CARLA recipe of the same Qwen network (--mixer-optimizer adamw_carla)
#   b66_sgd_s<seed>  the mixer trained with EZ-V2's SGD, as in the 2026-10-25 qnowd runs (qnowd minus the B62 flags)
# Against qnowd seed 1 (SGD + B62 flags, final 238.2) this separates the optimiser effect from the B62-flag effect.
# Training only (--eval-mode deferred); scores afterwards from eval_ez_checkpoints.py --updates 120000 (paper protocol).
# Resumes from checkpoint_latest.pt + replay_latest.npz of the same label (saved together every 2,500 updates; the mixer AdamW
# state is in the checkpoint since 2026-10-26).
# usage: run_b66_attrib.sh SEED      results: $WORKSPACE/b66/out/<label>, logs: $WORKSPACE/b66/logs/<label>.log
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/ez_train.sh"   # ez_env, ez_mps, ez_train_resumable (paths: common.sh)
L=$WORKSPACE/b66/out; G=$WORKSPACE/b66/logs; mkdir -p $L $G
ez_env 0; ez_mps
BASE="--trunk qwen --mixer-size small --mixer-out-init-std 0 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0"
FIX="$BASE --mixer-optimizer adamw_carla --mixer-lr 3e-4 --mixer-grad-clip 5.0"
SGD="$BASE --mixer-weight-decay 0"
S=${1:-1}
ez_train_resumable b66_fix_s$S $L $G $FIX --seed $S --graph-search & sleep 5
ez_train_resumable b66_sgd_s$S $L $G $SGD --seed $S --graph-search &
wait
echo "$(date -u +%H:%M) B66 DONE seed $S" > $G/done_s$S.marker
