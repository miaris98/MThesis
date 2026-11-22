#!/usr/bin/env bash
# TODO B67 (2026-10-26, user go: "lets go with that", QK-norm): the CARLA recipe for the mixer (--mixer-optimizer adamw_carla, AdamW 3e-4,
# warm-up then cosine) WITH Qwen3-style QK-norm in every mixer block (--mixer-qk-norm). Without QK-norm the same recipe collapsed the
# attention by update 8,004 (B66 b66_fix_s1: entropy 0.002, largest logit 9,249; training return at random level by env 19k).
# No B62 flags, so it differs from b66_fix_s1 only in the QK-norm. One run; under NVIDIA MPS when other runs share the GPU.
#   SIZE small = the 2x128 mixer of the B66 screens; SIZE 30m = CARLA's qwen30m shape (the claim arm, one run per 32 GB card).
# The edited block must also be rerun on CARLA (E15 with --qk_norm 1, user rule 2026-10-26) before any Atari number is a claim.
# Training only (--eval-mode deferred); score afterwards with scripts/atari/final_evals.sh (paper protocol + sticky 0.25).
# usage: run_b67_qkn.sh SEED [SIZE]     results: $WORKSPACE/b66/out/b67_qkn_<size>_s<seed>, logs: $WORKSPACE/b66/logs/<label>.log
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/ez_train.sh"   # ez_env, ez_mps, ez_train_resumable (paths: common.sh)
L=$WORKSPACE/b66/out; G=$WORKSPACE/b66/logs; mkdir -p $L $G
S=${1:?seed}; SIZE=${2:-small}
ez_env 0; ez_mps
ARM="--trunk qwen --mixer-size $SIZE --mixer-out-init-std 0 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0 --mixer-optimizer adamw_carla --mixer-lr 3e-4 --mixer-grad-clip 5.0 --mixer-qk-norm"
label=b67_qkn_${SIZE}_s$S
ez_train_resumable $label $L $G $ARM --seed $S --graph-search
echo "$(date -u +%H:%M) B67 DONE $label" > $G/done_$label.marker
