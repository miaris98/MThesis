#!/usr/bin/env bash
# TODO B61 step 2 (2026-10-23): Qwen blocks as the EZ-V2 token mixer, 10k screen against the GTrXL mixer. Same 100k-schedule recipe as the
# B22 / S058 10k screens (uniform replay, 4 envs, 16 sims, batch 256, SimNorm off as in the 363.1 recipe), stopped at 10,000 env steps.
#   qwen : --mixer-size SIZE, output projection std 0.02 (gradient reaches the blocks from update 1, S-116), AdamW mixer group lr 3e-5,
#          weight decay 0 (B37 failed at 3e-4); the rest of the net keeps EZ-V2's SGD.
#   gtrxl: the control, the unchanged 363.1 recipe (zero-init output, SGD everywhere); SIZE is ignored.
# SIMNORM (optional 5th arg, default 0): SimNorm group size on the latent (B22). The 2026-10-23 sn0 screen collapsed in 3 of 6 runs
# (both trunks; the Qwen mixer's |W| grew 60 -> 202 in the collapsed seed); SimNorm sits after the mixer, so it bounds the mixer's output.
# usage: run_b61_screen.sh GPU TRUNK SIZE SEED [SIMNORM]       (TRUNK = qwen | gtrxl, SIZE = small | 30m)
# label: B61_<trunk><size>_10k_s<seed> (sn0) or ..._s<seed>_sn<SIMNORM>; results under /workspace/b61/out/<label>
cd /workspace/MThesis
# B62=1 (env): the anti-collapse fixes on top (training-time stagnation exploration after 300 reward-free steps, 30-step random burst;
# policy-target smoothing 0.05); label suffix _b62. Evaluation is unchanged (paper protocol, no FIRE reset).
GPU=$1; TRUNK=$2; SIZE=$3; SEED=$4; SIMNORM=${5:-0}
SUF=""; [ "$SIMNORM" != 0 ] && SUF=_sn$SIMNORM
FIX=""; if [ "${B62:-0}" = 1 ]; then SUF=${SUF}_b62; FIX="--stagnation-explore 300 --stagnation-burst 30 --policy-target-smoothing 0.05"; fi
if [ "$TRUNK" = qwen ]; then
  LABEL=B61_qwen${SIZE}_10k_s${SEED}${SUF}
  ARM="--trunk qwen --mixer-size $SIZE --mixer-out-init-std 0.02 --mixer-optimizer adamw --mixer-lr 3e-5 --mixer-weight-decay 0 --audit-every 1000 --nan-guard"
else
  LABEL=B61_gtrxl_10k_s${SEED}${SUF}
  ARM="--trunk gtrxl"
fi
L=/workspace/b61/out; mkdir -p $L/$LABEL/checkpoints /workspace/b61/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis
for try in 1 2 3; do
  CK=$L/$LABEL/checkpoints/checkpoint_latest.pt; RES=""; [ -f $CK ] && RES="--resume-from $CK"
  echo "$(date -u) start $LABEL try $try ${RES:-fresh}" >> /workspace/b61/logs/$LABEL.log
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
    $ARM $FIX --norm batch --priority-alpha 0 --seed $SEED --simnorm $SIMNORM \
    --total-steps 10000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
    --graph-search --run-label $LABEL --log-dir $L $RES >> /workspace/b61/logs/$LABEL.log 2>&1
  rc=$?; echo "$(date -u) $LABEL exited rc=$rc" >> /workspace/b61/logs/$LABEL.log; [ $rc -eq 0 ] && break; sleep 30
done
echo SCREEN_DONE $LABEL
