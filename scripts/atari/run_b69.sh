#!/usr/bin/env bash
# TODO B69 (2026-10-27, user: "lets go with whats closest to the carla architecture"): the CARLA trunk layout on Atari against the current
# layout, same seed, same GPU, under NVIDIA MPS. Both runs use the recipe of the B66 SGD control b66_sgd_s1 (EZ-V2's SGD, mixer weight decay
# 0, no QK-norm: the block stays exactly the CARLA E15 block) plus every exact B68 speed flag (cache, merge, compiled search, compiled
# training step, graphed search). They differ only in --mixer-where:
#   b69_pred_s<seed>  the Qwen trunk as the prediction function (policy token -> value / policy heads), EZ-V2 conv representation + dynamics
#   b69_both_s<seed>  the Qwen mixer after the representation and after every dynamics step (the layout of b66_sgd_s1)
# b69_both against b66_sgd_s1 (paper 264.9, no B68 flags) is also the cache's own 100k screen (B68 step 2).
# Training only (--eval-mode deferred); scores afterwards with scripts/atari/final_evals.sh (paper protocol + sticky 0.25).
# usage: run_b69.sh SEED WHERE GPU      results: /workspace/b69/out/b69_<where>_s<seed>, logs: /workspace/b69/logs/<label>.log
# MIXER_SIZE=30m (2026-10-28, TODO B70): the CARLA network itself (qwen30m: 8 blocks, width 512), label b69_<where>_30m_s<seed>; default small
cd /workspace/MThesis
L=/workspace/b69/out; G=/workspace/b69/logs; mkdir -p $L $G
S=${1:?seed}; WHERE=${2:?pred or both}; GPU=${3:-0}; SIZE=${MIXER_SIZE:-small}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis CUDA_VISIBLE_DEVICES=$GPU
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /tmp/mps_pipe /tmp/mps_log
pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
ARM="--trunk qwen --mixer-size $SIZE --mixer-out-init-std 0 --audit-every 1000 --norm batch --priority-alpha 0 --simnorm 0 --mixer-weight-decay 0 --mixer-where $WHERE"
SPEED="--reanalyze-cache --reanalyze-merge --compile-search --compile-model --graph-search"
# B69_EXTRA / B69_TAG (2026-10-28): extra trainer flags and a label tag, e.g. the 30m trunk on its own AdamW after the SGD pilot's trunk died
# at update ~9,500 (B71): B69_EXTRA="--mixer-optimizer adamw --mixer-lr 3e-5" B69_TAG=aw -> b69_pred_30m_aw_s1
ARM="$ARM ${B69_EXTRA:-}"
label=b69_${WHERE}_s$S; [ "$SIZE" = small ] || label=b69_${WHERE}_${SIZE}_s$S
[ -n "${B69_TAG:-}" ] && label=${label%_s$S}_${B69_TAG}_s$S
mkdir -p $L/$label/checkpoints
for try in 1 2 3 4 5; do
  res=""; [ -f $L/$label/checkpoints/checkpoint_latest.pt ] && res="--resume-from $L/$label/checkpoints/checkpoint_latest.pt"
  echo "$(date -u +%H:%M) start $label try $try ${res:-fresh} on GPU $GPU" >> $G/$label.log
  /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py $ARM $SPEED --seed $S \
    --total-steps 100000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
    --run-label $label --log-dir $L $res >> $G/$label.log 2>&1
  rc=$?; echo "$(date -u +%H:%M) $label exited rc=$rc" >> $G/$label.log
  [ $rc -eq 0 ] && break; sleep 60
done
echo "$(date -u +%H:%M) B69 DONE $label" > $G/done_$label.marker
