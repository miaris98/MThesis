#!/usr/bin/env bash
# Retrain both WoR arms with the v2 input/perception stack.
#
# WHAT CHANGED SINCE THE FULL-8-TOWN RUN, AND WHY
# ------------------------------------------------
# The pilot's failure mode was unambiguous: ~60% of Leaderboard infractions were collisions with
# vehicles and ~11% with layout, while waypoint quality was fine (val ADE 0.42 m). The policy
# planned well and drove into things - a perception problem, not a planning one. Five changes,
# all aimed there:
#
#   1. Camera parity. The dataset was rendered at (x=-1.5, z=2.0, fov=110, 1024x512); the eval
#      agent was asking CARLA for (x=+1.3, z=1.3, fov=100, 256x256) - 2.8 m forward, 0.7 m low,
#      and a ~29 deg wider vertical cone. Every closed-loop number so far was produced by
#      showing the network a projection it was never trained on. Now one definition, both paths
#      (src/config/camera.py).
#   2. Aspect-correct input. 1024x512 -> 256x256 squashed horizontal geometry ~2.7x relative to
#      vertical. Now cropped to the source's 8:3 (bonnet removed, as TransFuser++ does) and
#      resized without distortion.
#   3. CARLA-pretrained vision. The frozen encoder is now regnety_032 lifted from the released
#      TransFuser++ checkpoint - trained on this exact dataset and simulator, jointly with
#      depth/BEV/detection supervision - instead of ImageNet ResNet-34.
#   4. Target-speed head. An 8-bin classifier with an exactly-zero class, so the policy can
#      command a stop outright rather than having one inferred from waypoint spacing (which can
#      express "slow" but only approaches "stopped").
#   5. Route overlay. The planned route drawn into the image like a reversing camera's guide
#      lines, so route and road share a spatial frame before the frozen encoder sees either.
#
# MEASURED COST (RTX 3080 Ti, clean GPU, 1000-batch steady-state runs - not extrapolated)
# -------------------------------------------------------------------------------------
#   cnn      387.1 samples/s  -> ~19.0 min/epoch train + ~1.5 min val  -> ~17h for 50 epochs
#   qwen30m  450.8 samples/s  -> ~16.3 min/epoch train + ~1.3 min val  -> ~15h for 50 epochs
#
# Two things worth knowing about those numbers. First, a 100-batch smoke reported 136 and 189
# samples/s for the same configurations - startup (cuDNN autotune, worker spin-up, CUDA context)
# dominates at that length, so short smokes understate steady-state by ~2.8x and must not be
# extrapolated from. Second, an isolated forward+backward on this GPU runs at 1057 (cnn) and 587
# (qwen) samples/s, so a single arm leaves the GPU idle roughly 60% of the time - the loop is
# limited by per-batch host work, not by the GPU. That idle time is why the two arms are launched
# concurrently rather than one after the other: they interleave into each other's gaps instead of
# queueing. num_workers is 8 rather than 12 for the same reason - two arms share 16 vCPUs, and
# data wait was only ~1% of step time at 12, so there is nothing to lose by halving it.
#
# ATTRIBUTION CAVEAT, STATED ONCE AND ON PURPOSE
# ----------------------------------------------
# These five ship together, so a change in driving score cannot be attributed to any one of
# them. That was a deliberate call (speed over attribution). If the result needs explaining
# later, the cheap follow-up is re-running with --route_overlay 0 and with
# --target_speed_loss_weight 0, which isolates the two genuinely optional pieces; the camera
# parity fix is a bug fix and would not be reverted for an ablation.
set -u

MTHESIS=/workspace/MThesis
DATA=/workspace/dataset/wor_trajectories
CKPT_ROOT=/workspace/checkpoints/wor_v2
TFPP=/workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
PY=/venv/main/bin/python

EPOCHS="${EPOCHS:-50}"
IMG_SIZE="${IMG_SIZE:-192x512}"
CROP="${CROP:-0.25}"
ROUTE_POINTS="${ROUTE_POINTS:-20}"
TS_WEIGHT="${TS_WEIGHT:-1.0}"
OVERLAY="${OVERLAY:-1}"
WORKERS="${WORKERS:-8}"

if [ ! -f "$TFPP" ]; then
  echo "FATAL: TransFuser++ weights not found at $TFPP"
  echo "Re-download:  curl -L -o /workspace/tfpp_pretrained.zip \\"
  echo "  https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip"
  echo "  && unzip -q /workspace/tfpp_pretrained.zip -d /workspace/tfpp_pretrained"
  exit 1
fi

common_args() {
  # batch_size stays 32. It is the one hyperparameter held fixed across every run in this
  # project so the arms remain comparable with the existing 59.14 / 56.86 Bench2Drive numbers;
  # the GPU headroom freed by a frozen backbone is spent on resolution and heads instead.
  echo "--data_dir $DATA \
    --backbone regnety_032 --weights_path $TFPP \
    --img_size $IMG_SIZE --crop_bottom_frac $CROP \
    --route_overlay $OVERLAY --route_points $ROUTE_POINTS \
    --target_speed_loss_weight $TS_WEIGHT \
    --epochs $EPOCHS --batch_size 32 --val_split 0.15 --seed 0 --split_seed 0 \
    --num_workers $WORKERS --cache_decoded 1"
}

mkdir -p "$CKPT_ROOT"
cd "$MTHESIS" || exit 1

echo "=== launching cnn arm (port-free, GPU shared) ==="
setsid $PY scripts/training/train_wor.py $(common_args) \
  --policy_arch cnn \
  --save_dir "$CKPT_ROOT/cnn_s0" --run_label cnn_v2_full8town \
  </dev/null >/workspace/train_v2_cnn.log 2>&1 &
echo "  cnn pid $!"

# Stagger so the two runs do not build the decode cache for the same frames simultaneously -
# the first epoch is cache-cold and disk-bound, and two processes racing on it just doubles the
# IO without halving the time.
sleep 120

echo "=== launching qwen30m+geom arm ==="
setsid $PY scripts/training/train_wor.py $(common_args) \
  --policy_arch qwen30m --vision_grid 6x16 \
  --heading_loss_weight 0.5 --curvature_loss_weight 0.2 \
  --save_dir "$CKPT_ROOT/qwen30m_geom_s0" --run_label qwen30m_geom_v2_full8town \
  </dev/null >/workspace/train_v2_qwen.log 2>&1 &
echo "  qwen pid $!"

echo
echo "=== launched. monitor with: ==="
echo "  tail -f /workspace/train_v2_cnn.log"
echo "  tail -f /workspace/train_v2_qwen.log"
echo "  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"
