#!/usr/bin/env bash
# Three arms, one route manifest. --route_seed is identical across all three, so every
# checkpoint drives the same routes through the same traffic and the results can be paired.
set -x
cd /workspace/MThesis
OUT=/workspace/closed_loop
mkdir -p "$OUT"

run_arm () {
  local label="$1" ckpt="$2" arch="$3"
  # CARLA is restarted per arm: a server that has already loaded a town, spawned and
  # destroyed 15 routes' worth of actors is not in the same state as a fresh one, and
  # Group 3 is a long record of what that costs. A restart is 30 s against a 20 min arm.
  bash /workspace/start_carla.sh 2000 || return 1
  CARLA_ROOT=/workspace/carla /workspace/venv_carla/bin/python eval_wor_closed_loop.py \
    --checkpoint "$ckpt" --policy_arch "$arch" \
    --town Town01 --routes 15 --route_seed 0 --max_steps 3000 \
    --num_vehicles 20 --blocked_timeout_s 45 \
    --record_video 1 --video_routes 2 \
    --out "$OUT/${label}.json" 2>&1 | grep -vE 'FutureWarning|torch.load|^\s*(ckpt|checkpoint|frozen) '
}

run_arm cnn_s0             /workspace/checkpoints/wor_cl_cnn/cnn_s0/best_model.pth                    cnn
run_arm qwen30m_s0         /workspace/checkpoints/wor_geom_ab_baseline/qwen30m_baseline_s0/best_model.pth qwen30m
run_arm qwen30m_geom_s0    /workspace/checkpoints/wor_geom_ab/qwen30m_geom_s0/best_model.pth          qwen30m

echo "ALL_CLOSED_LOOP_ARMS_DONE"
