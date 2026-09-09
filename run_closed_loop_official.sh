#!/usr/bin/env bash
# Same three arms and same tm_port discipline as run_closed_loop_arms.sh, but driving the
# official CARLA Leaderboard 1.0 routes (--route_source official) on Town04 instead of our
# own random routes on Town01. Town04 was chosen because it is genuinely held-out - never
# in this project's training data (Town01/02/03/10) - and because Town05 (also held-out,
# also in routes_testing.xml) crashes the CARLA server reproducibly on this instance even
# with the pre-existing random-route code, unrelated to anything built for this run. See
# TODO_leaderboard_benchmark.md for the full finding.
#
# --max_steps 8000 (400s sim) is sized from the real route lengths: Town04's 10 test routes
# average ~2177 m (computed directly from routes_testing.xml, not estimated), and this
# project's policies cruise near the 20 km/h (5.56 m/s) target used throughout training, so
# 2177 / 5.56 ~= 392s of driving for an average route. Town04's routes are 4-20x longer than
# this project's own 100-500 m random routes, which used --max_steps 3000.
set -x
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=/workspace/closed_loop_official
mkdir -p "$OUT"
ROUTE_FILE=/workspace/leaderboard/data/routes_testing.xml

run_arm () {
  local label="$1" ckpt="$2" arch="$3" port="$4" tm_port="$5"
  bash "$(dirname "${BASH_SOURCE[0]}")/start_carla_multi.sh" "$port" || return 1
  CARLA_ROOT=/workspace/carla /workspace/venv_carla/bin/python eval_wor_closed_loop.py \
    --checkpoint "$ckpt" --policy_arch "$arch" \
    --port "$port" --tm_port "$tm_port" \
    --town Town04 --route_source official --route_file "$ROUTE_FILE" \
    --routes 10 --route_seed 0 --max_steps 8000 \
    --num_vehicles 20 --blocked_timeout_s 45 \
    --record_video 1 --video_routes 2 \
    --out "$OUT/${label}.json" \
    > "$OUT/${label}.log" 2>&1 &
}

run_arm cnn_official          /workspace/checkpoints/wor_cl_arms/cnn_s0/best_model.pth             cnn     2000 8000
run_arm qwen30m_official      /workspace/checkpoints/wor_cl_arms/qwen30m_baseline_s0/best_model.pth qwen30m 2010 8010
run_arm qwen30m_geom_official /workspace/checkpoints/wor_cl_arms/qwen30m_geom_s0/best_model.pth     qwen30m 2020 8020

wait
echo "ALL_OFFICIAL_ARMS_DONE"
