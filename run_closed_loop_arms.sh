#!/usr/bin/env bash
# Three arms, one route manifest, run CONCURRENTLY on separate CARLA servers.
# --route_seed is identical across all three, so every checkpoint drives the same routes
# through the same traffic and the results can be paired.
#
# Concurrency requires two things a single-server setup does not, both discovered by
# running this concurrently for the first time (2026-09-09):
#   1. Each CARLA server needs a --carla-port at least 10 apart from the others: a server
#      actually claims a 3-port range (port, port+1, port+2 - RPC, streaming, secondary),
#      so ports spaced by less than 3 collide outright.
#   2. Each client needs an explicit, DISTINCT --tm_port. get_trafficmanager() binds a
#      fixed port (8000) regardless of which CARLA server the client is talking to, so
#      three concurrent evaluations against three different --port values still collide
#      on the traffic manager unless tm_port is set explicitly and differently for each.
#      Symptom without the fix: the second caller gets a clean RuntimeError on bind, but
#      the first can instead take the whole process down with an uncatchable C++
#      exception (clmdep_msgpack::v1::type_error -> std::terminate) - no Python
#      try/except reaches it, so this is not something a retry loop can paper over.
set -x
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=/workspace/closed_loop
mkdir -p "$OUT"

run_arm () {
  local label="$1" ckpt="$2" arch="$3" port="$4" tm_port="$5"
  bash "$(dirname "${BASH_SOURCE[0]}")/start_carla_multi.sh" "$port" || return 1
  CARLA_ROOT=/workspace/carla /workspace/venv_carla/bin/python eval_wor_closed_loop.py \
    --checkpoint "$ckpt" --policy_arch "$arch" \
    --port "$port" --tm_port "$tm_port" \
    --town Town01 --routes 15 --route_seed 0 --max_steps 3000 \
    --num_vehicles 20 --blocked_timeout_s 45 \
    --record_video 1 --video_routes 2 \
    --out "$OUT/${label}.json" \
    > "$OUT/${label}.log" 2>&1 &
}

run_arm cnn_s0             /workspace/checkpoints/wor_cl_arms/cnn_s0/best_model.pth             cnn     2000 8000
run_arm qwen30m_s0         /workspace/checkpoints/wor_cl_arms/qwen30m_baseline_s0/best_model.pth qwen30m 2010 8010
run_arm qwen30m_geom_s0    /workspace/checkpoints/wor_cl_arms/qwen30m_geom_s0/best_model.pth     qwen30m 2020 8020

wait
echo "ALL_CLOSED_LOOP_ARMS_DONE"
