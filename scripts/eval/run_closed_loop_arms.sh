#!/usr/bin/env bash
# Three arms (cnn, qwen30m, qwen30m_geom) on one route set, run CONCURRENTLY on separate CARLA servers with the same --route_seed,
# so every checkpoint drives the same routes through the same traffic and the results pair. One script for what were three
# (run_closed_loop_official.sh and run_closed_loop_arms_by_seed.sh were copies with the route set or the training seed changed;
# retired 2026-11-22, verbatim in E:\MThesis_EXP\local_repo_logs_20261122\retired_scripts).
#
# usage: scripts/eval/run_closed_loop_arms.sh [TRAIN_SEED] [PORT_BASE]        env: ROUTES=random (default) | official, OUT
#   ROUTES=random    Town01, 15 of our own random routes, --max_steps 3000 (the 13.31 comparison; TRAIN_SEED 1, 2 replicate it)
#   ROUTES=official  Town04 from the official CARLA Leaderboard 1.0 routes_testing.xml, 10 routes, --max_steps 8000. Town04 is
#                    held out (never in this project's training towns 01/02/03/10); Town05 crashed the server on that instance.
#                    8000 steps (400 s) from the real route lengths: the 10 test routes average ~2177 m, ~392 s at the 20 km/h target.
# TRAIN_SEED picks the checkpoints ($CHECKPOINTS/wor_cl_arms/<arm>_s<seed>); --route_seed stays 0 for every training seed.
# Output: $WORKSPACE/closed_loop[_official][_seed<N>] (the folders the three old scripts wrote).
#
# Concurrency needs two things a single server does not (found running this concurrently for the first time, 2026-09-09):
#   1. CARLA ports at least 10 apart: a server claims port, port+1, port+2 (RPC, streaming, secondary).
#   2. An explicit, DISTINCT --tm_port per client: get_trafficmanager() binds 8000 whichever server the client talks to, and the
#      collision can take the process down with an uncatchable C++ exception (clmdep_msgpack type_error -> std::terminate).
# Staggered launch: three servers started back to back overlap their heavy init (map load, shader compile); two of three
# segfaulted within a minute in the first seed-1 run, and a 20 s settle between launches fixed it. `wait` only proves the
# children exited, so each arm's own JSON is checked against the requested route count before the DONE line.
set -x
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # the repo root: it puts `src` on the import path
. scripts/lib/common.sh
TRAIN_SEED=${1:-0}; PORT_BASE=${2:-2000}; ROUTES=${ROUTES:-random}
SEED_DIR=$([ "$TRAIN_SEED" = 0 ] || echo "_seed$TRAIN_SEED")
case $ROUTES in
  random)   ROUTE_ARGS="--town Town01 --routes 15 --route_seed 0 --max_steps 3000"; N_ROUTES=15
            SUFFIX=s$TRAIN_SEED; OUT=${OUT:-$WORKSPACE/closed_loop$SEED_DIR} ;;
  official) ROUTE_ARGS="--town Town04 --route_source official --route_file $WORKSPACE/leaderboard/data/routes_testing.xml"
            ROUTE_ARGS="$ROUTE_ARGS --routes 10 --route_seed 0 --max_steps 8000"; N_ROUTES=10
            SUFFIX=official$([ "$TRAIN_SEED" = 0 ] || echo "_s$TRAIN_SEED"); OUT=${OUT:-$WORKSPACE/closed_loop_official$SEED_DIR} ;;
  *) echo "ROUTES must be random or official"; exit 2 ;;
esac
CKPT_DIR=$CHECKPOINTS/wor_cl_arms
mkdir -p "$OUT"

run_arm () {
  local label="$1" ckpt="$2" arch="$3" port="$4" tm_port="$5"
  bash scripts/setup/start_carla_multi.sh "$port" || return 1
  CARLA_ROOT=$CARLA_DIR "$VENV_CARLA/bin/python" scripts/eval/eval_wor_closed_loop.py \
    --checkpoint "$ckpt" --policy_arch "$arch" --port "$port" --tm_port "$tm_port" \
    $ROUTE_ARGS --num_vehicles 20 --blocked_timeout_s 45 --record_video 1 --video_routes 2 \
    --out "$OUT/${label}.json" > "$OUT/${label}.log" 2>&1 &
}

LABELS=(); i=0
for arm in cnn:cnn_s:cnn qwen30m:qwen30m_baseline_s:qwen30m qwen30m_geom:qwen30m_geom_s:qwen30m; do   # label:checkpoint dir:arch
  IFS=: read -r name dir arch <<< "$arm"
  [ $i -gt 0 ] && sleep 20
  run_arm "${name}_$SUFFIX" "$CKPT_DIR/${dir}${TRAIN_SEED}/best_model.pth" "$arch" $((PORT_BASE + 10 * i)) $((PORT_BASE + 6000 + 10 * i))
  LABELS+=("${name}_$SUFFIX"); i=$((i + 1))
done
wait

ALL_OK=1
for label in "${LABELS[@]}"; do
  n=$("$VENV_CARLA/bin/python" -c "import json, sys; print(len(json.load(open(sys.argv[1]))['routes']))" "$OUT/${label}.json" 2>/dev/null)
  if [ "$n" = "$N_ROUTES" ]; then echo "ARM_OK ${label} ($N_ROUTES/$N_ROUTES routes)"
  else echo "ARM_INCOMPLETE ${label} (${n:-0}/$N_ROUTES routes) - see $OUT/${label}.log"; ALL_OK=0; fi
done
if [ "$ALL_OK" = 1 ]; then echo "ALL_CLOSED_LOOP_ARMS_DONE ($ROUTES, training seed $TRAIN_SEED)"
else echo "CLOSED_LOOP_ARMS_INCOMPLETE ($ROUTES, training seed $TRAIN_SEED) - at least one arm needs a manual restart"; fi
