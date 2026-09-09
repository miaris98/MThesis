#!/usr/bin/env bash
# Same protocol as run_closed_loop_arms.sh (Town01, 15 routes, --route_seed 0, identical
# traffic across arms so results pair), parameterized by *training* seed so the seed-0
# closed-loop comparison in 13.31 can be replicated at seed 1, seed 2, ... without hand-
# editing checkpoint paths each time.
#
# usage: run_closed_loop_arms_by_seed.sh <training_seed> [port_base]
#
# --route_seed is NOT the training seed above - it stays 0 across every invocation of this
# script, deliberately, so all seed replications drive the exact same routes and traffic as
# the original seed-0 run (13.31) and as each other. Only the checkpoint (i.e. what the
# policy learned) varies between invocations; the test itself does not.
#
# STAGGERED LAUNCH, AND WHY: the seed-1 run of this exact script lost two of three arms to
# a server-side segfault (Signal 11) within their first minute, and `wait` printed the DONE
# sentinel anyway - it only waits on its own children's exit, which a fast crash satisfies
# just as well as a real finish. `start_carla_multi.sh` returning only proves the RPC port
# is open, not that the engine has finished its heavy init (map load, shader compile) - that
# continues in the background, so three servers launched back-to-back still overlap on the
# expensive part of startup even though the port-open checks were sequential. A 20s settle
# delay between launches (found by fixing the seed-1 failure by hand) gives each server's
# heavy init a real head start before the next one begins. This is a mitigation, not a proof
# - the segfault's root cause is still undiagnosed - so the post-wait verification below is
# the part that actually matters: it is what would have caught the seed-1 loss immediately
# instead of ~28 minutes later.
set -x
TRAIN_SEED="${1:?usage: run_closed_loop_arms_by_seed.sh <training_seed> [port_base]}"
PORT_BASE="${2:-2000}"
cd "$(dirname "${BASH_SOURCE[0]}")"
CKPT_DIR=/workspace/checkpoints/wor_cl_arms
OUT=/workspace/closed_loop_seed${TRAIN_SEED}
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

run_arm "cnn_s${TRAIN_SEED}" "$CKPT_DIR/cnn_s${TRAIN_SEED}/best_model.pth" cnn \
  $((PORT_BASE)) $((PORT_BASE + 6000))
sleep 20

run_arm "qwen30m_s${TRAIN_SEED}" "$CKPT_DIR/qwen30m_baseline_s${TRAIN_SEED}/best_model.pth" qwen30m \
  $((PORT_BASE + 10)) $((PORT_BASE + 6010))
sleep 20

run_arm "qwen30m_geom_s${TRAIN_SEED}" "$CKPT_DIR/qwen30m_geom_s${TRAIN_SEED}/best_model.pth" qwen30m \
  $((PORT_BASE + 20)) $((PORT_BASE + 6020))

wait

# The check that actually matters (see header): `wait` only proves the child processes
# exited, not that they finished 15 routes. Verify each arm's own output against what was
# requested before trusting the sentinel below.
ALL_OK=1
for label in "cnn_s${TRAIN_SEED}" "qwen30m_s${TRAIN_SEED}" "qwen30m_geom_s${TRAIN_SEED}"; do
  n=$(/workspace/venv_carla/bin/python -c "
import json, sys
try:
    print(len(json.load(open('$OUT/${label}.json'))['routes']))
except Exception:
    print(0)
" 2>/dev/null)
  if [ "$n" = "15" ]; then
    echo "ARM_OK ${label} (15/15 routes)"
  else
    echo "ARM_INCOMPLETE ${label} (${n:-0}/15 routes) - see $OUT/${label}.log"
    ALL_OK=0
  fi
done

if [ "$ALL_OK" = "1" ]; then
  echo "ALL_CLOSED_LOOP_ARMS_SEED${TRAIN_SEED}_DONE"
else
  echo "CLOSED_LOOP_ARMS_SEED${TRAIN_SEED}_INCOMPLETE - at least one arm needs a manual restart"
fi
