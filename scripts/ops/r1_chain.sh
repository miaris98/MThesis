#!/usr/bin/env bash
# R1 day (2026-11-19; plan: docs/ops/2026-11-18_teacher_ladder.md, steps 4-6 + 8a), run on the box after
# `SCRATCH=DIR scripts/ops/box_deploy.sh PORT HOST scripts/ops/manifests/ladder_r1.txt`. Status lines: $WORKSPACE/ladder/status.
#   setup: data (base D + the six coverage families DX) and CARLA in parallel
#   audit (leak check, merged dir DC) -> cache D (+ PCA-256 map) -> cache DX (PCA in D's basis) -> targets (_visns on D, all objects on DX)
#   cut-in log-side check (CPU, background) | detector (12 epochs, best by val F1) -> export tokens for D and DX
#   r1_s0, r1_s1 (RGB + 20 detector tokens <= 30 m) and r0ft (the same recipe without objects) side by side on the one GPU
#   gate: R1's val waypoint ADE must beat r0ft's; then NL lanes drive LANE_ARMS (default r1_s0 and r0ft) on all 220 routes, CARLA Epic
# R1 is an RGB-only agent: the detector runs on the frozen encoder's own map, live (single RGB camera at inference, user 2026-11-18).
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
NL=${NL:-4}
LANE_ARMS=${LANE_ARMS:-"r1_s0 r0ft"}
CORES=$(cgroup_cores)   # the cgroup quota, not nproc
F="CARLA_QUALITY=Epic"
mark "R1_CHAIN_START NL=$NL LANE_ARMS=$LANE_ARMS"
grep -q SETUP_START $S || bash $B setup &
wait_files $CHECKPOINTS/E15/READY
mark "CKPTS_READY"

render_when_carla &

await DATA_DONE DATA_FAILED data
step AUDIT_DONE audit
await CARLA_PROV_END CARLA_PROV_FAILED "carla venv"       # caches and training run in venv_carla
step "CACHE_DONE D " cache D
step "CACHE_DONE DX" cache DX
step TARGETS_DONE targets_r1
( bash $B cutin ) &
step DETECTOR_DONE detector
step EXPORT_DONE export
export NW=$(( CORES / 4 > 6 ? 6 : (CORES / 4 > 2 ? CORES / 4 : 2) ))
for A in r1_s0 r1_s1 r0ft; do
  [ -f $CHECKPOINTS/r1d_$A/TRAINED ] || ( bash $B train $A ) &
  sleep 60
done
wait_trained() { for A in r1_s0 r1_s1 r0ft; do [ -f $CHECKPOINTS/r1d_$A/TRAINED ] || grep -q "TRAIN_END $A " $S || return 1; done; }
until wait_trained; do sleep 60; done
mark "TRAINED:$(trained_list r1d_ r1_s0 r1_s1 r0ft)"
bash $B gate
grep -q GATE_PASS $S || chain_stop "gate (R1 does not beat its RGB control offline: no CARLA run; save and destroy)"

grep -q RENDER_OK $S || chain_stop "no render"
SPECS=()
for A in $LANE_ARMS; do
  [ -f $CHECKPOINTS/r1d_$A/TRAINED ] && SPECS+=("r1d_$A:r1d_$A/model_epoch_030.pth:$L/ids220.txt:$F")
done
[ ${#SPECS[@]} -gt 0 ] || chain_stop "no lane arm trained"
bash $B lanes $NL "${SPECS[@]}"
mark R1_LANES_STARTED
