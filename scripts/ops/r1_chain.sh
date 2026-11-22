#!/usr/bin/env bash
# R1 day (2026-11-19; plan: docs/ops/2026-11-18_teacher_ladder.md, steps 4-6 + 8a), run on the box after
# `W8_SHIP=0 scripts/ops/ladder_deploy.sh PORT HOST SCRATCH`. Status lines: /workspace/ladder/status.
#   setup: data (base D + the six coverage families DX) and CARLA in parallel
#   audit (leak check, merged dir DC) -> cache D (+ PCA-256 map) -> cache DX (PCA in D's basis) -> targets (_visns on D, all objects on DX)
#   cut-in log-side check (CPU, background) | detector (12 epochs, best by val F1) -> export tokens for D and DX
#   r1_s0, r1_s1 (RGB + 20 detector tokens <= 30 m) and r0ft (the same recipe without objects) side by side on the one GPU
#   gate: R1's val waypoint ADE must beat r0ft's; then NL lanes drive LANE_ARMS (default r1_s0 and r0ft) on all 220 routes, CARLA Epic
# R1 is an RGB-only agent: the detector runs on the frozen encoder's own map, live (single RGB camera at inference, user 2026-11-18).
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
NL=${NL:-4}
LANE_ARMS=${LANE_ARMS:-"r1_s0 r0ft"}
CORES=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null)   # the cgroup quota, not nproc
[ "${CORES:-0}" -gt 0 ] 2>/dev/null || CORES=$(nproc)
F="CARLA_QUALITY=Epic"
L=/workspace/ladder
B=scripts/ops/ladder_box.sh
mark "R1_CHAIN_START NL=$NL LANE_ARMS=$LANE_ARMS"
grep -q SETUP_START $S || bash $B setup &
until [ -f /workspace/checkpoints/E15/READY ]; do sleep 20; done
mark "CKPTS_READY"

# CARLA render check as soon as the simulator is installed (the lanes start hours later; a box that cannot render is found early)
( until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done
  grep -q CARLA_PROV_END $S || { mark "RENDER_SKIPPED carla"; exit 1; }
  bash /workspace/carla_render_test.sh > $L/render_test.log 2>&1
  grep -q RENDER_OK $L/render_test.log && mark "$(grep -o 'RENDER_OK.*' $L/render_test.log | head -1)" || mark "RENDER_FAILED (see $L/render_test.log)" ) &

# training side
until grep -q -E "DATA_DONE|DATA_FAILED" $S; do sleep 30; done
grep -q DATA_DONE $S || { mark "CHAIN_STOP data"; exit 1; }
grep -q AUDIT_DONE $S || bash $B audit || { mark "CHAIN_STOP audit"; exit 1; }
until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done       # caches and training run in venv_carla
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP carla venv"; exit 1; }
grep -q "CACHE_DONE D " $S || bash $B cache D || { mark "CHAIN_STOP cache D"; exit 1; }
grep -q "CACHE_DONE DX" $S || bash $B cache DX || { mark "CHAIN_STOP cache DX"; exit 1; }
grep -q TARGETS_DONE $S || bash $B targets_r1 || { mark "CHAIN_STOP targets"; exit 1; }
( bash $B cutin ) &
grep -q DETECTOR_DONE $S || bash $B detector || { mark "CHAIN_STOP detector"; exit 1; }
grep -q EXPORT_DONE $S || bash $B export || { mark "CHAIN_STOP export"; exit 1; }
export NW=$(( CORES / 4 > 6 ? 6 : (CORES / 4 > 2 ? CORES / 4 : 2) ))
for A in r1_s0 r1_s1 r0ft; do
  [ -f /workspace/checkpoints/r1d_$A/TRAINED ] || ( bash $B train $A ) &
  sleep 60
done
wait_trained() { for A in r1_s0 r1_s1 r0ft; do [ -f /workspace/checkpoints/r1d_$A/TRAINED ] || grep -q "TRAIN_END $A " $S || return 1; done; }
until wait_trained; do sleep 60; done
ok=""; for A in r1_s0 r1_s1 r0ft; do [ -f /workspace/checkpoints/r1d_$A/TRAINED ] && ok="$ok $A"; done
mark "TRAINED:$ok"
bash $B gate
grep -q GATE_PASS $S || { mark "CHAIN_STOP gate (R1 does not beat its RGB control offline: no CARLA run; save and destroy)"; exit 1; }

# closed loop
grep -q RENDER_OK $S || { mark "CHAIN_STOP no render"; exit 1; }
SPECS=()
for A in $LANE_ARMS; do
  [ -f /workspace/checkpoints/r1d_$A/TRAINED ] && SPECS+=("r1d_$A:r1d_$A/model_epoch_030.pth:$L/ids220.txt:$F")
done
[ ${#SPECS[@]} -gt 0 ] || { mark "CHAIN_STOP no lane arm trained"; exit 1; }
bash $B lanes $NL "${SPECS[@]}"
mark R1_LANES_STARTED
