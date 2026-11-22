#!/usr/bin/env bash
# R1 + experts box (2026-11-20; plan and results: docs/ops/2026-11-20_controller_and_moe.md), run on the box after
# `scripts/ops/box_deploy.sh PORT HOST MANIFEST` (E15, R1 seed 0, the 11-19 detector, and the 11-19 PCA basis rebuilt from R1's
# buffers into $WORKSPACE/pooled_ov0, so the new cache's PCA map is the one the detector was trained on). Status: $WORKSPACE/ladder/status.
#   setup (data + CARLA) -> audit -> cache D (reuses the shipped basis) -> cache DX (D's basis) -> export (the shipped detector)
#   -> NL lanes and two trainers side by side:
#        r1w4_s0  R1 + w4's waypoint experts (16 experts, top-4), seed 0
#        r0ftf    the RGB control with a fresh AdamW (WOR_RESET_OPTIMIZER=1): every R1 arm restarts its optimizer, r0ft did not
#      each lane drives in turn: mx_r1s0 (R1 seed 0, this box's anchor), mx_r1w4, mx_r0ftf on all 220 routes, CARLA Epic, pid;
#      the last two wait for go_<arm> (ENV_FROM), written when their training ends (LANE_SKIP=1 if it failed)
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
NL=${NL:-4}; Q="CARLA_QUALITY=Epic"
ARMS="r1w4_s0 r0ftf"
mark "MX_CHAIN_START NL=$NL"
grep -q SETUP_START $S || bash $B setup &       # data + CARLA need none of the shipped files: start while the deploy finishes
wait_files $CHECKPOINTS/E15/READY $CHECKPOINTS/r1d_r1_s0/READY $CHECKPOINTS/det_visns/READY $WORKSPACE/pooled_ov0/READY
mark CKPTS_READY

render_when_carla &      # render check as soon as CARLA is in

await DATA_DONE DATA_FAILED data
step AUDIT_DONE audit
await CARLA_PROV_END CARLA_PROV_FAILED "carla venv"
step "CACHE_DONE D " cache D
grep -q "PCA basis:" $L/cache_D.log && chain_stop "cache D fitted a NEW PCA basis (the shipped one was not used)"
step "CACHE_DONE DX" cache DX
step EXPORT_DONE export
CHAIN_POLL=20 await RENDER_OK "RENDER_FAILED|RENDER_SKIPPED" "no render"

export NW=6
for A in $ARMS; do
  [ -f $L/go_$A ] && continue
  ( bash $B train $A
    if [ -f $CHECKPOINTS/r1d_$A/TRAINED ]; then echo "# trained $(date -u +%T)" > $L/go_$A; else echo "LANE_SKIP=1" > $L/go_$A; fi ) &
  sleep 60
done
all_go() { local A; for A in $ARMS; do [ -f $L/go_$A ] || return 1; done; }
until all_go; do sleep 60; done
bash $B valade
mark "TRAINED:$(trained_list r1d_ $ARMS)"

SPECS=("mx_r1s0:r1d_r1_s0/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid"
       "mx_r1w4:r1d_r1w4_s0/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid,ENV_FROM=$L/go_r1w4_s0"
       "mx_r0ftf:r1d_r0ftf/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid,ENV_FROM=$L/go_r0ftf")
bash $B lanes $NL "${SPECS[@]}"
mark MX_LANES_STARTED

until lanes_done $NL; do sleep 60; done
mark ALL_LANES_DONE
