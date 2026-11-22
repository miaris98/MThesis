#!/usr/bin/env bash
# R1 + experts box (2026-11-20; plan and results: docs/ops/2026-11-20_controller_and_moe.md), run on the box after
# `scripts/ops/box_deploy.sh PORT HOST MANIFEST` (E15, R1 seed 0, the 11-19 detector, and the 11-19 PCA basis rebuilt from R1's
# buffers into /workspace/pooled_ov0, so the new cache's PCA map is the one the detector was trained on). Status: /workspace/ladder/status.
#   setup (data + CARLA) -> audit -> cache D (reuses the shipped basis) -> cache DX (D's basis) -> export (the shipped detector)
#   -> NL lanes and two trainers side by side:
#        r1w4_s0  R1 + w4's waypoint experts (16 experts, top-4), seed 0
#        r0ftf    the RGB control with a fresh AdamW (WOR_RESET_OPTIMIZER=1): every R1 arm restarts its optimizer, r0ft did not
#      each lane drives in turn: mx_r1s0 (R1 seed 0, this box's anchor), mx_r1w4, mx_r0ftf on all 220 routes, CARLA Epic, pid;
#      the last two wait for go_<arm> (ENV_FROM), written when their training ends (LANE_SKIP=1 if it failed)
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
NL=${NL:-4}; L=/workspace/ladder; B=scripts/ops/ladder_box.sh; Q="CARLA_QUALITY=Epic"
ARMS="r1w4_s0 r0ftf"
mark "MX_CHAIN_START NL=$NL"
grep -q SETUP_START $S || bash $B setup &       # data + CARLA need none of the shipped files: start while the deploy finishes
until [ -f /workspace/checkpoints/E15/READY ] && [ -f /workspace/checkpoints/r1d_r1_s0/READY ] && [ -f /workspace/checkpoints/det_visns/READY ] \
      && [ -f /workspace/pooled_ov0/READY ]; do sleep 20; done
mark CKPTS_READY

( until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done      # render check as soon as CARLA is in
  grep -q CARLA_PROV_END $S || { mark "RENDER_SKIPPED carla"; exit 1; }
  bash /workspace/carla_render_test.sh > $L/render_test.log 2>&1
  grep -q RENDER_OK $L/render_test.log && mark "$(grep -o 'RENDER_OK.*' $L/render_test.log | head -1)" || mark "RENDER_FAILED (see $L/render_test.log)" ) &

until grep -q -E "DATA_DONE|DATA_FAILED" $S; do sleep 30; done
grep -q DATA_DONE $S || { mark "CHAIN_STOP data"; exit 1; }
grep -q AUDIT_DONE $S || bash $B audit || { mark "CHAIN_STOP audit"; exit 1; }
until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP carla venv"; exit 1; }
grep -q "CACHE_DONE D " $S || bash $B cache D || { mark "CHAIN_STOP cache D"; exit 1; }
grep -q "PCA basis:" $L/cache_D.log && { mark "CHAIN_STOP cache D fitted a NEW PCA basis (the shipped one was not used)"; exit 1; }
grep -q "CACHE_DONE DX" $S || bash $B cache DX || { mark "CHAIN_STOP cache DX"; exit 1; }
grep -q EXPORT_DONE $S || bash $B export || { mark "CHAIN_STOP export"; exit 1; }
until grep -q -E "RENDER_OK|RENDER_FAILED|RENDER_SKIPPED" $S; do sleep 20; done
grep -q RENDER_OK $S || { mark "CHAIN_STOP no render"; exit 1; }

# the trainers first, the lanes after them: on the 46 GB L40S, 4 Epic lanes (up to 35 GB on 11-19) + 2 trainers (~7.5 GB) leave < 10% headroom
export NW=6
for A in $ARMS; do
  [ -f $L/go_$A ] && continue
  ( bash $B train $A
    if [ -f /workspace/checkpoints/r1d_$A/TRAINED ]; then echo "# trained $(date -u +%T)" > $L/go_$A; else echo "LANE_SKIP=1" > $L/go_$A; fi ) &
  sleep 60
done
all_go() { local A; for A in $ARMS; do [ -f $L/go_$A ] || return 1; done; }
until all_go; do sleep 60; done
bash $B valade
mark "TRAINED:$(for A in $ARMS; do [ -f /workspace/checkpoints/r1d_$A/TRAINED ] && printf ' %s' $A; done)"

SPECS=("mx_r1s0:r1d_r1_s0/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid"
       "mx_r1w4:r1d_r1w4_s0/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid,ENV_FROM=$L/go_r1w4_s0"
       "mx_r0ftf:r1d_r0ftf/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid,ENV_FROM=$L/go_r0ftf")
bash $B lanes $NL "${SPECS[@]}"
mark MX_LANES_STARTED

lanes_done() { local k; for k in $(seq 1 $NL); do
  grep -q "lane $((2000 + 10 * k))/GPU0 done" /workspace/evlogs/ladder_L$k.out 2>/dev/null || return 1; done; }
until lanes_done; do sleep 60; done
mark ALL_LANES_DONE
