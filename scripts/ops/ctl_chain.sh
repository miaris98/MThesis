#!/usr/bin/env bash
# Controller study box (2026-11-20; plan and results: docs/ops/2026-11-20_controller_and_moe.md), run on the box after
# `scripts/ops/box_deploy.sh PORT HOST MANIFEST` (R1 seeds 0 and 1 from the 11-19 archive, the route lists). No data, no training.
# Status lines: /workspace/ladder/status. CARLA only -> render test -> NL lanes; each lane runs these jobs in turn, every route list
# split round-robin over the lanes (CARLA Epic, as on 11-19):
#   ctl_scr_<c>   R1 seed 0 with controller c (pid, brake, pdm_long, pdm_lat, pdm: src/models/world_on_rails/pid_controller.py)
#                 on the stratified 80-route list (docs/ops/fast_eval/tw_seed0_lists.json "80")
#   ctl_cfm_pid   R1 seed 0 + pid on the other 140 routes        (with ctl_scr_pid: R1 + pid on all 220)
#   ctl_cfm_win   R1 seed 0 + the screen's best alternative on the same 140 (fresh routes: the screen picked it on the 80);
#                 waits for ctl_winner.env (ENV_FROM), written once every lane has finished its screen
#   ctl_r1s1      R1 seed 1 + pid on all 220 (the seed replicate of R1 seed 0 + pid)
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
NL=${NL:-4}; L=/workspace/ladder; B=scripts/ops/ladder_box.sh; Q="CARLA_QUALITY=Epic"
mark "CTL_CHAIN_START NL=$NL"
until [ -f /workspace/checkpoints/r1d_r1_s0/READY ] && [ -f /workspace/checkpoints/r1d_r1_s1/READY ]; do sleep 20; done
mark CKPTS_READY
grep -q CARLA_PROV_END $S || bash $B setup_carla
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP carla"; exit 1; }
bash /workspace/carla_render_test.sh > $L/render_test.log 2>&1
grep -q RENDER_OK $L/render_test.log && mark "$(grep -o 'RENDER_OK.*' $L/render_test.log | head -1)" \
  || { mark "CHAIN_STOP render (see $L/render_test.log)"; exit 1; }

R1=r1d_r1_s0/model_epoch_030.pth
SPECS=()
for m in pid:pid brake:brake pdmlong:pdm_long pdmlat:pdm_lat pdm:pdm; do
  SPECS+=("ctl_scr_${m%%:*}:$R1:$L/ids80.txt:$Q,WOR_CONTROLLER=${m#*:}")
done
SPECS+=("ctl_cfm_pid:$R1:$L/ids140.txt:$Q,WOR_CONTROLLER=pid")
SPECS+=("ctl_cfm_win:$R1:$L/ids140.txt:$Q,ENV_FROM=$L/ctl_winner.env")
SPECS+=("ctl_r1s1:r1d_r1_s1/model_epoch_030.pth:$L/ids220.txt:$Q,WOR_CONTROLLER=pid")
bash $B lanes $NL "${SPECS[@]}"
mark CTL_LANES_STARTED

# the screen's winner, once every lane has finished its last screen job (a crash continuation prints "done ctl_scr_pdm_L<k>r")
screen_done() { local k; for k in $(seq 1 $NL); do
  grep -qE "(done|skip) ctl_scr_pdm_L$k|PLAN FAILED for ctl_scr_pdm_L$k" /workspace/evlogs/ladder_L$k.out 2>/dev/null || return 1; done; }
until screen_done; do sleep 60; done
bash $B ctl_winner
[ -f $L/ctl_winner.env ] || { echo "LANE_SKIP=1" > $L/ctl_winner.env; mark "CTL_WINNER_FAILED (no pick; confirm run skipped)"; }

lanes_done() { local k; for k in $(seq 1 $NL); do
  grep -q "lane $((2000 + 10 * k))/GPU0 done" /workspace/evlogs/ladder_L$k.out 2>/dev/null || return 1; done; }
until lanes_done; do sleep 60; done
mark ALL_LANES_DONE
