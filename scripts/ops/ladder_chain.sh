#!/usr/bin/env bash
# Teacher ladder (2026-11-18), run on the box after `SCRATCH=DIR scripts/ops/box_deploy.sh PORT HOST scripts/ops/manifests/ladder.txt`: the whole
# day in order. Status: $WORKSPACE/ladder/status.
#   setup (data + CARLA in parallel)
#   early eval, as soon as CARLA renders: w8_teacher on all 220 routes (the 30 uncovered-family routes first, token dump on)
#   training side: audit -> caches D, DX -> token targets -> t_vis, t_visns, t_cov side by side on the one GPU
#   late eval, when all three are trained and the early lanes are idle: NL lanes, each running t_vis (220), t_visns (220), t_cov (30 routes)
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
NL=${NL:-4}
CORES=$(cgroup_cores)   # the cgroup quota, not nproc (2026-11-18 box: 128 threads, quota 20)
F="CARLA_QUALITY=Epic"
lanes_idle() { ! pgrep -f "queue_A9c.sh 20[0-9]0 " > /dev/null; }
mark "CHAIN_START NL=$NL"
grep -q SETUP_START $S || bash $B setup &   # (a restarted chain does not set up twice) data + CARLA need no checkpoint; the uploads (READY after the sha256 check) are waited for below
wait_files $CHECKPOINTS/E15/READY $CHECKPOINTS/w8_teacher/READY
mark "CKPTS_READY"

( wait_for CARLA_PROV_END CARLA_PROV_FAILED || { mark "EARLY_STOP carla"; exit 1; }
  render_test || exit 1
  # the 30 uncovered-family routes with the token dump first, then w8_teacher's other 190 routes: R4 on this host (same-host reference
  # for R2/R3, and the teacher's run-to-run spread against the 2026-11-17 run) while the data downloads and the arms train
  awk 'NR == FNR { f[$1]; next } !($1 in f)' $L/fam_routes.txt $L/ids220.txt > $L/rest190.txt
  bash $B lanes $NL "w8dump:w8_teacher/model_epoch_030.pth:$L/fam_routes.txt:$F,B2D_PRIV_DUMP=$L/privdump/w8"     "w8rest:w8_teacher/model_epoch_030.pth:$L/rest190.txt:$F"
  mark EARLY_LANES_STARTED ) &

await DATA_DONE DATA_FAILED data
run_or_stop audit
await CARLA_PROV_END CARLA_PROV_FAILED "carla venv"       # caches and training run in venv_carla
run_or_stop cache D
run_or_stop cache DX
run_or_stop targets
export NW=$(( CORES / 4 > 6 ? 6 : (CORES / 4 > 2 ? CORES / 4 : 2) ))
for A in t_vis t_visns t_cov; do ( bash $B train $A ) & sleep 60; done
until grep -q "TRAIN_END t_visns" $S && grep -q "TRAIN_END t_cov" $S && grep -q "TRAIN_END t_vis " $S; do sleep 60; done
ok=$(trained_list ladder_ t_vis t_visns t_cov)
mark "TRAINED:$ok"
[ -n "$ok" ] || chain_stop "no arm trained"

wait_for EARLY_LANES_STARTED "RENDER_FAILED|EARLY_STOP" || chain_stop "no CARLA"
sleep 120; until lanes_idle; do sleep 60; done
SPECS=()
for A in $ok; do
  case $A in
    t_vis|t_visns) SPECS+=("ladder_$A:ladder_$A/model_epoch_030.pth:$L/ids220.txt:$F") ;;
    t_cov)         SPECS+=("ladder_t_cov:ladder_t_cov/model_epoch_030.pth:$L/fam_routes.txt:$F,B2D_PRIV_DUMP=$L/privdump/t_cov") ;;
  esac
done
bash $B lanes $NL "${SPECS[@]}"
mark LATE_LANES_STARTED
