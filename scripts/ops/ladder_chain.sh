#!/usr/bin/env bash
# Teacher ladder (2026-11-18), run on the box after ladder_deploy.sh: the whole day in order. Status: /workspace/ladder/status.
#   setup (data + CARLA in parallel)
#   early eval, as soon as CARLA renders: w8_teacher on all 220 routes (the 30 uncovered-family routes first, token dump on)
#   training side: audit -> caches D, DX -> token targets -> t_vis, t_visns, t_cov side by side on the one GPU
#   late eval, when all three are trained and the early lanes are idle: NL lanes, each running t_vis (220), t_visns (220), t_cov (30 routes)
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
NL=${NL:-4}
CORES=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null)   # the cgroup quota, not nproc
[ "${CORES:-0}" -gt 0 ] 2>/dev/null || CORES=$(nproc)                                                                 # (2026-11-18 box: 128 threads, quota 20)
F="CARLA_QUALITY=Epic"
L=/workspace/ladder
lanes_idle() { ! pgrep -f "queue_A9c.sh 20[0-9]0 " > /dev/null; }
mark "CHAIN_START NL=$NL"
grep -q SETUP_START $S || bash scripts/ops/ladder_box.sh setup &   # (a restarted chain does not set up twice) data + CARLA need no checkpoint; the uploads (READY after the sha256 check) are waited for below
for n in E15 w8_teacher; do until [ -f /workspace/checkpoints/$n/READY ]; do sleep 20; done; done
mark "CKPTS_READY"

# early eval
( until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done
  grep -q CARLA_PROV_END $S || { mark "EARLY_STOP carla"; exit 1; }
  bash /workspace/carla_render_test.sh > $L/render_test.log 2>&1
  grep -q RENDER_OK $L/render_test.log || { mark "RENDER_FAILED (see $L/render_test.log)"; exit 1; }
  mark "$(grep -o 'RENDER_OK.*' $L/render_test.log | head -1)"
  # the 30 uncovered-family routes with the token dump first, then w8_teacher's other 190 routes: R4 on this host (same-host reference
  # for R2/R3, and the teacher's run-to-run spread against the 2026-11-17 run) while the data downloads and the arms train
  awk 'NR == FNR { f[$1]; next } !($1 in f)' $L/fam_routes.txt $L/ids220.txt > $L/rest190.txt
  bash scripts/ops/ladder_box.sh lanes $NL "w8dump:w8_teacher/model_epoch_030.pth:$L/fam_routes.txt:$F,B2D_PRIV_DUMP=$L/privdump/w8"     "w8rest:w8_teacher/model_epoch_030.pth:$L/rest190.txt:$F"
  mark EARLY_LANES_STARTED ) &

# training side
until grep -q -E "DATA_DONE|DATA_FAILED" $S; do sleep 30; done
grep -q DATA_DONE $S || { mark "CHAIN_STOP data"; exit 1; }
bash scripts/ops/ladder_box.sh audit || { mark "CHAIN_STOP audit"; exit 1; }
until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done       # caches and training run in venv_carla
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP carla venv"; exit 1; }
bash scripts/ops/ladder_box.sh cache D || { mark "CHAIN_STOP cache D"; exit 1; }
bash scripts/ops/ladder_box.sh cache DX || { mark "CHAIN_STOP cache DX"; exit 1; }
bash scripts/ops/ladder_box.sh targets || { mark "CHAIN_STOP targets"; exit 1; }
export NW=$(( CORES / 4 > 6 ? 6 : (CORES / 4 > 2 ? CORES / 4 : 2) ))
for A in t_vis t_visns t_cov; do ( bash scripts/ops/ladder_box.sh train $A ) & sleep 60; done
until grep -q "TRAIN_END t_visns" $S && grep -q "TRAIN_END t_cov" $S && grep -q "TRAIN_END t_vis " $S; do sleep 60; done
ok=""; for A in t_vis t_visns t_cov; do [ -f /workspace/checkpoints/ladder_$A/TRAINED ] && ok="$ok $A"; done
mark "TRAINED:$ok"
[ -n "$ok" ] || { mark "CHAIN_STOP no arm trained"; exit 1; }

# late eval
until grep -q -E "EARLY_LANES_STARTED|RENDER_FAILED|EARLY_STOP" $S; do sleep 30; done
grep -q EARLY_LANES_STARTED $S || { mark "CHAIN_STOP no CARLA"; exit 1; }
sleep 120; until lanes_idle; do sleep 60; done
SPECS=()
for A in $ok; do
  case $A in
    t_vis|t_visns) SPECS+=("ladder_$A:ladder_$A/model_epoch_030.pth:$L/ids220.txt:$F") ;;
    t_cov)         SPECS+=("ladder_t_cov:ladder_t_cov/model_epoch_030.pth:$L/fam_routes.txt:$F,B2D_PRIV_DUMP=$L/privdump/t_cov") ;;
  esac
done
bash scripts/ops/ladder_box.sh lanes $NL "${SPECS[@]}"
mark LATE_LANES_STARTED
