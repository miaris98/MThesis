#!/usr/bin/env bash
# Generic CARLA evaluation box (2026-11-21; first use: the screen of the 11-21 arms, docs/ops/2026-11-21_memorization_ladder.md and TODO
# A85-A87). Run on the box after `scripts/ops/box_deploy.sh PORT HOST MANIFEST`, which ships the checkpoints (each dir gets a READY marker),
# the route lists, $WORKSPACE/ladder/specs.txt and $WORKSPACE/carla_render_test.sh. No data, no training.
#   specs.txt: one job per line, LABEL:CKPT_REL:ROUTEFILE:FLAGS (FLAGS = comma-separated K=V, e.g. CARLA_QUALITY=Epic,WOR_CONTROLLER=brake);
#              lines starting with # are skipped. Each lane runs the jobs in file order, every route list split round-robin over the lanes.
#   READY_DIRS (env): the checkpoint dirs (under $CHECKPOINTS) whose READY marker the chain waits for.
# Status: $WORKSPACE/ladder/status. CARLA -> render test -> NL lanes -> ALL_LANES_DONE.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
NL=${NL:-4}
mark "EVAL_CHAIN_START NL=$NL dirs: ${READY_DIRS:-none}"
grep -q SETUP_START $S || bash $B setup_carla &          # CARLA needs none of the shipped files
all_ready() { local d; for d in ${READY_DIRS:-}; do [ -f $CHECKPOINTS/$d/READY ] || return 1; done; [ -f $L/specs.txt ]; }
until all_ready; do sleep 20; done
mark CKPTS_READY
await CARLA_PROV_END CARLA_PROV_FAILED carla
if [ "${TFPP_ENV:-0}" = 1 ] && [ ! -f $L/tfpp_env.env ]; then
  ( PYDIR=$(dirname "$(uv python find 3.10 2>/dev/null)" 2>/dev/null); [ -n "$PYDIR" ] && export PATH=$PYDIR:$PATH
    bash scripts/setup/setup_tfpp_env.sh > $L/setup_tfpp.log 2>&1
    PYTHONPATH="$GARAGE/team_code:$GARAGE/leaderboard:$GARAGE/scenario_runner:$CARLA_DIR/PythonAPI:$CARLA_DIR/PythonAPI/carla:$MTHESIS_ROOT" \
      $WORKSPACE/venv_tfpp/bin/python -c "import importlib.util; s = importlib.util.spec_from_file_location('a', '$MTHESIS_ROOT/scripts/eval/tfpp_cam_agent.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('TFPP_CAM_AGENT_IMPORT_OK', m.get_entry_point())" >> $L/setup_tfpp.log 2>&1
    if grep -q TFPP_CAM_AGENT_IMPORT_OK $L/setup_tfpp.log; then echo "EVAL_PYTHON=$WORKSPACE/venv_tfpp/bin/python" > $L/tfpp_env.env; mark TFPP_ENV_OK
    else echo "LANE_SKIP=1" > $L/tfpp_env.env; mark "TFPP_ENV_FAILED (see $L/setup_tfpp.log)"; fi ) &
fi
render_test stop

mapfile -t SPECS < <(grep -v -E '^\s*(#|$)' $L/specs.txt)
for sp in "${SPECS[@]}"; do   # every checkpoint must exist before a lane is started on it
  IFS=: read -r _l ck _r _f <<< "$sp"
  [ -f $CHECKPOINTS/$ck ] || chain_stop "missing checkpoint $ck"
  [[ "$_f" == *EVAL_AGENT=* ]] || [ -f "$(dirname $CHECKPOINTS/$ck)/frozen_backbone.pth" ] || chain_stop "no frozen_backbone.pth next to $ck"   # 2026-11-22: a TF++ agent (EVAL_AGENT=) carries its encoder in the checkpoint (strict load)
done
bash $B lanes $NL "${SPECS[@]}"
mark "EVAL_LANES_STARTED ${#SPECS[@]} jobs"

until lanes_done $NL; do sleep 60; done
mark ALL_LANES_DONE
