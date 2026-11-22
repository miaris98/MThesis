#!/usr/bin/env bash
# Generic CARLA evaluation box (2026-11-21; first use: the screen of the 11-21 arms, docs/ops/2026-11-21_memorization_ladder.md and TODO
# A85-A87). Run on the box after `scripts/ops/box_deploy.sh PORT HOST MANIFEST`, which ships the checkpoints (each dir gets a READY marker),
# the route lists, /workspace/ladder/specs.txt and /workspace/carla_render_test.sh. No data, no training.
#   specs.txt: one job per line, LABEL:CKPT_REL:ROUTEFILE:FLAGS (FLAGS = comma-separated K=V, e.g. CARLA_QUALITY=Epic,WOR_CONTROLLER=brake);
#              lines starting with # are skipped. Each lane runs the jobs in file order, every route list split round-robin over the lanes.
#   READY_DIRS (env): the checkpoint dirs (under /workspace/checkpoints) whose READY marker the chain waits for.
# Status: /workspace/ladder/status. CARLA -> render test -> NL lanes -> ALL_LANES_DONE.
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
NL=${NL:-4}; L=/workspace/ladder; B=scripts/ops/ladder_box.sh
mark "EVAL_CHAIN_START NL=$NL dirs: ${READY_DIRS:-none}"
grep -q SETUP_START $S || bash $B setup_carla &          # CARLA needs none of the shipped files
all_ready() { local d; for d in ${READY_DIRS:-}; do [ -f /workspace/checkpoints/$d/READY ] || return 1; done; [ -f $L/specs.txt ]; }
until all_ready; do sleep 20; done
mark CKPTS_READY
until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP carla"; exit 1; }
# 2026-11-21 (A86): TFPP_ENV=1 builds venv_tfpp (TF++'s pinned stack) in the background and checks that the camera-only agent imports in it;
# the TF++ job's spec line waits on $L/tfpp_env.env (ENV_FROM), which carries EVAL_PYTHON, or LANE_SKIP=1 when the environment failed
if [ "${TFPP_ENV:-0}" = 1 ] && [ ! -f $L/tfpp_env.env ]; then
  ( PYDIR=$(dirname "$(uv python find 3.10 2>/dev/null)" 2>/dev/null); [ -n "$PYDIR" ] && export PATH=$PYDIR:$PATH
    bash scripts/setup/setup_tfpp_env.sh > $L/setup_tfpp.log 2>&1
    PYTHONPATH="/workspace/carla_garage/team_code:/workspace/carla_garage/leaderboard:/workspace/carla_garage/scenario_runner:/workspace/carla/PythonAPI:/workspace/carla/PythonAPI/carla:/workspace/MThesis" \
      /workspace/venv_tfpp/bin/python -c "import importlib.util; s = importlib.util.spec_from_file_location('a', '/workspace/MThesis/scripts/eval/tfpp_cam_agent.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('TFPP_CAM_AGENT_IMPORT_OK', m.get_entry_point())" >> $L/setup_tfpp.log 2>&1
    if grep -q TFPP_CAM_AGENT_IMPORT_OK $L/setup_tfpp.log; then echo "EVAL_PYTHON=/workspace/venv_tfpp/bin/python" > $L/tfpp_env.env; mark TFPP_ENV_OK
    else echo "LANE_SKIP=1" > $L/tfpp_env.env; mark "TFPP_ENV_FAILED (see $L/setup_tfpp.log)"; fi ) &
fi
bash /workspace/carla_render_test.sh > $L/render_test.log 2>&1
grep -q RENDER_OK $L/render_test.log && mark "$(grep -o 'RENDER_OK.*' $L/render_test.log | head -1)" \
  || { mark "CHAIN_STOP render (see $L/render_test.log)"; exit 1; }

mapfile -t SPECS < <(grep -v -E '^\s*(#|$)' $L/specs.txt)
for sp in "${SPECS[@]}"; do   # every checkpoint must exist before a lane is started on it
  IFS=: read -r _l ck _r _f <<< "$sp"
  [ -f /workspace/checkpoints/$ck ] || { mark "CHAIN_STOP missing checkpoint $ck"; exit 1; }
  [[ "$_f" == *EVAL_AGENT=* ]] || [ -f "$(dirname /workspace/checkpoints/$ck)/frozen_backbone.pth" ] || { mark "CHAIN_STOP no frozen_backbone.pth next to $ck"; exit 1; }   # 2026-11-22: a TF++ agent (EVAL_AGENT=) carries its encoder in the checkpoint (strict load)
done
bash $B lanes $NL "${SPECS[@]}"
mark "EVAL_LANES_STARTED ${#SPECS[@]} jobs"

lanes_done() { local k; for k in $(seq 1 $NL); do
  grep -q "lane $((2000 + 10 * k))/GPU0 done" /workspace/evlogs/ladder_L$k.out 2>/dev/null || return 1; done; }
until lanes_done; do sleep 60; done
mark ALL_LANES_DONE
