#!/usr/bin/env bash
# Builds venv_tfpp: the environment TransFuser++ is run in as a *reference* model.
#
# usage: setup_tfpp_env.sh          (idempotent - exits early if the agent already imports)
#
# WHY A SECOND VENV RATHER THAN ADDING TO venv_carla
# --------------------------------------------------
# venv_carla runs this project's own WoR agent on numpy 2.2.6 / torch 2.14, and works. TF++
# cannot join it: carla_garage pins numpy==1.26.4, and `team_code/data.py` imports `imgaug`
# at module scope, which calls `np.sctypes` - removed in NumPy 2.0. So the two agents cannot
# share one interpreter without downgrading the environment that currently works.
#
# The tempting shortcut is to stub `np.sctypes`, since imgaug is only used for *training-time*
# augmentation (data.py:1218-1230) and inference touches `CARLA_Data` solely for
# `lidar_to_histogram_features` (sensor_agent.py:485). That shortcut is refused deliberately.
# TF++ exists in this project for exactly one purpose: to reproduce a *published* number, so
# that a mismatch indicts our harness rather than the model. Run it on library versions its
# authors never tested and that property is gone - a low score becomes un-attributable between
# "our harness is wrong" and "torch 2.14 changed something". Environment fidelity IS the
# measurement here, not incidental packaging.
#
# run_leaderboard_official.sh selects the interpreter with EVAL_PYTHON, so the harness itself
# is unchanged - only which python executes it.
set -u

VENV=/workspace/venv_tfpp
GARAGE=/workspace/carla_garage
AGENT="$GARAGE/team_code/sensor_agent.py"

probe() {
  PYTHONPATH="$GARAGE/team_code:$GARAGE/leaderboard:$GARAGE/scenario_runner:/workspace/carla/PythonAPI:/workspace/carla/PythonAPI/carla" \
  "$VENV/bin/python" -c "
import importlib.util, sys
import leaderboard.leaderboard_evaluator
spec = importlib.util.spec_from_file_location('_probe', '$AGENT')
m = importlib.util.module_from_spec(spec); sys.modules['_probe'] = m
spec.loader.exec_module(m)
print('TFPP_AGENT_IMPORT_OK', m.get_entry_point())
" 2>&1
}

if [ -x "$VENV/bin/python" ] && probe | grep -q TFPP_AGENT_IMPORT_OK; then
  echo "venv_tfpp already present and TF++ agent imports - nothing to do"
  exit 0
fi

echo "=== creating $VENV (python3.10) ==="
python3.10 -m venv "$VENV"
# setuptools<81 is NOT cosmetic. leaderboard_evaluator.py:21 does `import pkg_resources`,
# which setuptools 81 removed - a bare `--upgrade setuptools` builds a venv where the
# evaluator cannot start at all ("ModuleNotFoundError: No module named 'pkg_resources'").
# venv_carla only escapes this by happening to hold an older setuptools; a fresh venv gets
# the current one. The deprecation notice naming this exact ceiling has been printing in
# every run log since the first pilot.
"$VENV/bin/pip" install -q --upgrade pip wheel "setuptools<81"

# carla_garage/team_code/requirements.txt pins. numpy is the load-bearing one (imgaug), the
# rest are kept at their pinned versions so this matches the environment TF++'s published
# numbers were produced in as closely as we can reach.
echo "=== installing carla_garage pinned stack (this downloads ~2.5GB of torch) ==="
"$VENV/bin/pip" install -q \
  "numpy==1.26.4" \
  "torch==2.5.0" "torchvision==0.20.0" \
  "scipy==1.14.1" "imgaug==0.4.0" "torchmetrics==0.11.0" \
  timm einops laspy scikit-learn jsonpickle ujson filterpy opencv-python matplotlib \
  rdp diskcache transformers tqdm shapely
# The full third-party set was enumerated by scanning every `import`/`from` in team_code/*.py
# and testing each against the venv, rather than discovered one CARLA cold start at a time.
# `agents` shows up in such a scan but is NOT a pip package - it is CARLA's own
# PythonAPI/carla/agents, supplied via PYTHONPATH.

# The leaderboard evaluator itself runs inside this venv too (the agent is imported in-process),
# so it needs the same vanilla runtime set venv_carla required - including the py-trees 0.8.3
# pin, which is not optional: the vendored leaderboard/scenario_runner use the pre-1.0 API and
# a bare `pip install py-trees` silently installs 2.6.0 (S-011).
echo "=== installing leaderboard/scenario_runner runtime deps ==="
"$VENV/bin/pip" install -q \
  six "py-trees==0.8.3" Shapely xmlschema ephem tabulate psutil pygame pexpect \
  dictor transforms3d simple-watchdog-timer requests

echo "=== installing the CARLA python client ==="
"$VENV/bin/pip" install -q "carla==0.9.15"

# $GARAGE is a plain checkout of autonomousvision/carla_garage, not cloned by this script and
# not something we push local commits to (it's a nested git repo of its own). The diagnostic
# instrumentation added to sensor_agent.py's stuck/creep-recovery logging on 2026-09-15 - lidar
# point counts in the safety box, nearest tracked actor, edge-triggered stop-sign logging - lives
# only in this project's tracked patches/ dir (Carla-utils/ itself is gitignored) and has to be
# re-applied to every fresh $GARAGE checkout by hand.
PATCH="$(dirname "$0")/patches/carla_garage_sensor_agent_diagnostics.patch"
if [ -f "$PATCH" ] && ! grep -q '_nearest_actor_summary' "$AGENT" 2>/dev/null; then
  echo "=== applying $PATCH to $AGENT ==="
  git -C "$GARAGE" apply "$PATCH" \
    || echo "WARNING: patch did not apply cleanly - sensor_agent.py has diverged, apply by hand"
fi

echo "=== verifying the TF++ agent imports ==="
OUT="$(probe)"
echo "$OUT" | tail -15
if ! echo "$OUT" | grep -q TFPP_AGENT_IMPORT_OK; then
  echo "FATAL: TF++ agent still does not import in $VENV (see trace above)"
  exit 1
fi
"$VENV/bin/python" -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "=== venv_tfpp ready ==="
echo "run TF++ with:"
echo "  EVAL_PYTHON=$VENV/bin/python \\"
echo "  EVAL_AGENT=$AGENT \\"
echo "  EVAL_AGENT_CONFIG=/workspace/tfpp_pretrained/pretrained_models/town13_withheld \\"
echo "  bash run_leaderboard_official.sh <label> tfpp - <routes.xml> <subset> <port> <tm_port>"
