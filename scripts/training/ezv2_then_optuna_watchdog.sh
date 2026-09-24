#!/usr/bin/env bash
# S-055 watchdog: nothing competes with the EfficientZero V2 baseline while it trains.
# Once its ez/train.py process exits:
#   1. evaluate its saved 10k / 20k checkpoints (scripts/eval/eval_ezv2_checkpoint.py: repo's own
#      ez.eval.eval, 10 episodes, 27,000-step cap) -> /workspace/ezv2_ckpt_eval/model_<N>.json
#   2. start the Optuna study (atari_qwen/training/optimize_mcts_optuna.py): 20k-step trials, 3
#      concurrent on the now-free GPU, stop target = EZ-V2's 20k score (S-055)
#   3. relaunch the study if the driver dies before the study is finished; the study lives in
#      sqlite (load_if_exists), so completed trials are kept and only the remainder runs.
# Status lines go to /workspace/watchdog.log (synced to E: by the 10-minute sync loop).
#
# usage (on the box): setsid nohup bash ezv2_then_optuna_watchdog.sh > /workspace/watchdog.log 2>&1 &
set -u
EZ_ROOT=/workspace/EfficientZeroV2
EVAL_OUT=/workspace/ezv2_ckpt_eval
MTHESIS=/workspace/MThesis
MAX_RESTARTS=${MAX_RESTARTS:-5}
log() { echo "$(date '+%F %T') $*"; }

log "waiting for EZ-V2 training (ez/train.py) to exit"
while pgrep -f "ez/train.py" >/dev/null; do sleep 120; done
log "EZ-V2 training finished"

RUN_DIR=$(ls -td "$EZ_ROOT"/results/Atari/Breakout/*/ | head -1)
mkdir -p "$EVAL_OUT"
cd "$EZ_ROOT"
for s in 10000 20000; do
  if [ ! -f "$EVAL_OUT/model_$s.json" ]; then
    log "evaluating EZ-V2 model_$s.p"
    WANDB_MODE=offline /workspace/venv_ezv2/bin/python "$MTHESIS/scripts/eval/eval_ezv2_checkpoint.py" \
      "$RUN_DIR/models/model_$s.p" "$EVAL_OUT/model_$s.json" 10 2>&1 | grep -E "EZV2_EVAL|Error|Traceback"
  fi
done
if [ ! -f "$EVAL_OUT/model_20000.json" ]; then
  log "FATAL: no 20k evaluation produced - not starting the study without its target"
  exit 1
fi
for s in 10000 20000; do
  log "EZ-V2 ${s} score: $(python3 -c "import json;d=json.load(open('$EVAL_OUT/model_$s.json'));print(round(d['mean'],2),'+/-',round(d['se'],2),'SE')")"
done
TARGET=$(python3 -c "import json;print(json.load(open('$EVAL_OUT/model_20000.json'))['mean'])")
log "study target = EZ-V2 20k score $TARGET"

cd "$MTHESIS"
# Utilisation record for the study (VRAM headroom rule: 10-15%); 5-minute samples.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader -l 300 \
  >> /workspace/gpu_util.log 2>&1 &
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  log "starting Optuna study (attempt $attempt/$MAX_RESTARTS)"
  PYTHONPATH="$MTHESIS" /venv/main/bin/python -u atari_qwen/training/optimize_mcts_optuna.py \
    --n-trials 50 --n-jobs 3 --cores-per-job 12 --first-core 0 --trial-timeout-h 2.5 \
    --target-score "$TARGET" >> /workspace/optuna_mcts_driver.log 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    log "study finished normally"
    exit 0
  fi
  log "study driver exited rc=$rc; restarting in 60 s (finished trials are kept in sqlite)"
  sleep 60
done
log "giving up after $MAX_RESTARTS driver restarts"
exit 1
