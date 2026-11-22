#!/usr/bin/env bash
# A86 + A87 round 2 training box (2026-11-21; TODO_ACTIVE.md A86 / A87). A training-only box (no CARLA), run after
# `scripts/ops/box_deploy.sh PORT HOST MANIFEST` has shipped E15 and the evaluation route file. Status: /workspace/ladder/status.
#   setup (data + trainer packages) -> audit -> cache_full: D and DX at 9x24 plus the exact 4x4 from the same encoder pass, sharded over
#   every GPU -> the new tests on this box -> trainers (each pinned to a GPU):
#     a86_tfpp_cam  TF++ camera-only on our sensor / resolution / frozen encoder / frames (TF++'s recipe, 31 epochs)
#     a87_fm        our network on the whole 9x24 map at 1512 channels (216 camera tokens)
#     a87_tp        our network + TF++'s target point and next target point (route_mode dense_tp)
#     a87_moelat    w4's experts restricted to sideways corrections (moe_lateral_only)
#   the three a87 arms use mem_L0's settings (E15 -> 30, fresh AdamW, val_every 1, train_eval_frames 20000): mem_L0 (11-21) is the control
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
B=scripts/ops/ladder_box.sh; L=/workspace/ladder
export NO_CARLA=1 PYC=/venv/main/bin/python PYTHONPATH=/workspace/MThesis
mark "TFPP_CHAIN_START"
grep -q SETUP_START $S || bash $B setup &
until [ -f /workspace/checkpoints/E15/READY ] && [ -f $L/bench2drive220.xml ]; do sleep 20; done
mark CKPTS_READY
until grep -q -E "DATA_DONE|DATA_FAILED" $S; do sleep 30; done
grep -q DATA_DONE $S || { mark "CHAIN_STOP data"; exit 1; }
until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP trainer packages"; exit 1; }
grep -q AUDIT_DONE $S || bash $B audit || { mark "CHAIN_STOP audit"; exit 1; }
grep -q "CACHE_FULL_DONE DX" $S || bash $B cache_full || { mark "CHAIN_STOP cache_full"; exit 1; }

if ! grep -q BOX_TESTS_OK $S; then
  $PYC -m pytest -q -p no:cacheprovider tests/test_tfpp_cam.py tests/test_dense_tp_route.py tests/test_moe_lateral_and_logging.py \
      tests/test_memorization_regularizers.py > $L/box_tests.log 2>&1 \
    && mark "BOX_TESTS_OK $(tail -1 $L/box_tests.log)" || { mark "CHAIN_STOP box tests ($(tail -1 $L/box_tests.log))"; exit 1; }
fi

NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
G_FM=0; G_REST=$(( NG > 1 ? 1 : 0 ))   # the full-map arm (216 tokens, ~13x the attention) gets a GPU of its own when there are two
run() {   # arm gpu workers
  local A=$1 G=$2 W=$3
  if [ $A = a86_tfpp_cam ]; then
    [ -f /workspace/checkpoints/a86_tfpp_cam/TRAINED ] || ( CUDA_VISIBLE_DEVICES=$G NW=$W bash $B train_tfpp ) &
  else
    [ -f /workspace/checkpoints/$A/TRAINED ] || ( CUDA_VISIBLE_DEVICES=$G NW=$W bash $B train $A ) &
  fi
  sleep 30
}
run a87_fm $G_FM 6
run a86_tfpp_cam $G_REST 6
run a87_tp $G_REST 3
run a87_moelat $G_REST 3
wait
mark "TFPP_DONE trained:$(for A in a86_tfpp_cam a87_fm a87_tp a87_moelat; do [ -f /workspace/checkpoints/$A/TRAINED ] && printf ' %s' $A; done)"
