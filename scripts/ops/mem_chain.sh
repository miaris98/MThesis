#!/usr/bin/env bash
# Memorization ladder, stage 1 (2026-11-21; plan and results: docs/ops/2026-11-21_memorization_ladder.md). A training-only box (no CARLA),
# run after `scripts/ops/box_deploy.sh PORT HOST MANIFEST` has shipped E15 and the evaluation route file. Status: /workspace/ladder/status.
#   setup (data into /venv/main's trainer, no CARLA) -> audit (leak-free data, as every arm since 11-18) -> cache D, DX (no PCA map)
#   -> the new unit tests on this box's GPU -> 8 trainers side by side (spread over the GPUs, one CUDA_VISIBLE_DEVICES each):
#        mem_L0 / mem_L0w   r0ftf's recipe (E15 -> epoch 30, fresh AdamW), without / with w4's experts
#        mem_L1..L3[w]      + feature noise 0.1 / 0.25 / 0.5 x token RMS and token dropout 10 / 20 / 30 % (training only)
#   -> memtable (clean-train vs held-out ADE per arm, experts' effect per level) -> MEM_DONE
set -uo pipefail
S=/workspace/ladder/status; mkdir -p /workspace/ladder
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
B=scripts/ops/ladder_box.sh; L=/workspace/ladder
export NO_CARLA=1 PYC=/venv/main/bin/python D_PCA=0 DX_PCA=0 PYTHONPATH=/workspace/MThesis
ARMS=${ARMS:-"mem_L0 mem_L0w mem_L1 mem_L1w mem_L2 mem_L2w mem_L3 mem_L3w"}
mark "MEM_CHAIN_START arms: $ARMS"
grep -q SETUP_START $S || bash $B setup &       # data + packages need none of the shipped files
until [ -f /workspace/checkpoints/E15/READY ] && [ -f $L/bench2drive220.xml ]; do sleep 20; done
mark CKPTS_READY

until grep -q -E "DATA_DONE|DATA_FAILED" $S; do sleep 30; done
grep -q DATA_DONE $S || { mark "CHAIN_STOP data"; exit 1; }
until grep -q -E "CARLA_PROV_END|CARLA_PROV_FAILED" $S; do sleep 30; done
grep -q CARLA_PROV_END $S || { mark "CHAIN_STOP trainer packages"; exit 1; }
grep -q AUDIT_DONE $S || bash $B audit || { mark "CHAIN_STOP audit"; exit 1; }
grep -q "CACHE_DONE D " $S || bash $B cache D || { mark "CHAIN_STOP cache D"; exit 1; }
grep -q "CACHE_DONE DX" $S || bash $B cache DX || { mark "CHAIN_STOP cache DX"; exit 1; }

# the new code on this box's GPU and packages before eight runs depend on it
if ! grep -q BOX_TESTS_OK $S; then
  $PYC -m pytest -q -p no:cacheprovider tests/test_memorization_regularizers.py > $L/box_tests.log 2>&1 \
    && mark "BOX_TESTS_OK $(tail -1 $L/box_tests.log)" || { mark "CHAIN_STOP box tests ($(tail -1 $L/box_tests.log))"; exit 1; }
fi

NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
export NW=${NW:-3}
i=0
for A in $ARMS; do
  LAB=$A$([ "${SEED:-0}" = 0 ] || echo _s${SEED})
  if [ -f /workspace/checkpoints/$LAB/TRAINED ]; then i=$((i + 1)); continue; fi
  ( CUDA_VISIBLE_DEVICES=$((i % NG)) bash $B train $A ) &
  i=$((i + 1)); sleep 30
done
wait
bash $B memtable
mark "MEM_DONE trained:$(for A in $ARMS; do LAB=$A$([ "${SEED:-0}" = 0 ] || echo _s${SEED}); [ -f /workspace/checkpoints/$LAB/TRAINED ] && printf ' %s' $LAB; done)"
