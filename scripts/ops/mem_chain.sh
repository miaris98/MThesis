#!/usr/bin/env bash
# Memorization ladder, stage 1 (2026-11-21; plan and results: docs/ops/2026-11-21_memorization_ladder.md). A training-only box (no CARLA),
# run after `scripts/ops/box_deploy.sh PORT HOST MANIFEST` has shipped E15 and the evaluation route file. Status: $WORKSPACE/ladder/status.
#   setup (data into /venv/main's trainer, no CARLA) -> audit (leak-free data, as every arm since 11-18) -> cache D, DX (no PCA map)
#   -> the new unit tests on this box's GPU -> 8 trainers side by side (spread over the GPUs, one CUDA_VISIBLE_DEVICES each):
#        mem_L0 / mem_L0w   r0ftf's recipe (E15 -> epoch 30, fresh AdamW), without / with w4's experts
#        mem_L1..L3[w]      + feature noise 0.1 / 0.25 / 0.5 x token RMS and token dropout 10 / 20 / 30 % (training only)
#   -> memtable (clean-train vs held-out ADE per arm, experts' effect per level) -> MEM_DONE
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
export NO_CARLA=1 PYC D_PCA=0 DX_PCA=0 PYTHONPATH=$MTHESIS_ROOT
ARMS=${ARMS:-"mem_L0 mem_L0w mem_L1 mem_L1w mem_L2 mem_L2w mem_L3 mem_L3w"}
SFX=$([ "${SEED:-0}" = 0 ] || echo _s${SEED})
mark "MEM_CHAIN_START arms: $ARMS"
grep -q SETUP_START $S || bash $B setup &       # data + packages need none of the shipped files
wait_files $CHECKPOINTS/E15/READY $L/bench2drive220.xml
mark CKPTS_READY

await DATA_DONE DATA_FAILED data
await CARLA_PROV_END CARLA_PROV_FAILED "trainer packages"
step AUDIT_DONE audit
step "CACHE_DONE D " cache D
step "CACHE_DONE DX" cache DX

# the new code on this box's GPU and packages before eight runs depend on it
box_tests tests/test_memorization_regularizers.py

NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
export NW=${NW:-3}
i=0
for A in $ARMS; do
  if [ -f $CHECKPOINTS/$A$SFX/TRAINED ]; then i=$((i + 1)); continue; fi
  ( CUDA_VISIBLE_DEVICES=$((i % NG)) bash $B train $A ) &
  i=$((i + 1)); sleep "${CHAIN_STAGGER:-30}"
done
wait
bash $B memtable
mark "MEM_DONE trained:$(for A in $ARMS; do [ -f $CHECKPOINTS/$A$SFX/TRAINED ] && printf ' %s' $A$SFX; done)"
