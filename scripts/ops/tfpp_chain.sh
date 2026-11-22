#!/usr/bin/env bash
# A86 + A87 round 2 training box (2026-11-21; TODO_ACTIVE.md A86 / A87). A training-only box (no CARLA), run after
# `scripts/ops/box_deploy.sh PORT HOST MANIFEST` has shipped E15 and the evaluation route file. Status: $WORKSPACE/ladder/status.
#   setup (data + trainer packages) -> audit -> cache_full: D and DX at 9x24 plus the exact 4x4 from the same encoder pass, sharded over
#   every GPU -> the new tests on this box -> trainers (each pinned to a GPU):
#     a86_tfpp_cam  TF++ camera-only on our sensor / resolution / frozen encoder / frames (TF++'s recipe, 31 epochs)
#     a87_fm        our network on the whole 9x24 map at 1512 channels (216 camera tokens)
#     a87_tp        our network + TF++'s target point and next target point (route_mode dense_tp)
#     a87_moelat    w4's experts restricted to sideways corrections (moe_lateral_only)
#   the three a87 arms use mem_L0's settings (E15 -> 30, fresh AdamW, val_every 1, train_eval_frames 20000): mem_L0 (11-21) is the control
# ARMS overrides the arm list (default: all four). The 11-22 resume of a86 alone (formerly a86_chain.sh):
#   ARMS=a86_tfpp_cam NW=12 TFPP_EPOCHS=15 TFPP_RESUME=$CHECKPOINTS/a86_tfpp_cam/snapshot_003.pth bash scripts/ops/tfpp_chain.sh
#   (resume the epoch-3 snapshot, stop at the end of the 8-epoch cosine cycle, epoch 15)
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
export NO_CARLA=1 PYC PYTHONPATH=$MTHESIS_ROOT
ARMS=${ARMS:-"a87_fm a86_tfpp_cam a87_tp a87_moelat"}
mark "TFPP_CHAIN_START"
grep -q SETUP_START $S || bash $B setup &
wait_files $CHECKPOINTS/E15/READY $L/bench2drive220.xml
mark CKPTS_READY
await DATA_DONE DATA_FAILED data
await CARLA_PROV_END CARLA_PROV_FAILED "trainer packages"
step AUDIT_DONE audit
step "CACHE_FULL_DONE DX" cache_full
box_tests tests/test_tfpp_cam.py tests/test_dense_tp_route.py tests/test_moe_lateral_and_logging.py tests/test_memorization_regularizers.py

NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
# the full-map arm (216 tokens, ~13x the attention) gets a GPU of its own when there are two
G_REST=0; case " $ARMS " in *" a87_fm "*) [ "$NG" -gt 1 ] && G_REST=1 ;; esac
for A in $ARMS; do
  case $A in a87_fm) G=0; W=6 ;; a86_tfpp_cam) G=$G_REST; W=6 ;; *) G=$G_REST; W=3 ;; esac
  [ -f $CHECKPOINTS/$A/TRAINED ] && continue
  if [ $A = a86_tfpp_cam ]; then STEP=train_tfpp; else STEP="train $A"; fi
  ( CUDA_VISIBLE_DEVICES=$G NW=${NW:-$W} bash $B $STEP ) &
  sleep "${CHAIN_STAGGER:-30}"   # stagger the trainers' start-up (CHAIN_STAGGER=0 in tests)
done
wait
mark "TFPP_DONE trained:$(trained_list '' $ARMS)"
