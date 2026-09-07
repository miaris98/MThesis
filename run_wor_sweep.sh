#!/usr/bin/env bash
# =============================================================================
# run_wor_sweep.sh - Phase 1 of the WoR decision-head architecture study.
#
# Seven runs, in the order that makes the later ones interpretable:
#
#   1. NOISE FLOOR   qwen30m at three seeds. Nothing else in this sweep means
#                    anything until the spread between identical runs is known.
#                    A component that moves the metric by less than this spread
#                    has not been shown to do anything.
#   2. SIZE CURVE    qwen10m / qwen100m at one seed. The dataset is ~9,600
#                    frames and the 100M trunk underfits it, so the size
#                    question comes before any architectural one.
#   3. ATTRIBUTION   cnn, and cnn with --pool_vision 1. If the conv head's
#                    advantage survives being made spatially blind, it is an
#                    architecture result; if it collapses, the original
#                    comparison was measuring access to spatial features.
#
# Runs are sequential (one GPU) and resumable - a completed run leaves a
# .sweep_done marker and is skipped on re-invocation, so a dropped SSH session
# costs one run, not the sweep.
#
# Usage:
#   ./run_wor_sweep.sh                       # defaults below
#   EPOCHS=10 ./run_wor_sweep.sh             # quick shakedown
#   VISION_GRID=8 ./run_wor_sweep.sh         # full 64-token grid (~9x slower)
#   ONLY=noise ./run_wor_sweep.sh            # just the noise floor
#
# Run it under tmux on vast.ai; it prints the comparison table at the end.
# =============================================================================
# No -e: one failed run must not abandon the six that would still be useful.
set -uo pipefail

DATA_DIR="${DATA_DIR:-/workspace/dataset/wor_trajectories}"
OUT_ROOT="${OUT_ROOT:-/workspace/checkpoints/wor_sweep}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKBONE="${BACKBONE:-resnet34}"
# 4 (20-token sequence) is the development setting: ~2.9x the pooled step time
# against ~9x for grid 8, which is what makes a 7-run sweep affordable. Confirm
# the winners at VISION_GRID=8.
VISION_GRID="${VISION_GRID:-4}"
VAL_SPLIT="${VAL_SPLIT:-0.15}"
NOISE_SEEDS="${NOISE_SEEDS:-0 1 2}"
BASE_SEED="${BASE_SEED:-0}"
USE_MLFLOW="${USE_MLFLOW:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ONLY="${ONLY:-all}"          # all | noise | size | attribution
EXTRA_ARGS="${EXTRA_ARGS:-}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

mkdir -p "$OUT_ROOT"
SWEEP_LOG="$OUT_ROOT/sweep.log"

log() { echo -e "$@" | tee -a "$SWEEP_LOG"; }

if [ ! -d "$DATA_DIR" ]; then
    log "${RED}[FATAL] Dataset not found at $DATA_DIR${NC}"
    log "Run setup_vastai.sh first, or set DATA_DIR. Training would otherwise fall"
    log "back to synthetic samples and produce a sweep of meaningless numbers."
    exit 1
fi

frame_count=$(find "$DATA_DIR" -name "*.json.gz" 2>/dev/null | wc -l)
log "${BOLD}=== WoR architecture sweep - Phase 1 ===${NC}"
log "  data      : $DATA_DIR (~$frame_count measurement files)"
log "  output    : $OUT_ROOT"
log "  epochs    : $EPOCHS | batch $BATCH_SIZE | backbone $BACKBONE"
log "  vision    : grid $VISION_GRID | val_split $VAL_SPLIT"
log "  seeds     : noise floor [$NOISE_SEEDS], others $BASE_SEED"
log "  started   : $(date -Is)"
log ""

# ---------------------------------------------------------------------------
# run_one <label> <seed> [extra flags...]
# ---------------------------------------------------------------------------
run_one() {
    local label="$1"; shift
    local seed="$1"; shift
    local dir="$OUT_ROOT/${label}_s${seed}"

    if [ -f "$dir/.sweep_done" ]; then
        log "${YELLOW}--> SKIP ${label} seed=${seed} (already complete)${NC}"
        return 0
    fi

    log "${GREEN}--> RUN  ${label} seed=${seed}  ->  $dir${NC}"
    mkdir -p "$dir"
    local started
    started=$(date +%s)

    "$PYTHON_BIN" train_wor.py \
        --data_dir "$DATA_DIR" \
        --save_dir "$dir" \
        --run_label "$label" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --backbone "$BACKBONE" \
        --vision_grid "$VISION_GRID" \
        --val_split "$VAL_SPLIT" \
        --use_mlflow "$USE_MLFLOW" \
        --experiment_name "WoR_Architecture_Sweep" \
        $EXTRA_ARGS \
        "$@" 2>&1 | tee -a "$dir/train.log" | grep -E "^\[Epoch|^-->|^\[Warning|^\[FATAL|Error"

    local status=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - started ))

    if [ "$status" -eq 0 ]; then
        touch "$dir/.sweep_done"
        log "${GREEN}    done in $((elapsed / 60))m $((elapsed % 60))s${NC}"
    else
        log "${RED}    FAILED (exit $status) after $((elapsed / 60))m - see $dir/train.log${NC}"
    fi
    log ""
    return 0
}

# --- 1. Noise floor -------------------------------------------------------
if [ "$ONLY" = "all" ] || [ "$ONLY" = "noise" ]; then
    log "${BOLD}[1/3] Noise floor: qwen30m at seeds $NOISE_SEEDS${NC}"
    for seed in $NOISE_SEEDS; do
        run_one "qwen30m" "$seed" --policy_arch qwen30m
    done
fi

# --- 2. Size curve --------------------------------------------------------
if [ "$ONLY" = "all" ] || [ "$ONLY" = "size" ]; then
    log "${BOLD}[2/3] Size curve: qwen10m, qwen100m${NC}"
    run_one "qwen10m"  "$BASE_SEED" --policy_arch qwen10m
    run_one "qwen100m" "$BASE_SEED" --policy_arch qwen100m
fi

# --- 3. Attribution -------------------------------------------------------
if [ "$ONLY" = "all" ] || [ "$ONLY" = "attribution" ]; then
    log "${BOLD}[3/3] Attribution: cnn with and without spatial vision${NC}"
    run_one "cnn"        "$BASE_SEED" --policy_arch cnn --pool_vision 0
    run_one "cnn_pooled" "$BASE_SEED" --policy_arch cnn --pool_vision 1
fi

# --- Summary --------------------------------------------------------------
log "${BOLD}=== Sweep finished $(date -Is) ===${NC}"
log ""
"$PYTHON_BIN" compare_wor_runs.py --glob "$OUT_ROOT/*" \
    --baseline qwen30m --csv "$OUT_ROOT/summary.csv" 2>&1 | tee -a "$SWEEP_LOG"

log ""
log "Full log: $SWEEP_LOG"
log "Re-run this script to retry any run that failed; completed runs are skipped."
