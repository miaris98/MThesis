#!/usr/bin/env bash
# TODO B36 (S-116): does the GTrXL token mixer train when it is exempt from weight decay? GTrXL 30k screen, same recipe as B35
# (uniform replay, --schedule-steps 100000, deferred 10-episode eval) except --mixer-weight-decay (default 0) and the CUDA-graphed
# search (--graph-search, bit-identical to eager, TODO B23 step 1b). One run after another per GPU lane; one deferred evaluator
# (--graph) and one diagnostics watcher per run. Checkpoints every 5000 env steps so the mixer can be inspected early.
#
# usage: run_b36.sh GPU SEED [SEED ...]            (e.g. 4 GPUs: run_b36.sh 0 0 & run_b36.sh 1 1 & run_b36.sh 2 2 & run_b36.sh 3 3 &)
# env:   MIXER_WD (0)  OUT_STD (0 = zero-init output projection; try 0.02 if the blocks stay at init, see B36's decision rule)
#        TOTAL_STEPS (30000)  EVAL_INTERVAL (5000)  LABEL_SUFFIX ("")  GRAPH (1; 0 = eager)
#        STATES (/workspace/b15_states_512.npy, from HF diag/b15_states_512.npy)
#        EXTRA (""; extra trainer flags, e.g. B37: EXTRA="--mixer-optimizer adamw --mixer-lr 3e-4 --audit-every 1000" MIXER_WD=0.05 LABEL_SUFFIX=_b37
# smoke test: TOTAL_STEPS=3000 EVAL_INTERVAL=1500 LABEL_SUFFIX=_smoke run_b36.sh 0 9
cd /workspace/MThesis
GPU=$1; shift
MIXER_WD=${MIXER_WD:-0}; OUT_STD=${OUT_STD:-0}; GRAPH=${GRAPH:-1}
TOTAL_STEPS=${TOTAL_STEPS:-30000}; EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
STATES=${STATES:-/workspace/b15_states_512.npy}
L=/workspace/MThesis/results/100k_benchmark/S058_ezv2_match; mkdir -p $L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true
GFLAG=""; EGFLAG=""; [ "$GRAPH" = 1 ] && GFLAG="--graph-search" && EGFLAG="--graph"
for seed in "$@"; do
  LABEL=S058k_gtrxl_30k_s${seed}_mixwd${MIXER_WD}$([ "$OUT_STD" != 0 ] && echo _out$OUT_STD)${LABEL_SUFFIX:-}
  mkdir -p $L/$LABEL/checkpoints
  echo "$(date -u) start $LABEL on GPU $GPU (mixer wd $MIXER_WD, out std $OUT_STD, graph $GRAPH)"
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/eval_ez_checkpoints.py $L/$LABEL --watch $EGFLAG \
    >> /workspace/${LABEL}_eval.log 2>&1 &
  EVPID=$!
  bash scripts/atari/b36_diag_watch.sh $L/$LABEL $STATES >> /workspace/${LABEL}_diag.log 2>&1 &
  DGPID=$!
  for try in 1 2 3; do
    CK=$L/$LABEL/checkpoints/checkpoint_latest.pt; RES=""; [ -f $CK ] && RES="--resume-from $CK"
    CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
      --trunk gtrxl --norm batch --priority-alpha 0 --seed $seed --mixer-weight-decay $MIXER_WD --mixer-out-init-std $OUT_STD \
      --total-steps $TOTAL_STEPS --schedule-steps 100000 --eval-interval $EVAL_INTERVAL --eval-episodes 10 --eval-mode deferred \
      $GFLAG ${EXTRA:-} --run-label $LABEL --log-dir $L $RES >> /workspace/$LABEL.log 2>&1
    rc=$?; echo "$(date -u) $LABEL exited rc=$rc"; [ $rc -eq 0 ] && break; sleep 30
  done
  sleep 120; kill $DGPID 2>/dev/null   # the evaluator exits by itself once the final checkpoint is scored
done
wait
echo "$(date -u) B36 lane GPU $GPU done: $*"
