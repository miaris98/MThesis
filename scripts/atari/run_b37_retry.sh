#!/usr/bin/env bash
# TODO B37 step 2b (retry of the failed lr 3e-4 screens): AdamW mixer at lr 3e-6 / 1e-5 / 3e-5 and one open-gate setting, 20k steps, seed 0, one setting per GPU,
# with the guards that the lr 3e-4 runs lacked: --nan-guard (skip a non-finite update) and --gn-stop 1e3 (stop after 3 log lines above it).
# usage: run_b37_retry.sh        (4 GPUs; read at 15k / 20k against B36's SGD runs: 22-48 at 15k, 52-98 at 20k)
cd /workspace/MThesis
G="--nan-guard --gn-stop 1e3 --audit-every 1000 --mixer-optimizer adamw"
export TOTAL_STEPS=20000 EVAL_INTERVAL=5000 MIXER_WD=0.05
EXTRA="$G --mixer-lr 3e-6" LABEL_SUFFIX=_b37r3e-6 bash scripts/atari/run_b36.sh 0 0 > /workspace/lane0.out 2>&1 &
EXTRA="$G --mixer-lr 1e-5" LABEL_SUFFIX=_b37r1e-5 bash scripts/atari/run_b36.sh 1 0 > /workspace/lane1.out 2>&1 &
EXTRA="$G --mixer-lr 3e-5" LABEL_SUFFIX=_b37r3e-5 bash scripts/atari/run_b36.sh 2 0 > /workspace/lane2.out 2>&1 &
EXTRA="$G --mixer-lr 1e-5 --mixer-bg-init 0" LABEL_SUFFIX=_b37r1e-5_open bash scripts/atari/run_b36.sh 3 0 > /workspace/lane3.out 2>&1 &
wait
