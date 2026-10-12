#!/usr/bin/env bash
# S-117 (2026-10-04) Atari box: a bounded queue of deferred evaluations in the GPUs' spare headroom (TODO B24 rest + B2 sticky column), all with the
# graph-captured search. usage: run_eval_queue.sh NPAR   (reads jobs from /workspace/eval_jobs.txt: "RUNDIR extra eval_ez_checkpoints.py args...")
# Jobs are spread round-robin over the GPUs; NPAR run at a time (each evaluator = one CPU core + a few hundred MB of GPU).
cd /workspace/MThesis
NPAR=${1:-6}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MLFLOW_ALLOW_FILE_STORE=true
mkdir -p /workspace/logs_eval
NG=$(nvidia-smi -L | wc -l)
i=0
while read -r d args; do
  [ -z "$d" ] && continue
  echo "$((i % NG)) $d $args"; i=$((i + 1))
done < /workspace/eval_jobs.txt | xargs -P "$NPAR" -L1 bash -c '
  gpu=$0; d=$1; shift; r=$(basename $d); tag=$(echo "$@" | tr -c "A-Za-z0-9" _)
  echo "$(date -u) start $r $@ on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu /venv/main/bin/python atari_qwen/training/eval_ez_checkpoints.py $d "$@" > /workspace/logs_eval/${r}${tag}.log 2>&1
  echo "$(date -u) done $r $@ rc=$?"'
echo "$(date -u) eval queue finished"
