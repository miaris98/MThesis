#!/usr/bin/env bash
# Atari 10k screen queue (TODO B20 / B22, 2026-10-18). One worker runs its jobs in order; two workers can share one GPU.
#   job "TRUNK SEED SIMNORM POS"        train 10k (the 100k-schedule recipe stopped at 10k, uniform replay, 4 envs, 16 sims), then score
#                                       the final checkpoint. TRUNK = resnet|gtrxl, SIMNORM = 0|4|8|16 (B22), POS = abs|sym (B20).
#   job "eval LABEL CKPT [STAGNATION]"  score an existing checkpoint only; STAGNATION = reward-free steps before random actions (B44 step 1)
#   job "evals LABEL CKPT STICKY"       score an existing checkpoint with ALE sticky actions (0.25 = B2's protocol): about 30 distinct games instead of about 4
#   job "evalp LABEL CKPT LAMBDA [SIMS]" score an existing checkpoint with the B38 step 0 pessimistic search (two-network disagreement, weight LAMBDA, default 16 sims)
# Scoring = eval_loop_stats.py: 30 games, 16 sims, raw reward, cap 27,000 steps, mean + per-game loop statistics.
# usage: run_atari_10k_queue.sh GPU "JOB" ["JOB" ...]      (quote every job)
# Layout: /workspace/q/out/<label>/ training, /workspace/q/eval/*_loops.json scores, /workspace/q/logs/, /workspace/q/done/<label> marker.
# SCORE_STICKY=0.25 in the environment makes the scoring after a training job use ALE sticky actions (about 30 distinct games, 3-4 minutes) instead of the deterministic 4-game protocol.
# Labels: Q_<trunk>_10k_s<seed>_sn<simnorm>_pos<pos> (Q_gtrxl_10k_s0_sn0_posabs = the B22_gtrxl_10k_s0_sn0 runs of 2026-10-17).
cd /workspace/MThesis
GPU=$1; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true PYTHONPATH=/workspace/MThesis
Q=/workspace/q; mkdir -p $Q/out $Q/eval $Q/logs $Q/done

score() {  # LABEL CKPT [STAGNATION] [PESS_LAMBDA] [SIMS] [STICKY]     (B38 step 0: PESS_LAMBDA = the two-network pessimism weight, eager search)
  local label=$1 ck=$2 stag=${3:-0} pl=${4:-} sims=${5:-16} sticky=${6:-} tag="" extra=""
  [ -n "$pl" ] && { tag="_pess${pl}_sims${sims}"; extra="--pess-lambda $pl"; }
  [ -n "$sticky" ] && { tag="${tag}_sticky${sticky}"; extra="$extra --sticky $sticky"; }
  echo "$(date -u) score $label stag=$stag${tag}" >> $Q/logs/queue_$GPU.log
  CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/eval_loop_stats.py $ck --episodes 30 --sims $sims --max-steps 27000 \
    --stagnation $stag $extra --out $Q/eval > $Q/logs/eval_${label}_stag${stag}${tag}.log 2>&1
  echo "$(date -u) scored $label stag=$stag${tag} rc=$?" >> $Q/logs/queue_$GPU.log
}

train() {  # TRUNK SEED SIMNORM POS
  local trunk=$1 seed=$2 sn=$3 pos=$4 try rc ck res
  local label=Q_${trunk}_10k_s${seed}_sn${sn}_pos${pos}
  mkdir -p $Q/out/$label/checkpoints
  for try in 1 2 3; do
    ck=$Q/out/$label/checkpoints/checkpoint_latest.pt; res=""; [ -f $ck ] && res="--resume-from $ck"
    echo "$(date -u) start $label try $try ${res:-fresh}" >> $Q/logs/queue_$GPU.log
    CUDA_VISIBLE_DEVICES=$GPU /venv/main/bin/python -u atari_qwen/training/train_ez_offpolicy.py \
      --trunk $trunk --norm batch --priority-alpha 0 --seed $seed --simnorm $sn --mixer-pos $pos \
      --total-steps 10000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
      --graph-search --run-label $label --log-dir $Q/out $res >> $Q/logs/$label.log 2>&1
    rc=$?; echo "$(date -u) $label exited rc=$rc" >> $Q/logs/queue_$GPU.log
    [ $rc -eq 0 ] && break; sleep 30
  done
  ck=$Q/out/$label/checkpoints/checkpoint_env10000_upd10000.pt
  if [ -f $ck ]; then score $label $ck 0 "" 16 ${SCORE_STICKY:-}; echo "$(date -u) ok" > $Q/done/$label
  else echo "$(date -u) FAILED no final checkpoint" > $Q/done/$label.FAILED; fi
}

for job in "$@"; do
  set -- $job
  if [ "$1" = eval ]; then
    label=$2; ck=$3; stag=${4:-0}
    score $label $ck $stag; echo "$(date -u) ok" > $Q/done/${label}_stag${stag}
  elif [ "$1" = evals ]; then   # evals LABEL CKPT STICKY   (B2 robustness protocol: ALE sticky actions, 30 distinct games instead of about 4)
    label=$2; ck=$3; st=$4
    score $label $ck 0 "" 16 $st; echo "$(date -u) ok" > $Q/done/${label}_sticky${st}
  elif [ "$1" = evalp ]; then   # evalp LABEL CKPT LAMBDA [SIMS]
    label=$2; ck=$3; pl=$4; sims=${5:-16}
    score $label $ck 0 $pl $sims; echo "$(date -u) ok" > $Q/done/${label}_pess${pl}_sims${sims}
  else
    train "$1" "$2" "$3" "$4"
  fi
done
echo "$(date -u) QUEUE_DONE $GPU" >> $Q/logs/queue_$GPU.log
