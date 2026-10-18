#!/usr/bin/env bash
# Keep an Atari GPU busy (2026-10-18): whenever fewer than TARGET queue workers run, start the next job from a pool file.
# Pool = one job per line, same syntax as run_atari_10k_queue.sh ("TRUNK SEED SIMNORM POS" or "eval LABEL CKPT [STAG]").
# Stops starting jobs once the UTC time passes CUTOFF, so a job can still finish before the final save.
# usage: run_atari_refill.sh GPU TARGET CUTOFF_HHMM_UTC POOLFILE        (run it detached; log: /workspace/q/logs/refill_<gpu>.log)
cd /workspace/MThesis
GPU=$1; TARGET=$2; CUTOFF=$3; POOL=$4
LOG=/workspace/q/logs/refill_$GPU.log; mkdir -p /workspace/q/logs
echo "$(date -u) refill started gpu=$GPU target=$TARGET cutoff=$CUTOFF pool=$POOL" >> $LOG
while true; do
  if [ "$(date -u +%H%M)" -ge "$CUTOFF" ]; then echo "$(date -u) cutoff reached, refill stops" >> $LOG; exit 0; fi
  n=$(ps -eo args | grep -c "[r]un_atari_10k_queue.sh $GPU ")
  if [ "$n" -lt "$TARGET" ] && [ -s "$POOL" ]; then
    job=$(flock "$POOL.lock" bash -c 'j=$(head -n 1 "$1"); sed -i 1d "$1"; echo "$j"' _ "$POOL")
    if [ -n "$job" ]; then
      echo "$(date -u) workers=$n -> starting: $job" >> $LOG
      setsid nohup bash scripts/atari/run_atari_10k_queue.sh $GPU "$job" > /workspace/q/refill_worker_$(date +%s).out 2>&1 < /dev/null &
      sleep 120; continue
    fi
  fi
  sleep 60
done
