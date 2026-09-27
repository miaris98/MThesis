#!/usr/bin/env bash
# Official EfficientZero V2, Breakout, seeds 0-4. Runs on box W (RTX PRO 4000 Blackwell) with /workspace/ezv2_setup.sh =
# ezv2_setup_blackwell.sh; box U's RTX 4090 with the pinned py3.8 stack crashed in cuDNN (S-086).
# Each seed is stopped once its 10k checkpoint (model_10000.p) exists (the trainer's "Aborted" line in the log is
# that kill, not a crash), then that checkpoint evaluated
# with 30 episodes (eval_ezv2_checkpoint.py, EZ-V2's own eval). Gives EZ-V2 a 10k distribution to compare with our
# port's 6+ seeds (TODO_ACTIVE B1/B2). Seed 0's 10k checkpoint (9.8) was not kept on E:, so seed 0 is rerun too.
set -uo pipefail
cd /workspace/EfficientZeroV2
grep -q EZV2_SETUP_DONE /workspace/ezv2_setup.log 2>/dev/null || bash /workspace/ezv2_setup.sh > /workspace/ezv2_setup.log 2>&1 || { echo "EZV2 SETUP FAILED"; exit 1; }
grep -q EZV2_SETUP_DONE /workspace/ezv2_setup.log || { echo "EZV2 SETUP FAILED"; exit 1; }
export WANDB_MODE=offline RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 EZV2_OBJECT_STORE_GB=20 \
  EZV2_NUM_CPUS=${EZV2_NUM_CPUS:-12} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  # container pids.max 2816: 48 prestarted workers x threads exhausted it
O=/workspace/ezv2_10k_eval; mkdir -p $O
for s in 0 1 2 3 4; do
  sed "s/^  base_seed: 0/  base_seed: $s/" ez/config/exp/atari_breakout.yaml > ez/config/exp/atari_breakout_s$s.yaml
  grep -q "base_seed: $s" ez/config/exp/atari_breakout_s$s.yaml || { echo "seed edit failed for $s"; exit 1; }
  echo "$(date) seed $s: training until model_10000.p"
  /workspace/venv_ezv2/bin/python ez/train.py exp_config=ez/config/exp/atari_breakout_s$s.yaml > /workspace/ezv2_train_s$s.log 2>&1 &
  TP=$!
  until C=$(ls results/Atari/Breakout/EZ-V2-seed=$s-*/models/model_10000.p 2>/dev/null | head -1) && [ -n "$C" ]; do
    kill -0 $TP 2>/dev/null || { echo "$(date) seed $s: trainer exited before model_10000.p"; break; }
    sleep 60
  done
  sleep 30  # let the checkpoint write finish
  pkill -P $TP 2>/dev/null; kill $TP 2>/dev/null; sleep 10; ray stop --force > /dev/null 2>&1
  C=$(ls results/Atari/Breakout/EZ-V2-seed=$s-*/models/model_10000.p 2>/dev/null | head -1)
  [ -n "$C" ] || continue
  mkdir -p $O/seed$s && cp "$C" $O/seed$s/model_10000.p
  echo "$(date) seed $s: evaluating $C (30 episodes)"
  /workspace/venv_ezv2/bin/python /workspace/MThesis/scripts/eval/eval_ezv2_checkpoint.py $O/seed$s/model_10000.p $O/seed${s}_10k.json 30 > $O/seed${s}_eval.log 2>&1
  ray stop --force > /dev/null 2>&1
  tail -2 $O/seed${s}_eval.log
done
echo "$(date) EZ-V2 10k seeds done"
