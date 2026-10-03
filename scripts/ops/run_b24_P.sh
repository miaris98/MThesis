# S-115 box P: TODO B24 (30 episodes, no sticky, 64 simulations) on the 16 B35 checkpoints at env 30000 (upd 28004),
# one evaluator per checkpoint, spread over P's 4 GPUs (B2 at 16 simulations runs on box Q).
cd /workspace/MThesis_atari
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MLFLOW_ALLOW_FILE_STORE=true
mkdir -p /workspace/logs_eval
i=0
for d in results/100k_benchmark/S058_ezv2_match/S058*; do
  r=$(basename $d)
  CUDA_VISIBLE_DEVICES=$((i % 4)) setsid nohup /venv/main/bin/python atari_qwen/training/eval_ez_checkpoints.py $d \
    --episodes 30 --sims 64 --env-steps 30000 > /workspace/logs_eval/${r}_sim64.log 2>&1 < /dev/null &
  i=$((i + 1))
done
sleep 5
echo "evaluators $(pgrep -fc '[e]val_ez_checkpoints.py')"
