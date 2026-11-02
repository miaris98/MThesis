#!/usr/bin/env bash
# A74 RL box driver (2026-11-02): closed-loop PPO of the maneuver head on top of E15. Runs on an eval-type box (CARLA provisioned,
# E15 in /workspace/checkpoints/E15). usage: scripts/ops/a74_rl_box.sh STEP      status lines -> /workspace/a74/status
#   routes   training batches from carla_garage's PDM-Lite routes (build_a74_routes.py; never the 220 evaluation routes)
#   learner  the PPO learner (train_maneuver_ppo.py; resumable: rerun it after a stop)
#   lanes    A74_LANES RL lanes (ports 2000, 2400, 2800, ...; lane k on GPU k % A74_GPUS); each runs batch after batch through
#            run_bench2drive.sh with WOR_RL set, one fresh evaluator per batch (resets the leaderboard's memory growth), until STOP or the
#            learner is done. Lane k starts at batch k + A74_BOFF and steps by A74_LANES: boxes feeding one learner get disjoint offsets
#   val      the current head, greedy, on the held-out val.xml (one lane, the port after the last lane): the learning curve on routes never
#            trained on
#   stop     lanes finish their current batch and exit; the learner keeps its state
# env: A74_RUN (run1), A74_LANES (3), A74_BATCHES (400), A74_PER_BATCH (25), A74_SEED (0), A74_BATCH (4096 decisions per update),
#      A74_MAX (300000 decisions), A74_EPS (0.01), A74_EPS_STUCK (0.1), A74_GPUS (1), A74_BOFF (0), A74_STAGGER_S (15),
#      A74_THREADS (4: OpenMP / MKL threads per evaluator), A74_ROUTES (/workspace/a74/routes: the batch folder, e.g. a probe's own),
#      A74_OBST (0.5: obstacle-scenario share of a batch, routes step), A74_COMMIT (40 ticks a non-default maneuver is held)
# The learner can also run elsewhere (scripts/ops/a74_relay.sh: learner on the PC, episodes and heads relayed); then skip the learner step.
set -uo pipefail
A=/workspace/a74; RUN=${A74_RUN:-run1}; R=$A/$RUN; RT=${A74_ROUTES:-$A/routes}
mkdir -p $R/logs; S=$A/status
mark() { echo "$(date -u +%H:%M:%S) $RUN $1" >> $S; }
cd /workspace/MThesis; export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true
. scripts/lib/common.sh
PY=/venv/main/bin/python
E15=/workspace/checkpoints/E15/model_epoch_015.pth
case $1 in
routes)
  $PY scripts/training/build_a74_routes.py --src /workspace/carla_garage/data --out $RT --batches ${A74_BATCHES:-400} \
    --per_batch ${A74_PER_BATCH:-25} --seed ${A74_SEED:-0} --obstacle_share ${A74_OBST:-0.5} > $RT.log 2>&1
  mark "ROUTES_EXIT $? $(tail -1 $RT.log)" ;;
learner)
  pgrep -f "train_maneuver_pp[o].py --rollouts $R/" > /dev/null && { mark LEARNER_ALREADY_RUNNING; exit 1; }
  mark LEARNER_START
  $PY -u scripts/training/train_maneuver_ppo.py --rollouts $R/rollouts --out $R/learner --batch_decisions ${A74_BATCH:-4096} \
    --max_decisions ${A74_MAX:-300000} --seed ${A74_SEED:-0} >> $R/learner.log 2>&1
  mark "LEARNER_EXIT $?" ;;
lanes)
  [ -f $RT/manifest.json ] || { mark "LANES_NOT_STARTED no routes (run the routes step)"; exit 1; }
  rm -f $R/STOP
  for k in $(seq 0 $(( ${A74_LANES:-3} - 1 ))); do
    setsid nohup bash scripts/ops/a74_rl_box.sh lane $k > $R/lane_$k.out 2>&1 < /dev/null &
    sleep ${A74_STAGGER_S:-15}     # 12 CARLA servers started at once: one died in start-up ("GameThread timed out waiting for RenderThread")
  done
  mark "LANES_START ${A74_LANES:-3}" ;;
lane)
  k=$2; L=${A74_LANES:-3}; g=$(( k % ${A74_GPUS:-1} )); port=$(( 2000 + 400 * k )); tm=$(( 8000 + 400 * k ))
  # the evaluator's OpenMP / MKL pools default to every hardware thread: 12 lanes on the 128-thread 4x3090 ran ~330 threads and 6-11 cores
  # each (load 89), starving the CARLA servers (2026-11-02)
  export OMP_NUM_THREADS=${A74_THREADS:-4} MKL_NUM_THREADS=${A74_THREADS:-4} OPENBLAS_NUM_THREADS=${A74_THREADS:-4}
  nb=$(ls $RT/batch_*.xml | wc -l)
  export WOR_RL="head=$R/learner/head_latest.pth,mode=train,log=$R/rollouts,period=5,commit=${A74_COMMIT:-40},eps=${A74_EPS:-0.01},eps_stuck=${A74_EPS_STUCK:-0.1},stuck_s=4"
  b=$(( k + ${A74_BOFF:-0} ))
  while [ ! -f $R/STOP ] && ! grep -q A74_LEARNER_DONE $R/learner.log 2>/dev/null; do
    f=$RT/batch_$(printf %04d $(( b % nb ))).xml
    label=a74_${RUN}_l${k}_b$(printf %04d $b)
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g bash scripts/eval/run_bench2drive.sh $label qwen30m $E15 $f $port $tm $g > $R/logs/$label.log 2>&1
    mark "LANE $k batch $b exit $? $(( $(date +%s) - t0 )) s, episodes $(ls $R/rollouts/*.npz 2>/dev/null | wc -l) waiting"
    clear_carla_ports $port
    b=$(( b + L ))
  done
  mark "LANE $k stopped" ;;
val)
  vp=$(( 2000 + 400 * ${A74_LANES:-3} )); vt=$(( 8000 + 400 * ${A74_LANES:-3} ))
  H=${2:-$R/learner/head_latest.pth}; it=$($PY -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False)['meta'].get('iter', 0))" $H)
  label=a74_${RUN}_val_iter$(printf %04d $it)
  cp $H $R/learner/val_head_iter$(printf %04d $it).pth
  WOR_RL="head=$R/learner/val_head_iter$(printf %04d $it).pth,mode=eval,log=$R/val_rollouts" CUDA_VISIBLE_DEVICES=0 \
    bash scripts/eval/run_bench2drive.sh $label qwen30m $E15 $RT/val.xml $vp $vt 0 > $R/logs/$label.log 2>&1
  mark "VAL iter $it exit $?"
  clear_carla_ports $vp ;;
stop)
  touch $R/STOP; mark STOP ;;
esac
