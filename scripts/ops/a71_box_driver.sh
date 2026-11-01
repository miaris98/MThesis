#!/usr/bin/env bash
# A71-Q1 box driver (2026-11-01). usage: scripts/ops/a71_box_driver.sh STEP     status lines -> /workspace/a71/status
#   setup   deps + A71 tests, the 8 E15 towns (152 GB), CARLA in the background
#   labels  qcf_v4: every 4th frame, every 2nd in obstacle scenarios (all cores), then hold / overspeed ablations on two Town12 scenarios
#   eval    lanes: E15 | a71q1 off | a71q1 on (Q decode), 104 lead + obstacle routes at Low, the 5 crash routes at Epic
#   train   E15 epoch 15 -> 20 with the Q head on qcf_v4, E15's recipe
#   check   offline Q-head check of the trained epochs
set -uo pipefail
mkdir -p /workspace/a71; S=/workspace/a71/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis; export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python
D=/workspace/dataset/wor_trajectories
DENSE="Accident|Obstacle|HazardAtSideLane|OpensDoor|Construction|InvadingTurn|ParkedObstacle"
NC=$(cat /workspace/MThesis/scripts/ops/a71_routes_lead_obstacle_104.txt)
EPIC=27515,25854,25896,25955,24816
E15=/workspace/checkpoints/E15
case $1 in
setup)
  mark SETUP_START
  ( $PY -m pip install -q timm mlflow tensorboard opencv-python-headless pytest gymnasium requests huggingface_hub > /workspace/a71/pip.log 2>&1 && mark PIP_DONE \
    && timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/test_counterfactual_q.py tests/test_q_critic_policy.py tests/test_build_q_labels.py \
         tests/test_qhead_offline_check.py tests/test_carla_encoder_frozen_backbone.py tests/test_world_on_rails.py > /workspace/a71/pytest_a71.log 2>&1; mark "PYTEST_EXIT $?" ) &
  ( $PY -m pip install -q requests huggingface_hub >> /workspace/a71/pip.log 2>&1
    $PY scripts/setup/download_pdm_lite.py --towns Town01,Town02,Town03,Town04,Town05,Town10,Town12,Town13 --include-huge --reserve-gb 60 --dest $D \
      > /workspace/a71/data.log 2>&1 && mark DATA_DONE || mark DATA_FAILED ) &
  ( bash scripts/ops/provision_carla_no_hf.sh > /workspace/a71/carla_prov.log 2>&1; mark CARLA_PROV_END ) &
  wait; mark SETUP_END ;;
labels)
  mark LABELS_START
  $PY scripts/training/build_q_labels.py --data_dir $D --tag qcf_v4 --stride 4 --dense_regex "$DENSE" --dense_stride 2 --workers ${LW:-16}     --summary /workspace/a71/qlabels_v4.json > /workspace/a71/qlabels_v4.log 2>&1
  mark "LABELS_EXIT $?"
  # ablations for the expert_top3_share drop (0.52 qcf_v2 -> 0.35 qcf_v3), on the two obstacle scenarios of Town12 only
  mkdir -p /workspace/dataset/abl/Town12/data
  for sc in ParkedObstacleTwoWays HardBreakRoute; do ln -sfn $D/Town12/data/$sc /workspace/dataset/abl/Town12/data/$sc; done
  for v in "hold8:--set hold=8" "osm0:--set overspeed_margin=0" "v3:"; do
    tag=abl_${v%%:*}; $PY scripts/training/build_q_labels.py --data_dir /workspace/dataset/abl --tag $tag ${v#*:} --stride 2 --summary /workspace/a71/$tag.json > /workspace/a71/$tag.log 2>&1
    mark "ABL_${tag}_EXIT $?"
  done ;;
train)
  pgrep -af "train_wor.py.*a71_q1" && { mark "TRAIN_ALREADY_RUNNING"; exit 1; }
  mark TRAIN_START
  WOR_Q_MIN_COVERAGE=0.2 $PY -u scripts/training/train_wor.py --data_dir $D --save_dir /workspace/checkpoints/A71_q1 --run_label a71_q1 \
    --policy_arch qwen30m --backbone regnety_032 --weights_path $E15/frozen_backbone.pth --pretrained 1 --freeze_backbone 1 \
    --img_size 288x768 --crop_bottom_frac 0.0 --vision_grid 4 --route_overlay 1 --route_points 4 \
    --use_augmented_camera 1 --target_speed_loss_weight 0.2 --target_speed_input policy --lateral_loss_weight 3.0 --wp_loss_weight 1.0 \
    --batch_size 256 --lr_heads 3e-4 --lr_backbone 1e-4 --grad_clip 5.0 --warmup_frac 0.05 --val_split 0.15 --seed 0 \
    --epochs 50 --stop_epoch 20 --save_freq 1 --val_every 1 --resume_from $E15/model_epoch_015.pth \
    --q_critic 1 --q_labels qcf_v4 --q_critic_loss_weight 1.0 --num_workers ${NW:-14} --use_mlflow 1 --kill_stale 0 \
    --experiment_name A71_q1 > /workspace/a71/train_q1.log 2>&1
  mark "TRAIN_EXIT $?" ;;
check)
  for e in ${EPOCHS:-16 18 20}; do
    f=/workspace/checkpoints/A71_q1/model_epoch_$(printf %03d $e).pth; [ -f $f ] || continue
    $PY scripts/analysis/qhead_offline_check.py --ckpt $f --q_labels qcf_v4 --out /workspace/a71/qhead_e$e.json --num_workers 12 > /workspace/a71/qhead_e$e.log 2>&1
    mark "QHEAD_e${e}_EXIT $?"
  done ;;
eval_e15)  # one lane while the labels build (CPU), started as soon as CARLA is provisioned
  setsid nohup bash scripts/eval/queue_A9c.sh 2000 0 "CKL:a71_e15:E15/model_epoch_015.pth:$NC" "CKE:a71_e15_epic:E15/model_epoch_015.pth:$EPIC:CARLA_QUALITY=Epic"     >> /workspace/a71/lane_2000.out 2>&1 < /dev/null & ;;
eval_q)    # the two Q-critic arms, same checkpoint, after training; QARGS e.g. "beta=1;lat_margin=1"
  CK=A71_q1/model_epoch_${QEPOCH:-020}.pth
  setsid nohup bash scripts/eval/queue_A9c.sh 2400 0 "CKL:a71q1_off:$CK:$NC" "CKE:a71q1_off_epic:$CK:$EPIC:CARLA_QUALITY=Epic"     >> /workspace/a71/lane_2400.out 2>&1 < /dev/null &
  setsid nohup bash scripts/eval/queue_A9c.sh 2800 0 "CKE:a71q1_on:$CK:$NC:WOR_QCRITIC=${QARGS:-beta=1},WOR_QCRITIC_LOG=/workspace/a71/qdecode_on.jsonl"     "CKE:a71q1_on_epic:$CK:$EPIC:WOR_QCRITIC=${QARGS:-beta=1},CARLA_QUALITY=Epic" >> /workspace/a71/lane_2800.out 2>&1 < /dev/null & ;;
esac
