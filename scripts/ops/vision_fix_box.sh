#!/usr/bin/env bash
# Vision-fix arms (user, 2026-11-04: "start with the following order 3,2,4 in regards to obstacles as well ... make sure we dont train directly
# on the evaluation routes"). One 2-GPU box; every arm is arm J's recipe (fine-tune of E15 e15 on the 6 towns + the Town12/13 obstacle archives,
# route_original) with ONE change, trained on the same leak-filtered data, 15 epochs (16 -> 30 of the 50-epoch schedule):
#   ctrl   J's recipe itself on the filtered data                                   (GPU 1, pooled cache with the overlay)
#   s2d    fix 3, the 4x4 average pool: stride-16 space-to-depth 6x16 tokens + 20% lateral token masking (TODO A54)   (GPU 0, live encoder)
#   clean  fix 2, the painted route: no overlay, the route enters as tokens only (TODO A17c)                       (GPU 1, cache without overlay)
#   path   fix 4, the head cannot go around: space-indexed path head + path-conditioned speed (TODO A15); eval steers from it (WOR_PATH_STEER=1)
# usage: scripts/ops/vision_fix_box.sh STEP      status lines -> $WORKSPACE/vf/status
#   setup  deps, data (J's archives), TF++ weights, CARLA in the background; E15 comes from E: (deploy step)
#   audit  drop every Town12/13 route folder that sits on a Bench2Drive-220 evaluation route (scripts/analysis/route_overlap_audit.py)
#   cache  pooled 4x4 caches with and without the overlay, both GPUs
#   train  s2d on GPU 0; ctrl, path, clean one after another on GPU 1
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
mkdir -p $WORKSPACE/vf; S=$WORKSPACE/vf/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis; export PYTHONPATH=$WORKSPACE/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python; PYC=$WORKSPACE/venv_carla/bin/python
D=$WORKSPACE/dataset/wor_trajectories
TFPP=$WORKSPACE/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
E15=$WORKSPACE/checkpoints/E15
ARCH='^Town(0[1-5]|10)/|^Town1[23]/.*(Accident|Obstacle|HazardAtSideLane|InvadingTurn|OpensDoor)'
COMMON="--data_dir $D --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 30
 --resume_from $E15/model_epoch_015.pth --route_key route_original --weights_path $TFPP --freeze_backbone 1 --pretrained 1 --route_points 4
 --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4 --lr_backbone 1e-4
 --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --use_mlflow 1 --kill_stale 0 --seed 0 --experiment_name vision_fix"
NG=$(nvidia-smi -L | wc -l)
case $1 in
setup)
  mark SETUP_START
  ( $PY -m pip install -q timm mlflow tensorboard opencv-python-headless pytest requests huggingface_hub > $WORKSPACE/vf/pip.log 2>&1 && mark PIP_DONE ) &
  ( $PY -m pip install -q requests huggingface_hub >> $WORKSPACE/vf/pip.log 2>&1
    $PY scripts/setup/download_pdm_lite.py --towns Town01,Town02,Town03,Town04,Town05,Town10,Town12,Town13 --archive-regex "$ARCH" --reserve-gb 60 \
      --dest $D > $WORKSPACE/vf/data.log 2>&1 && mark DATA_DONE || mark DATA_FAILED ) &
  ( curl -sL -o $WORKSPACE/tfpp_pretrained.zip https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip && \
    mkdir -p $WORKSPACE/tfpp_pretrained && unzip -q -o $WORKSPACE/tfpp_pretrained.zip "pretrained_models/all_towns/model_0030_0.pth" -d $WORKSPACE/tfpp_pretrained && \
    mark TFPP_DONE || mark TFPP_FAILED ) &
  if [ "${NO_CARLA:-0}" = 1 ]; then  # 2026-11-04: training-only box (the s2d arm on a 5090): the trainer runs in /venv/main
    ( $PY -m pip install -q timm mlflow tensorboard opencv-python-headless pytest >> $WORKSPACE/vf/pip.log 2>&1; mark CARLA_PROV_END ) &
  else
    # 2026-11-13: the tools go through install_guard.sh (the CARLA provisioning above is the other writer of this venv, so the guard's
    # flock runs them one after the other); CARLA_PROV_END only after an import check passes, otherwise CARLA_PROV_FAILED.
    ( bash scripts/ops/provision_carla_no_hf.sh > $WORKSPACE/vf/carla_prov.log 2>&1
      bash scripts/ops/install_guard.sh carla_tools $WORKSPACE/venv_carla -- -q tensorboard mlflow pytest >> $WORKSPACE/vf/carla_prov.log 2>&1
      if $WORKSPACE/venv_carla/bin/python -c "import torch, carla, py_trees, tensorboard, mlflow, pytest" >> $WORKSPACE/vf/carla_prov.log 2>&1
      then mark CARLA_PROV_END; else mark CARLA_PROV_FAILED; fi ) &
  fi
  wait; mark SETUP_END ;;
audit)
  # the evaluation routes (bench2drive220.xml, shipped to $WORKSPACE/vf by the deploy step); leaks move to $WORKSPACE/dataset_excluded (kept, not deleted)
  $PY scripts/analysis/route_overlap_audit.py --eval_xml $WORKSPACE/vf/bench2drive220.xml --data_dir $D --out_json $WORKSPACE/vf/leak_traj.json \
    > $WORKSPACE/vf/audit.log 2>&1 || { mark AUDIT_FAILED; exit 1; }
  $PY - >> $WORKSPACE/vf/audit.log 2>&1 <<'PY' || { mark AUDIT_MOVE_FAILED; exit 1; }
import json, os, shutil
D, X = "/workspace/dataset/wor_trajectories", "/workspace/dataset_excluded"
rows = json.load(open("/workspace/vf/leak_traj.json"))["routes"]
moved = 0
for r in rows:
    if r["leak"]:
        dst = os.path.join(X, os.path.relpath(r["dir"], D)); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(r["dir"], dst); moved += 1
by = {}
for r in rows:
    k = r["type"]; n, m = by.get(k, (0, 0)); by[k] = (n + 1, m + int(r["leak"]))
print(f"moved {moved} of {len(rows)} Town12/13 route folders to {X}")
for k, (n, m) in sorted(by.items()):
    print(f"  {k}: {m} / {n} excluded")
PY
  mark "AUDIT_DONE $(grep -oE 'moved [0-9]+ of [0-9]+' $WORKSPACE/vf/audit.log)" ;;
cache)
  for OV in 1 0; do
    mkdir -p $WORKSPACE/pooled_ov$OV
    for g in $(seq 0 $((NG - 1))); do
      ( $PYC scripts/training/build_pooled_cache.py --data_dir $D --out_dir $WORKSPACE/pooled_ov$OV --backbone regnety_032 --weights_path $TFPP \
          --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay $OV --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 \
          --device cuda:$g --shard $g --num_shards $NG --num_workers $(( $(nproc) / NG - 2 )) --batch_size 128 \
          > $WORKSPACE/vf/cache_ov${OV}_$g.log 2>&1 && mark CACHE_OV${OV}_SHARD_$g || mark CACHE_OV${OV}_SHARD_${g}_FAILED ) &
      sleep 5
    done
    wait
  done
  ls $WORKSPACE/pooled_ov1/*.json $WORKSPACE/pooled_ov0/*.json > /dev/null && mark CACHE_DONE || mark CACHE_FAILED ;;
train|train_cached|train_s2d|seed1)
  # train = all four on one 2-GPU box; 2026-11-04 split: train_s2d on its own box (a 5090, ~3x the A6000's live-encoder rate: 617 vs
  # 172-191 frames/s), train_cached = the three cached arms over both GPUs of the A6000 box
  grep -q AUDIT_DONE $S || { echo "audit first"; exit 1; }
  [ -x $PYC ] || PYC=$PY
  run() {  # GPU LABEL extra-args (one run, foreground)
    CUDA_VISIBLE_DEVICES=$1 $PYC -u scripts/training/train_wor.py $COMMON $3 --run_label $2 --save_dir $WORKSPACE/checkpoints/$2 \
      > $WORKSPACE/vf/$2.log 2>&1; mark "TRAIN_END $2 rc=$?"
  }
  S2D="--route_overlay 1 --token_mode s2d --s2d_patch 3 --token_mask 0.2 --vision_grid 6x16 --encoder_channels_last 1 --num_workers $(( $(nproc) / NG - 4 ))"
  if [ "$1" != train_s2d ]; then
    P1=$(ls $WORKSPACE/pooled_ov1/*.json | head -1); P1=${P1%.json}; P0=$(ls $WORKSPACE/pooled_ov0/*.json | head -1); P0=${P0%.json}
    CTRL="--route_overlay 1 --vision_grid 4 --num_workers 8 --pooled_cache $P1"
    CLEAN="--route_overlay 0 --vision_grid 4 --num_workers 8 --pooled_cache $P0"
    PATHA="--route_overlay 1 --vision_grid 4 --num_workers 8 --pooled_cache $P1 --path_head 1 --path_cascade 1"
  fi
  case $1 in
  train)        ( run 0 vf_s2d "$S2D" ) & ( run 1 vf_ctrl "$CTRL"; run 1 vf_clean "$CLEAN"; run 1 vf_path "$PATHA" ) & ;;
  train_s2d)    ( run ${S2D_GPU:-0} vf_s2d "$S2D" ) & ;;
  # 2026-11-04 (approved: a second process per A6000 GPU). The queue had already moved on (fix 4 and the baseline's second seed started
  # when the first pair finished), so the second process per GPU is seed-1 replicates of the two arms that have one seed: the protocol
  # wants 3 trained seeds before a finalist (seed noise ~4 DS). Each ~40-60 min; ~2 processes per GPU keeps both busy.
  seed1)        ( run 0 vf_path_s1 "$PATHA --seed 1" ) & ( run 1 vf_clean_s1 "$CLEAN --seed 1" ) & ;;
  # three 1-hour jobs on two GPUs leave one GPU idle for an hour unless it gets a fourth job: the baseline's second seed (the noise
  # estimate the protocol needs: seed noise is ~4 DS). Same makespan as before (about 2 hours), no extra box time.
  train_cached) ( run 0 vf_ctrl "$CTRL"; run 0 vf_path "$PATHA" ) & ( run 1 vf_clean "$CLEAN"; run 1 vf_ctrl_s1 "$CTRL --seed 1" ) & ;;  # the user's order: fix 2 before fix 4
  esac
  sleep 240
  for L in vf_s2d vf_ctrl vf_clean; do [ -f $WORKSPACE/vf/$L.log ] && { echo "== $L"; grep -aE "Resumed|Validation split|pooled|token_mode|channels_last|Error|Traceback" $WORKSPACE/vf/$L.log | tail -5; }; done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  wait ;;
esac
