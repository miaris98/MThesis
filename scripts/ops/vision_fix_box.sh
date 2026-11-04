#!/usr/bin/env bash
# Vision-fix arms (user, 2026-11-04: "start with the following order 3,2,4 in regards to obstacles as well ... make sure we dont train directly
# on the evaluation routes"). One 2-GPU box; every arm is arm J's recipe (fine-tune of E15 e15 on the 6 towns + the Town12/13 obstacle archives,
# route_original) with ONE change, trained on the same leak-filtered data, 15 epochs (16 -> 30 of the 50-epoch schedule):
#   ctrl   J's recipe itself on the filtered data                                   (GPU 1, pooled cache with the overlay)
#   s2d    fix 3, the 4x4 average pool: stride-16 space-to-depth 6x16 tokens + 20% lateral token masking (TODO A54)   (GPU 0, live encoder)
#   clean  fix 2, the painted route: no overlay, the route enters as tokens only (TODO A17c)                       (GPU 1, cache without overlay)
#   path   fix 4, the head cannot go around: space-indexed path head + path-conditioned speed (TODO A15); eval steers from it (WOR_PATH_STEER=1)
# usage: scripts/ops/vision_fix_box.sh STEP      status lines -> /workspace/vf/status
#   setup  deps, data (J's archives), TF++ weights, CARLA in the background; E15 comes from E: (deploy step)
#   audit  drop every Town12/13 route folder that sits on a Bench2Drive-220 evaluation route (scripts/analysis/route_overlap_audit.py)
#   cache  pooled 4x4 caches with and without the overlay, both GPUs
#   train  s2d on GPU 0; ctrl, path, clean one after another on GPU 1
set -uo pipefail
mkdir -p /workspace/vf; S=/workspace/vf/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis; export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python; PYC=/workspace/venv_carla/bin/python
D=/workspace/dataset/wor_trajectories
TFPP=/workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
E15=/workspace/checkpoints/E15
ARCH='^Town(0[1-5]|10)/|^Town1[23]/.*(Accident|Obstacle|HazardAtSideLane|InvadingTurn|OpensDoor)'
COMMON="--data_dir $D --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 30
 --resume_from $E15/model_epoch_015.pth --route_key route_original --weights_path $TFPP --freeze_backbone 1 --pretrained 1 --route_points 4
 --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4 --lr_backbone 1e-4
 --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --use_mlflow 1 --kill_stale 0 --seed 0 --experiment_name vision_fix"
NG=$(nvidia-smi -L | wc -l)
case $1 in
setup)
  mark SETUP_START
  ( $PY -m pip install -q timm mlflow tensorboard opencv-python-headless pytest requests huggingface_hub > /workspace/vf/pip.log 2>&1 && mark PIP_DONE ) &
  ( $PY -m pip install -q requests huggingface_hub >> /workspace/vf/pip.log 2>&1
    $PY scripts/setup/download_pdm_lite.py --towns Town01,Town02,Town03,Town04,Town05,Town10,Town12,Town13 --archive-regex "$ARCH" --reserve-gb 60 \
      --dest $D > /workspace/vf/data.log 2>&1 && mark DATA_DONE || mark DATA_FAILED ) &
  ( curl -sL -o /workspace/tfpp_pretrained.zip https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip && \
    mkdir -p /workspace/tfpp_pretrained && unzip -q -o /workspace/tfpp_pretrained.zip "pretrained_models/all_towns/model_0030_0.pth" -d /workspace/tfpp_pretrained && \
    mark TFPP_DONE || mark TFPP_FAILED ) &
  ( bash scripts/ops/provision_carla_no_hf.sh > /workspace/vf/carla_prov.log 2>&1
    VIRTUAL_ENV=/workspace/venv_carla uv pip install -q tensorboard mlflow pytest >> /workspace/vf/carla_prov.log 2>&1; mark CARLA_PROV_END ) &
  wait; mark SETUP_END ;;
audit)
  # the evaluation routes (bench2drive220.xml, shipped to /workspace/vf by the deploy step); leaks move to /workspace/dataset_excluded (kept, not deleted)
  $PY scripts/analysis/route_overlap_audit.py --eval_xml /workspace/vf/bench2drive220.xml --data_dir $D --out_json /workspace/vf/leak_traj.json \
    > /workspace/vf/audit.log 2>&1 || { mark AUDIT_FAILED; exit 1; }
  $PY - >> /workspace/vf/audit.log 2>&1 <<'PY' || { mark AUDIT_MOVE_FAILED; exit 1; }
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
  mark "AUDIT_DONE $(grep -oE 'moved [0-9]+ of [0-9]+' /workspace/vf/audit.log)" ;;
cache)
  for OV in 1 0; do
    mkdir -p /workspace/pooled_ov$OV
    for g in $(seq 0 $((NG - 1))); do
      ( $PYC scripts/training/build_pooled_cache.py --data_dir $D --out_dir /workspace/pooled_ov$OV --backbone regnety_032 --weights_path $TFPP \
          --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay $OV --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 \
          --device cuda:$g --shard $g --num_shards $NG --num_workers $(( $(nproc) / NG - 2 )) --batch_size 128 \
          > /workspace/vf/cache_ov${OV}_$g.log 2>&1 && mark CACHE_OV${OV}_SHARD_$g || mark CACHE_OV${OV}_SHARD_${g}_FAILED ) &
      sleep 5
    done
    wait
  done
  ls /workspace/pooled_ov1/*.json /workspace/pooled_ov0/*.json > /dev/null && mark CACHE_DONE || mark CACHE_FAILED ;;
train)
  grep -q AUDIT_DONE $S || { echo "audit first"; exit 1; }
  P1=$(ls /workspace/pooled_ov1/*.json | head -1); P1=${P1%.json}; P0=$(ls /workspace/pooled_ov0/*.json | head -1); P0=${P0%.json}
  run() {  # GPU LABEL extra-args (one run, foreground)
    CUDA_VISIBLE_DEVICES=$1 $PYC -u scripts/training/train_wor.py $COMMON $3 --run_label $2 --save_dir /workspace/checkpoints/$2 \
      > /workspace/vf/$2.log 2>&1; mark "TRAIN_END $2 rc=$?"
  }
  # fix 3 first (the user's order); live encoder, so it takes the longest and gets its own GPU
  ( run 0 vf_s2d "--route_overlay 1 --token_mode s2d --s2d_patch 3 --token_mask 0.2 --vision_grid 6x16 --encoder_channels_last 1 --num_workers $(( $(nproc) / 2 - 4 ))" ) &
  ( run 1 vf_ctrl "--route_overlay 1 --vision_grid 4 --num_workers 8 --pooled_cache $P1"
    run 1 vf_clean "--route_overlay 0 --vision_grid 4 --num_workers 8 --pooled_cache $P0"
    run 1 vf_path "--route_overlay 1 --vision_grid 4 --num_workers 8 --pooled_cache $P1 --path_head 1 --path_cascade 1" ) &
  sleep 240
  for L in vf_s2d vf_ctrl; do echo "== $L"; grep -aE "Resumed|Validation split|pooled|token_mode|Error|Traceback" /workspace/vf/$L.log | tail -5; done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  wait ;;
esac
