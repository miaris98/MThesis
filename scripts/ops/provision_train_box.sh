#!/usr/bin/env bash
# TODO A59 training box (S-126): arm J's data, the TF++ weights, arm E e15, then the pooled-token cache of every frame, built on all GPUs. Run on the box, repo at /workspace/MThesis,
# HF token in /dev/shm/hf_token (stdin from the laptop, never printed). Needs >= 400 GB disk (6-town set ~150 GB + Town12/13 obstacle archives ~70 GB + cache 20 GB), >= 64 GB RAM.
#
#   bash scripts/ops/provision_train_box.sh          # all steps; each writes a marker in /workspace/train_prov/
#   (steps are resumable: the downloader skips extracted towns, the cache builder skips frames flagged in <prefix>.done)
set -uo pipefail
cd /workspace/MThesis; mkdir -p /workspace/train_prov /workspace/pooled
PY=/venv/main/bin/python; PYC=/workspace/venv_carla/bin/python
mark() { echo "$(date -u +%H:%M:%S) $1" >> /workspace/train_prov/status; }
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 https://huggingface.co || true); [ "$code" = 200 ] || { echo "HF NOT REACHABLE: destroy this box"; exit 1; }
df -h /workspace | tail -1; free -g | sed -n 2p; nproc; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 1. data = arm J's recipe (6 towns + the Town12/13 obstacle archives); the regex of docs/todo A14 / launch_armJ2_armK.sh
ARCH='^Town(0[1-5]|10)/|^Town1[23]/.*(Accident|Obstacle|HazardAtSideLane|InvadingTurn|OpensDoor)'
( $PY scripts/setup/download_pdm_lite.py --towns Town01,Town02,Town03,Town04,Town05,Town10,Town12,Town13 --archive-regex "$ARCH" --reserve-gb 60 --dest /workspace/dataset/wor_trajectories \
    > /workspace/train_prov/data.log 2>&1 && mark DATA_DONE || mark DATA_FAILED ) &
# 2. TF++ weights (the encoder's source) + arm E e15 (the resume point of every head-only arm) from the HF relay
( curl -sL -o /workspace/tfpp_pretrained.zip https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip && mkdir -p tfpp_pretrained && \
  unzip -q -o /workspace/tfpp_pretrained.zip "pretrained_models/all_towns/model_0030_0.pth" -d /workspace/tfpp_pretrained && mark TFPP_DONE || mark TFPP_FAILED ) &
( $PY - > /workspace/train_prov/ckpt.log 2>&1 <<'PY' && mark E15_DONE || mark E15_FAILED
import hashlib, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
sub = "e15_20260930/carla_armE_aug1_hires"
lfs = {x.path: x.lfs.sha256 for x in api.list_repo_tree(repo, path_in_repo=sub, recursive=True) if getattr(x, "lfs", None)}
dest = "/workspace/checkpoints/carla_armE_aug1_hires"; os.makedirs(dest, exist_ok=True)
for f in ("frozen_backbone.pth", "model_epoch_015.pth", "run_config.json"):
    p = hf_hub_download(repo, f"{sub}/{f}", token=api.token, local_dir="/workspace/hf_dl"); os.replace(p, f"{dest}/{f}")
    h = hashlib.sha256(open(f"{dest}/{f}", "rb").read()).hexdigest(); ref = lfs.get(f"{sub}/{f}")
    assert ref is None or ref == h, f"sha mismatch {f}"; print(f, os.path.getsize(f"{dest}/{f}"), h[:16], flush=True)
PY
) &
# the training venv needs tensorboard + mlflow (venv_carla has torch/timm; run scripts/ops/provision_eval.sh first on a box that also evaluates)
[ -x $PYC ] && VIRTUAL_ENV=/workspace/venv_carla uv pip install -q tensorboard mlflow 2>/dev/null
wait
grep -q DATA_DONE /workspace/train_prov/status && grep -q TFPP_DONE /workspace/train_prov/status && grep -q E15_DONE /workspace/train_prov/status || { echo "a download step failed: see /workspace/train_prov/"; cat /workspace/train_prov/status; exit 1; }
# 3. the pooled cache: one build process per GPU (shards), fp32, J's pixels (288x768, overlay, route_original, recovery views), 4x4 grid
NG=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NG - 1))); do
  ( PYTHONPATH=/workspace/MThesis $PYC scripts/training/build_pooled_cache.py --data_dir /workspace/dataset/wor_trajectories --out_dir /workspace/pooled \
      --backbone regnety_032 --weights_path /workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth --img_size 288x768 --crop_bottom_frac 0.0 \
      --route_overlay 1 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 --device cuda:$g --shard $g --num_shards $NG \
      --num_workers $(( $(nproc) / NG - 2 )) --batch_size 128 > /workspace/train_prov/cache_$g.log 2>&1 && mark CACHE_SHARD_$g || mark CACHE_SHARD_${g}_FAILED ) &
  sleep 5
done
wait
ls /workspace/pooled/*.json && mark PROVISION_TRAIN_DONE
