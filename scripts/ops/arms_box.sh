#!/usr/bin/env bash
# TODO A81-A83 (2026-11-13, "full throttle"): the five new arms on vf_ctrl's exact recipe (E15 epoch 15 -> 30, leak-filtered data, pooled
# overlay cache, seed 0), then each arm on the 220 Bench2Drive routes, CARLA Epic, against vf_ctrl on the same lanes (same host).
#   s4  speed-head experts, 16 experts, top-4
#   w2  waypoint experts, 16 experts, top-2
#   w4  waypoint experts, 16 experts, top-4
#   ea  vision-token experts (A82), 16 experts, top-2
#   wr  waypoint experts, top-2, scenario-labelled router (A83, weight 0.5)
# usage: scripts/ops/arms_box.sh STEP [ARM]     status lines -> /workspace/arms/status
#   audit           drop Town12/13 route folders that sit on an evaluation route (vision_fix_box.sh audit; must have run)
#   cache           pooled 4x4 cache with the route overlay (the same cache vf_ctrl used)
#   train ARM       one arm, foreground, GPU 0; checkpoints under /workspace/checkpoints/arm_ARM/
#   loadcheck ARM   the CARLA agent loads the arm and its new experts are live
#   lanes ARM       the arm on the three 220-route lanes (/workspace/lanes220.txt), refuses while any lane runs
#   base_lanes      vf_ctrl on the same three lanes (needs CARLA only)
set -uo pipefail
mkdir -p /workspace/arms; S=/workspace/arms/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis; export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python; PYC=/workspace/venv_carla/bin/python; [ -x $PYC ] || PYC=$PY
D=/workspace/dataset/wor_trajectories
TFPP=/workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
LANES=/workspace/lanes220.txt
F="CARLA_QUALITY=Epic"
COMMON="--data_dir $D --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 30
 --resume_from /workspace/checkpoints/E15/model_epoch_015.pth --route_key route_original --weights_path $TFPP --freeze_backbone 1 --pretrained 1
 --route_points 4 --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4
 --lr_backbone 1e-4 --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --use_mlflow 1 --kill_stale 0 --seed 0 --experiment_name arms_moe
 --route_overlay 1 --vision_grid 4 --num_workers 8"

arm_flags() {   # the only place an arm is defined
  case $1 in
    s4) echo "--speed_moe_experts 16 --speed_moe_hidden 32 --speed_moe_top_k 4 --moe_aux_weight 0.01" ;;
    w2) echo "--moe_experts 16 --moe_hidden 32 --moe_top_k 2 --moe_aux_weight 0.01" ;;
    w4) echo "--moe_experts 16 --moe_hidden 32 --moe_top_k 4 --moe_aux_weight 0.01" ;;
    ea) echo "--vision_moe_experts 16 --vision_moe_hidden 16 --vision_moe_top_k 2 --moe_aux_weight 0.01" ;;
    wr) echo "--moe_experts 16 --moe_hidden 32 --moe_top_k 2 --moe_route_label_weight 0.5 --moe_aux_weight 0.01" ;;
    *) return 1 ;;
  esac
}

lanes() {  # ARM_LABEL CKPT_REL (under /workspace/checkpoints)
  [ "$(wc -l < $LANES)" -ge 3 ] || { mark "LANES_NO_ROUTE_FILE"; return 1; }
  if pgrep -f "queue_A9c.sh 20[0-4]0 " > /dev/null; then mark "LANES_BUSY $1"; return 1; fi
  mkdir -p /workspace/evlogs
  for k in 1 2 3; do
    L=$(sed -n ${k}p $LANES); P=$((2000 + 10 * k))
    (setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "CKE:${1}_L$k:$2:$L:$F" > /workspace/evlogs/${1}_L$k.out 2>&1 < /dev/null &)
  done
  mark "LANES_STARTED $1 ckpt $2"
}

case $1 in
audit)
  bash scripts/ops/vision_fix_box.sh audit; grep -q AUDIT_DONE /workspace/vf/status && mark "AUDIT_DONE" || { mark AUDIT_FAILED; exit 1; } ;;
cache)
  mkdir -p /workspace/pooled_ov1
  $PYC scripts/training/build_pooled_cache.py --data_dir $D --out_dir /workspace/pooled_ov1 --backbone regnety_032 --weights_path $TFPP \
    --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 1 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 \
    --device cuda:0 --shard 0 --num_shards 1 --num_workers $(( $(nproc) > 20 ? 16 : $(nproc) - 4 )) --batch_size 128 \
    > /workspace/arms/cache.log 2>&1 && ls /workspace/pooled_ov1/*.json > /dev/null && mark CACHE_DONE || { mark CACHE_FAILED; exit 1; } ;;
train)
  ARM=${2:?arm}; FL=$(arm_flags "$ARM") || { mark "TRAIN_REFUSED unknown arm $ARM"; exit 1; }
  grep -q AUDIT_DONE /workspace/vf/status || { mark "TRAIN_REFUSED audit first"; exit 1; }
  P1=$(ls /workspace/pooled_ov1/*.json | head -1); P1=${P1%.json}
  mark "TRAIN_START $ARM flags: $FL"
  CUDA_VISIBLE_DEVICES=0 $PYC -u scripts/training/train_wor.py $COMMON --pooled_cache $P1 $FL \
    --run_label arm_$ARM --save_dir /workspace/checkpoints/arm_$ARM > /workspace/arms/$ARM.log 2>&1
  rc=$?; mark "TRAIN_END $ARM rc=$rc"; exit $rc ;;
loadcheck)
  ARM=${2:?arm}; CKF=/workspace/checkpoints/arm_$ARM/model_epoch_030.pth
  $PYC - > /workspace/arms/loadcheck_$ARM.log 2>&1 <<EOF && mark "LOADCHECK_OK $ARM $(tail -1 /workspace/arms/loadcheck_$ARM.log)" || { mark "LOADCHECK_FAILED $ARM"; exit 1; }
import numpy as np, torch
from src.agents.wor_agent import WorldOnRailsAgent
a = WorldOnRailsAgent(checkpoint_path="$CKF", policy_arch="qwen30m", backbone_name="regnety_032")
n = a.net
arm = "$ARM"
if arm == "s4":
    assert n.speed_moe is not None and float(n.speed_moe.w2.abs().sum()) > 0, "speed experts missing or still at zero init"
if arm in ("w2", "w4", "wr"):
    assert n.trunk.moe is not None and float(n.trunk.moe.w2.abs().sum()) > 0, "waypoint experts missing or still at zero init"
if arm == "ea":
    assert n.vis_moe is not None and float(n.vis_moe.w2.abs().sum()) > 0, "vision experts missing or still at zero init"
x = np.random.default_rng(0).integers(0, 255, (512, 1024, 3), dtype=np.uint8)
r = np.array([[2, 0], [4, 0], [6, 0], [8, 0]], np.float32)
for v in (0.0, 6.0):
    c = a.run_step({"rgb_front": x, "speed": v, "command": 3, "route": r})
print(f"arm {arm} control {c.steer:.3f} {c.throttle:.3f} {c.brake:.3f}")
EOF
  ;;
lanes)
  ARM=${2:?arm}; [ -f /workspace/checkpoints/arm_$ARM/model_epoch_030.pth ] || { mark "ARM_CKPT_MISSING $ARM"; exit 1; }
  lanes arm_$ARM arm_$ARM/model_epoch_030.pth ;;
base_lanes)
  [ -f /workspace/checkpoints/vf_ctrl/model_epoch_030.pth ] || { mark BASE_CKPT_MISSING; exit 1; }
  lanes vf_ctrl vf_ctrl/model_epoch_030.pth ;;
*) echo "unknown step $1"; exit 2 ;;
esac
