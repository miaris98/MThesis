#!/usr/bin/env bash
# TODO A84 (2026-11-16, "fix all of that": learn the expert's reasons, re-balance rare frames, a crash classifier). Every arm is vf_clean's
# exact recipe (E15 epoch 15 -> 30, leak-filtered data, pooled cache WITHOUT the route overlay, seed 0) plus one A84 change, so the controls
# are the three vf_clean seeds already trained (E:\MThesis_EXP\live_20261104_visionfix\checkpoints\vf_clean{,_s1,_s2}); they need lanes only.
#   rb  reason heads + cascade + bucket sampler   (the combined arm: does the package help at all?)
#   dg  detached heads (a danger classifier beside an unchanged policy): lanes twice, gate off and WOR_DANGER_GATE on (a paired test of the gate)
#   rh  reason heads, multi-task only             (ablations, only after rb shows something)
#   rc  reason heads + cascade
#   bs  bucket sampler only
# usage: scripts/ops/a84_box.sh STEP [ARM]       status lines -> $WORKSPACE/a84/status
#   audit            drop Town12/13 route folders that sit on an evaluation route (vision_fix_box.sh audit; must have run)
#   labels           build_reason_labels.py over the data dir (CPU; prints the step-0 counts, JSON in $WORKSPACE/a84/labels_v1.json)
#   cache            pooled 4x4 cache without the route overlay (the cache vf_clean used)
#   readout          offline per-cause readouts (val_reason_*) of the existing checkpoints in $WORKSPACE/checkpoints (no training)
#   train ARM        one arm, foreground, GPU ${GPU:-0}; checkpoints under $WORKSPACE/checkpoints/a84_ARM/
#   loadcheck ARM    the CARLA agent loads the arm and its heads are live
#   lanes ARM [gate] the arm on the three 220-route lanes; `gate` adds WOR_DANGER_GATE (label a84_ARM_gate)
#   base_lanes [S]   vf_clean seed S (0, 1, 2) on the same lanes
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
mkdir -p $WORKSPACE/a84; S=$WORKSPACE/a84/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis; export PYTHONPATH=$WORKSPACE/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python; PYC=$WORKSPACE/venv_carla/bin/python; [ -x $PYC ] || PYC=$PY
D=$WORKSPACE/dataset/wor_trajectories
TFPP=$WORKSPACE/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth
ENC="--weights_path $TFPP"
# 2026-11-17: without the 2.7 GB TF++ zip (5 h from S3 to a Japanese box), the encoder comes from E15's own frozen_backbone.pth: the same
# TF++ image encoder tensor for tensor (the encoder loader and build_pooled_cache --frozen_backbone both accept a run's frozen_backbone.pth)
if [ ! -f "$TFPP" ]; then TFPP=$WORKSPACE/checkpoints/E15/frozen_backbone.pth; ENC="--frozen_backbone $TFPP"; fi
LANES=$WORKSPACE/lanes220.txt
TAG=v1
F="CARLA_QUALITY=Epic"
GATE="WOR_DANGER_GATE=theta=0.6;off=0.4;T=4;release=3;max_release=3"
COMMON="--data_dir $D --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --batch_size 256 --epochs 50 --stop_epoch 30
 --resume_from $WORKSPACE/checkpoints/E15/model_epoch_015.pth --route_key route_original --weights_path $TFPP --freeze_backbone 1 --pretrained 1
 --route_points 4 --use_augmented_camera 1 --lateral_loss_weight 3.0 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --lr_heads 3e-4
 --lr_backbone 1e-4 --crop_bottom_frac 0.0 --save_freq 1 --val_every 5 --use_mlflow 1 --kill_stale 0 --seed 0 --experiment_name a84_reasons
 --route_overlay 0 --vision_grid 4 --num_workers 8 --reason_labels $TAG"

arm_flags() {   # the only place an arm is defined
  case $1 in
    rb) echo "--reason_heads 1 --reason_loss_weight 0.1 --reason_cascade 1 --bucket_alpha 0.5 --bucket_min_cruise 0.3" ;;
    dg) echo "--reason_heads 1 --reason_loss_weight 0.1 --reason_detach 1" ;;
    rh) echo "--reason_heads 1 --reason_loss_weight 0.1" ;;
    rc) echo "--reason_heads 1 --reason_loss_weight 0.1 --reason_cascade 1" ;;
    bs) echo "--bucket_alpha 0.5 --bucket_min_cruise 0.3" ;;
    *) return 1 ;;
  esac
}

cache_prefix() { local p; p=$(ls $WORKSPACE/pooled_ov0/*.json 2>/dev/null | head -1); echo "${p%.json}"; }

lanes() {  # LABEL CKPT_REL EXTRA_FLAGS
  [ "$(wc -l < $LANES)" -ge 3 ] || { mark "LANES_NO_ROUTE_FILE"; return 1; }
  if pgrep -f "queue_A9c.sh 20[0-4]0 " > /dev/null; then mark "LANES_BUSY $1"; return 1; fi
  mkdir -p $WORKSPACE/evlogs
  for k in 1 2 3; do
    L=$(sed -n ${k}p $LANES); P=$((2000 + 10 * k))
    (setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "CKE:${1}_L$k:$2:$L:$3" > $WORKSPACE/evlogs/${1}_L$k.out 2>&1 < /dev/null &)
  done
  mark "LANES_STARTED $1 ckpt $2 flags $3"
}

case $1 in
audit)
  bash scripts/ops/vision_fix_box.sh audit; grep -q AUDIT_DONE $WORKSPACE/vf/status && mark "AUDIT_DONE" || { mark AUDIT_FAILED; exit 1; } ;;
labels)
  $PY scripts/training/build_reason_labels.py --data_dir $D --tag $TAG --stats_out $WORKSPACE/a84/labels_$TAG.json \
    > $WORKSPACE/a84/labels.log 2>&1 && mark "LABELS_DONE $(grep -m1 '^danger:' $WORKSPACE/a84/labels.log)" || { mark LABELS_FAILED; exit 1; } ;;
cache)
  mkdir -p $WORKSPACE/pooled_ov0
  $PYC scripts/training/build_pooled_cache.py --data_dir $D --out_dir $WORKSPACE/pooled_ov0 --backbone regnety_032 $ENC \
    --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 0 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 \
    --device cuda:0 --shard 0 --num_shards 1 --num_workers $(( $(nproc) > 20 ? 16 : $(nproc) - 4 )) --batch_size 128 \
    > $WORKSPACE/a84/cache.log 2>&1 && ls $WORKSPACE/pooled_ov0/*.json > /dev/null && mark CACHE_DONE || { mark CACHE_FAILED; exit 1; } ;;
readout)
  # Per-cause speed error of the existing policies (where does each one deviate from the expert, by reason?) on the held-out split.
  # Each checkpoint is scored on the cache of its own pixels: vf_clean seeds without the overlay; overlay checkpoints need pooled_ov1.
  P0=$(cache_prefix); [ -n "$P0" ] || { mark "READOUT_REFUSED no pooled_ov0 cache"; exit 1; }
  CK=$(ls -d $WORKSPACE/checkpoints/vf_clean*/model_epoch_030.pth $WORKSPACE/checkpoints/a84_*/model_epoch_030.pth 2>/dev/null | tr '\n' ' ')
  WOR_ROUTE_KEY=route_original WOR_POOLED_CACHE=$P0 WOR_REASON_LABELS=$TAG $PYC scripts/analysis/score_checkpoints_on_val.py --checkpoints $CK --data_dir $D \
    --backbone regnety_032 --policy_arch qwen30m --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 0 --route_points 4 --batch_size 256 \
    --output_dir $WORKSPACE/a84 --run_label a84_readout --experiment_name a84_reasons > $WORKSPACE/a84/readout.log 2>&1 \
    && mark "READOUT_DONE $WORKSPACE/a84/val_rescore.csv" || { mark READOUT_FAILED; exit 1; } ;;
train)
  ARM=${2:?arm}; FL=$(arm_flags "$ARM") || { mark "TRAIN_REFUSED unknown arm $ARM"; exit 1; }
  grep -q AUDIT_DONE $WORKSPACE/vf/status || { mark "TRAIN_REFUSED audit first"; exit 1; }
  grep -q LABELS_DONE $S || { mark "TRAIN_REFUSED labels first"; exit 1; }
  P0=$(cache_prefix); [ -n "$P0" ] || { mark "TRAIN_REFUSED no pooled_ov0 cache"; exit 1; }
  mark "TRAIN_START $ARM flags: $FL"
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PYC -u scripts/training/train_wor.py $COMMON --pooled_cache $P0 $FL \
    --run_label a84_$ARM --save_dir $WORKSPACE/checkpoints/a84_$ARM > $WORKSPACE/a84/$ARM.log 2>&1
  rc=$?; mark "TRAIN_END $ARM rc=$rc $(grep '\[obst\] epoch 030' $WORKSPACE/a84/$ARM.log | grep -o 'reason_danger_auroc [0-9.]*')"; exit $rc ;;
loadcheck)
  ARM=${2:?arm}; CKF=$WORKSPACE/checkpoints/a84_$ARM/model_epoch_030.pth
  $PYC scripts/eval/agent_smoke.py "$CKF" --env WOR_DANGER_GATE=theta=0.6 --trained reason_head.out.weight > $WORKSPACE/a84/loadcheck_$ARM.log 2>&1 \
    && mark "LOADCHECK_OK $ARM $(tail -1 $WORKSPACE/a84/loadcheck_$ARM.log)" || { mark "LOADCHECK_FAILED $ARM"; exit 1; }
  ;;
lanes)
  ARM=${2:?arm}; [ -f $WORKSPACE/checkpoints/a84_$ARM/model_epoch_030.pth ] || { mark "ARM_CKPT_MISSING $ARM"; exit 1; }
  if [ "${3:-}" = gate ]; then lanes a84_${ARM}_gate a84_$ARM/model_epoch_030.pth "$F,$GATE"; else lanes a84_$ARM a84_$ARM/model_epoch_030.pth "$F"; fi ;;
base_lanes)
  SD=${2:-0}; DIR=vf_clean; [ "$SD" = 0 ] || DIR=vf_clean_s$SD
  [ -f $WORKSPACE/checkpoints/$DIR/model_epoch_030.pth ] || { mark "BASE_CKPT_MISSING $DIR"; exit 1; }
  lanes $DIR $DIR/model_epoch_030.pth "$F" ;;
*) echo "unknown step $1"; exit 2 ;;
esac
