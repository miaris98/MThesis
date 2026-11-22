#!/usr/bin/env bash
# Teacher ladder (2026-11-18; user: skip the noise sweep, do steps 2-4, but first understand what limits the RGB student).
# Plan: docs/ops/2026-11-17_challenges_and_solutions.md, "Next steps". One box trains and drives:
#   R3 t_vis    privileged teacher on the objects inside the camera image (with speed)          -> camera-reachable ceiling, one frame + speed
#   R2 t_visns  the same without speed                                                          -> single-frame ceiling
#   t_cov       w8_teacher's recipe (all objects) + the Town12/13 archives of the six scenario families with NO training data at all
#               (StaticCutIn, ParkingCutIn, OppositeVehicleTakingPriority, NonSignalizedJunction{Left,Right}Turn, EnterActorFlow),
#               where the teacher scored 33-65                                                  -> step 4: data coverage or a behaviour-cloning limit?
#   w8 dump     the existing w8_teacher (R4, 82.1 DS) on those families' 30 routes with B2D_PRIV_DUMP (live tokens, step 4) and as the
#               same-host reference for t_cov
# Every teacher arm is w8_teacher's recipe (spatial2 2026-11-08: E15 epoch 15 -> 30, 288x768, seed 0) except the cache's route overlay: off,
# like vf_clean (the student). A teacher never reads the camera tokens, so the overlay only changes pixels it ignores. The base cache also
# stores the PCA-256 full 9x24 map (as spatial2 did) for step 2, the detector on the student's own features. Teachers read
# true objects: diagnostics only, never a thesis agent (single RGB camera at inference, user 2026-11-18).
# usage: scripts/ops/ladder_box.sh STEP [ARGS]        status lines -> $WORKSPACE/ladder/status
#   setup               deps, both data sets (parallel per town), CARLA (background, waited for)
#   audit               leak audit of both data sets (Town12/13 route folders on a Bench2Drive-220 route move out), then the merged dir
#   cache D|DX          pooled 4x4 cache, no overlay, of the base data (D, + PCA-256 full map) or the six new families only (DX);
#                       t_cov reads both prefixes over the merged dir DC (one encoder pass less)
#   targets             privileged token files: _vis and _visns on D, all objects on DC
#   train ARM           t_vis | t_visns | t_cov, foreground
#   lanes LIST N JOBSPEC...   start N lanes; each JOBSPEC = LABEL:CKPT_REL:ROUTEFILE:FLAGS, its routes split round-robin over the lanes
# R1 day (2026-11-19, r1_chain.sh): the RGB student with detector object tokens appended (docs/ops/2026-11-18_teacher_ladder.md, steps 4-6, 8a)
#   cache DX            also writes the PCA-256 map of the coverage data, IN THE BASE CACHE'S BASIS (the detector reads it on both)
#   targets_r1          detector targets (_visns on D) and all-object tokens on DX (the cut-in check)
#   detector            train_object_detector.py on D, best checkpoint by val F1
#   export              detector tokens (<prefix>.boxtok20_det.f16) for D and DX
#   cutin               cut-in log-side token counts (scripts/analysis/cutin_token_check.py) -> $WORKSPACE/ladder/cutin_log_side.json
#   train r1_s0|r1_s1|r0ft   R1 seeds 0/1 and the same recipe without object tokens (the same-box RGB control), on D + DX
#   gate                offline gate: R1's val waypoint ADE at epoch 30 must beat r0ft's
# 2026-11-20 (ctl_chain.sh: controller study box; mx_chain.sh: R1 + experts box)
#   setup_carla         CARLA only (a box that only drives shipped checkpoints)
#   train r1w4_s0       R1 + w4's waypoint experts (16 experts, top-4, hidden 32, aux 0.01; w4 = the best MoE arm of 11-15)
#   train r0ftf         r0ft with a fresh AdamW (WOR_RESET_OPTIMIZER=1), so it restarts its optimizer as every R1 arm does
#   valade              VALADE status line: val waypoint ADE of the r1d_* arms on this box
#   ctl_winner          after the 80-route controller screen: the best non-pid controller -> $WORKSPACE/ladder/ctl_winner.env
#                       (LANE_SKIP=1 when it trails pid by more than 7 DS, the futility rule of docs/ops/fast_eval/carla-subset-proxy.md)
# 2026-11-21 (mem_chain.sh: memorization ladder, a training-only box; plan: docs/ops/2026-11-21_memorization_ladder.md)
#   NO_CARLA=1          setup installs the trainer's packages into /venv/main instead of provisioning CARLA (pass PYC=/venv/main/bin/python)
#   D_PCA=0 / DX_PCA=0  cache without the PCA-256 full map (only the detector arms read it)
#   train mem_L<k>[w]   the RGB control recipe (r0ftf: E15 -> 30, fresh AdamW) at regularization level k (0-3), w = + w4's experts;
#                       SEED=1 trains seed 1 (label suffix _s1)
#   memtable            MEMTABLE status lines: clean-train vs held-out ADE, ratio, experts' effect per level
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
mkdir -p $WORKSPACE/ladder; S=$WORKSPACE/ladder/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis; export PYTHONPATH=$WORKSPACE/MThesis MLFLOW_ALLOW_FILE_STORE=true
PY=/venv/main/bin/python; PYC=${PYC:-/workspace/venv_carla/bin/python}
D=$WORKSPACE/dataset/wor_trajectories            # w8_teacher's data: 6 towns + the Town12/13 obstacle archives
DX=$WORKSPACE/dataset_cov/wor_trajectories       # the six uncovered families (Town12/13)
DC=$WORKSPACE/dataset_all/wor_trajectories       # D + DX as hard links (no extra disk)
ARCH='^Town(0[1-5]|10)/|^Town1[23]/.*(Accident|Obstacle|HazardAtSideLane|InvadingTurn|OpensDoor)'
COV='^Town1[23]/data/(StaticCutIn|ParkingCutIn|OppositeVehicleTakingPriority|NonSignalizedJunctionLeftTurn|NonSignalizedJunctionRightTurn|EnterActorFlow)\.zip$'
E15=$WORKSPACE/checkpoints/E15
ENC=$E15/frozen_backbone.pth   # the TF++ image encoder tensor for tensor (2026-11-17: no 2.7 GB S3 zip); teachers do not read it anyway
# spatial2's COMMON (w8_teacher's run_config) + its old geometry
COMMON="--backbone regnety_032 --policy_arch qwen30m --epochs 50 --stop_epoch 30 --resume_from $E15/model_epoch_015.pth --route_key route_original
 --weights_path $ENC --freeze_backbone 1 --pretrained 1 --route_points 4 --use_augmented_camera 1 --lateral_loss_weight 3.0
 --target_speed_loss_weight 0.2 --wp_loss_weight 1.0 --q_loss_weight 0.0 --use_rail_q 0 --lr_heads 3e-4 --lr_backbone 1e-4 --split_seed 0
 --seed 0 --kill_stale 0 --route_overlay 0 --vision_grid 4 --img_size 288x768 --crop_bottom_frac 0.0 --batch_size 256 --val_split 0.15
 --val_every 5 --save_freq 5 --use_mlflow 1"
COMMON_RGB="$COMMON --experiment_name r1_detector"          # R1 day: the student and its control read the camera tokens
COMMON="$COMMON --experiment_name teacher_ladder --privileged_boxes 20"
DET=$WORKSPACE/checkpoints/det_visns/detector_best.pt
OBJ_RANGE=${OBJ_RANGE:-30}

CORES=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null)   # the cgroup quota, not nproc
[ "${CORES:-0}" -gt 0 ] 2>/dev/null || CORES=$(nproc)                                                                 # (2026-11-18 box: 128 threads, quota 20)

prefix_of() { local p; p=$(ls $1/*.json 2>/dev/null | grep -v -E "pca|basis|boxtok" | head -1); echo "${p%.json}"; }   # boxtok: the token export's .json sorts first (2026-11-19 chain stop)
prefix_grid() { local p; p=$(ls $1/*_g$2.json 2>/dev/null | head -1); echo "${p%.json}"; }   # 2026-11-21: one dir holds the 9x24 and the 4x4 cache

case $1 in
setup)
  mark SETUP_START
  mkdir -p $D $DX $WORKSPACE/logs_prov
  ( $PY -m pip install -q requests huggingface_hub > $WORKSPACE/ladder/pip.log 2>&1
    for T in Town01 Town02 Town03 Town04 Town05 Town10 Town12 Town13; do
      ( $PY scripts/setup/download_pdm_lite.py --towns $T --archive-regex "$ARCH" --reserve-gb 20 --dest $D > $WORKSPACE/ladder/data_$T.log 2>&1 ) &
    done
    for T in Town12 Town13; do
      ( $PY scripts/setup/download_pdm_lite.py --towns $T --archive-regex "$COV" --reserve-gb 20 --dest $DX > $WORKSPACE/ladder/datacov_$T.log 2>&1 ) &
    done
    wait
    nf=$(cat $WORKSPACE/ladder/data_*.log $WORKSPACE/ladder/datacov_*.log | grep -c -E "FAILED|Traceback")
    [ "$nf" = 0 ] && mark "DATA_DONE D $(du -sh $D | cut -f1) DX $(du -sh $DX | cut -f1)" || mark "DATA_FAILED $nf error lines" ) &
  if [ "${NO_CARLA:-0}" = 1 ]; then   # training-only box: the cache builder and the trainers run in /venv/main
    ( $PY -m pip install -q timm mlflow tensorboard opencv-python-headless pytest scipy jsonpickle shapely einops >> $WORKSPACE/logs_prov/prov.out 2>&1   # jsonpickle / shapely / einops: carla_garage (A86)
      if $PY -c "import torch, torchvision, timm, mlflow, tensorboard, cv2, jsonpickle, shapely; assert torch.cuda.is_available(); print('torch', torch.__version__)" >> $WORKSPACE/logs_prov/prov.out 2>&1
      then mark CARLA_PROV_END; else mark CARLA_PROV_FAILED; fi ) &
  else
  ( bash scripts/ops/provision_carla_no_hf.sh > $WORKSPACE/logs_prov/prov.out 2>&1
    bash scripts/ops/install_guard.sh carla_tools $WORKSPACE/venv_carla -- -q tensorboard mlflow pytest scipy >> $WORKSPACE/logs_prov/prov.out 2>&1
    if grep -q PROVISION_DONE $WORKSPACE/setup_carla.log 2>/dev/null && \
       $PYC -c "import torch, timm, carla, py_trees, tensorboard, mlflow" >> $WORKSPACE/logs_prov/prov.out 2>&1
    then mark CARLA_PROV_END; else mark CARLA_PROV_FAILED; fi ) &
  fi
  wait; mark SETUP_END ;;
setup_carla)
  mark SETUP_START; mkdir -p $WORKSPACE/logs_prov
  bash scripts/ops/provision_carla_no_hf.sh > $WORKSPACE/logs_prov/prov.out 2>&1
  bash scripts/ops/install_guard.sh carla_tools $WORKSPACE/venv_carla -- -q tensorboard mlflow pytest scipy >> $WORKSPACE/logs_prov/prov.out 2>&1
  if grep -q PROVISION_DONE $WORKSPACE/setup_carla.log 2>/dev/null && \
     $PYC -c "import torch, timm, carla, py_trees, tensorboard, mlflow" >> $WORKSPACE/logs_prov/prov.out 2>&1
  then mark CARLA_PROV_END; else mark CARLA_PROV_FAILED; fi
  mark SETUP_END ;;
valade)
  $PY - <<'PYX' >> $S
import csv, glob, os
out = []
for f in sorted(glob.glob("/workspace/checkpoints/r1d_*/wor_training_telemetry.csv")):
    rows = [r for r in csv.DictReader(open(f)) if (r.get("val_wp_ade_m") or "").strip()]
    if rows:
        out.append(f"{os.path.basename(os.path.dirname(f))[4:]} {float(rows[-1]['val_wp_ade_m']):.4f} ep {rows[-1]['epoch']}")
print("VALADE " + " | ".join(out) + "   (11-19 box: r1_s0 0.2715, r1_s1 0.2710, r0ft 0.2811)")
PYX
  ;;
ctl_winner)
  $PY - <<'PYX' >> $S
import glob, json, os, statistics
OUT = "/workspace/bench2drive_out"
MODES = {"pid": "pid", "brake": "brake", "pdmlong": "pdm_long", "pdmlat": "pdm_lat", "pdm": "pdm"}
rec = {}
for tag in MODES:
    d = {}
    for f in glob.glob(f"{OUT}/ctl_scr_{tag}_L*_x.json"):
        for x in json.load(open(f))["_checkpoint"]["records"]:
            if "rash" not in x["status"]:
                d[x["route_id"]] = x["scores"]["score_composed"]
    rec[tag] = d
common = set.intersection(*[set(v) for v in rec.values()])
mean = {t: statistics.mean(rec[t][r] for r in common) for t in MODES} if common else {}
alts = [t for t in MODES if t != "pid"]
best = max(alts, key=lambda t: mean.get(t, -1)) if mean else None
table = " ".join(f"{t} {mean[t]:.1f}" for t in MODES) if mean else "no common routes"
if best is None or mean[best] < mean["pid"] - 7.0:
    open("/workspace/ladder/ctl_winner.env", "w").write("LANE_SKIP=1\n")
    print(f"CTL_SCREEN n {len(common)}: {table} -> no confirm run (best alternative {best} trails pid by more than 7 DS)")
else:
    open("/workspace/ladder/ctl_winner.env", "w").write(f"WOR_CONTROLLER={MODES[best]}\n")
    print(f"CTL_SCREEN n {len(common)}: {table} -> WINNER {best} (WOR_CONTROLLER={MODES[best]}) confirmed on the other 140 routes")
PYX
  ;;
audit)
  for pair in "$D:$WORKSPACE/dataset_excluded" "$DX:$WORKSPACE/dataset_excluded_cov"; do
    DD=${pair%%:*}; XX=${pair#*:}; tag=$(basename $(dirname $DD))
    $PY scripts/analysis/route_overlap_audit.py --eval_xml $WORKSPACE/ladder/bench2drive220.xml --data_dir $DD --out_json $WORKSPACE/ladder/leak_$tag.json \
      > $WORKSPACE/ladder/audit_$tag.log 2>&1 || { mark "AUDIT_FAILED $tag"; exit 1; }
    $PY - "$DD" "$XX" $WORKSPACE/ladder/leak_$tag.json >> $WORKSPACE/ladder/audit_$tag.log 2>&1 <<'PYX' || { mark "AUDIT_MOVE_FAILED $tag"; exit 1; }
import json, os, shutil, sys
D, X, J = sys.argv[1:4]
rows = json.load(open(J))["routes"]
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
PYX
    mark "AUDIT_DONE $tag $(grep -oE 'moved [0-9]+ of [0-9]+' $WORKSPACE/ladder/audit_$tag.log)"
  done
  rm -rf $DC; mkdir -p $DC && cp -al $D/. $DC/ && cp -al $DX/. $DC/ || { mark MERGE_FAILED; exit 1; }
  mark "MERGED $(find $D -mindepth 4 -maxdepth 4 -type d | wc -l) + $(find $DX -mindepth 4 -maxdepth 4 -type d | wc -l) route folders -> $(find $DC -mindepth 4 -maxdepth 4 -type d | wc -l)" ;;
cache)
  W=${2:?D or DX}; DD=$D; OUTD=$WORKSPACE/pooled_ov0; PCA="--pca_full 256 --pca_fit_frames 3000"
  [ "${D_PCA:-1}" = 0 ] && PCA=""
  [ "$W" = DX ] && { DD=$DX; OUTD=$WORKSPACE/pooled_cov_ov0; PCA=""; }
  mkdir -p $OUTD
  if [ "$W" = DX ] && [ "${DX_PCA:-1}" = 1 ]; then   # R1: the detector reads the coverage frames too, so their map is compressed in D's basis
    PD=$(prefix_of $WORKSPACE/pooled_ov0); B=$PD.pca256.f16.basis.pt
    [ -n "$PD" ] && [ -f "$B" ] || { mark "CACHE_REFUSED DX: no base-cache PCA basis to reuse"; exit 1; }
    cp $B $OUTD/$(basename $PD).pca256.f16.basis.pt && PCA="--pca_full 256"
  fi
  $PYC scripts/training/build_pooled_cache.py --data_dir $DD --out_dir $OUTD --backbone regnety_032 --frozen_backbone $ENC \
    --img_size 288x768 --crop_bottom_frac 0.0 --route_overlay 0 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 4 $PCA \
    --device cuda:0 --shard 0 --num_shards 1 --num_workers $(( CORES > 20 ? 16 : CORES - 4 )) --batch_size 128 \
    > $WORKSPACE/ladder/cache_$W.log 2>&1 && [ -n "$(prefix_of $OUTD)" ] && mark "CACHE_DONE $W $(prefix_of $OUTD)" || { mark "CACHE_FAILED $W"; exit 1; } ;;
targets)
  PD=$(prefix_of $WORKSPACE/pooled_ov0); PC=$(prefix_of $WORKSPACE/pooled_cov_ov0)
  [ -n "$PD" ] && [ -n "$PC" ] || { mark "TARGETS_REFUSED caches missing"; exit 1; }
  NW=$(( CORES / 3 > 2 ? CORES / 3 : 2 ))
  ( $PY scripts/training/build_box_targets.py --prefix $PD --data_dir $D --box_tokens 20 --box_variant _vis --workers $NW > $WORKSPACE/ladder/targets_vis.log 2>&1 ) & a=$!
  ( $PY scripts/training/build_box_targets.py --prefix $PD --data_dir $D --box_tokens 20 --box_variant _visns --workers $NW > $WORKSPACE/ladder/targets_visns.log 2>&1 ) & b=$!
  ( $PY scripts/training/build_box_targets.py --prefix $PC --data_dir $DX --box_tokens 20 --workers $NW > $WORKSPACE/ladder/targets_cov.log 2>&1 ) & c=$!
  ( $PY scripts/training/build_box_targets.py --prefix $PD --data_dir $D --box_tokens 20 --workers $NW > $WORKSPACE/ladder/targets_all.log 2>&1 ) & e=$!
  wait $a; ra=$?; wait $b; rb=$?; wait $c; rc=$?; wait $e; re=$?
  [ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] && [ $re = 0 ] && [ -f $PD.boxtok20.f16 ] && [ -f $PD.boxtok20_vis.f16 ] && [ -f $PD.boxtok20_visns.f16 ] && [ -f $PC.boxtok20.f16 ] \
    && mark TARGETS_DONE || { mark "TARGETS_FAILED rc $ra $rb $rc $re"; exit 1; } ;;
targets_r1)
  PD=$(prefix_of $WORKSPACE/pooled_ov0); PC=$(prefix_of $WORKSPACE/pooled_cov_ov0)
  [ -n "$PD" ] && [ -n "$PC" ] || { mark "TARGETS_REFUSED caches missing"; exit 1; }
  NW=$(( CORES / 2 > 2 ? CORES / 2 : 2 ))
  ( $PY scripts/training/build_box_targets.py --prefix $PD --data_dir $D --box_tokens 20 --box_variant _visns --workers $NW > $WORKSPACE/ladder/targets_visns.log 2>&1 ) & a=$!
  ( $PY scripts/training/build_box_targets.py --prefix $PC --data_dir $DX --box_tokens 20 --workers $NW > $WORKSPACE/ladder/targets_cov.log 2>&1 ) & c=$!
  wait $a; ra=$?; wait $c; rc=$?
  [ $ra = 0 ] && [ $rc = 0 ] && [ -f $PD.boxtok20_visns.f16 ] && [ -f $PC.boxtok20.f16 ] && mark TARGETS_DONE || { mark "TARGETS_FAILED rc $ra $rc"; exit 1; } ;;
detector)
  PD=$(prefix_of $WORKSPACE/pooled_ov0); OUT=$(dirname $DET)
  mark DETECTOR_START
  $PYC -u scripts/training/train_object_detector.py --prefix $PD --out_dir $OUT --epochs ${DET_EPOCHS:-12} --max_minutes ${DET_MINUTES:-170} \
    --select f1 --num_workers $(( CORES > 12 ? 8 : 4 )) --experiment_name r1_detector > $WORKSPACE/ladder/detector.log 2>&1 && [ -f $DET ] \
    && mark "DETECTOR_DONE $(grep -a '^FINAL' $WORKSPACE/ladder/detector.log | grep -oE '"(recall|precision|f1|recall_0_10m|recall_30_50m)": [0-9.]+' | tr '\n' ' ')" \
    || { mark DETECTOR_FAILED; exit 1; } ;;
export)
  PD=$(prefix_of $WORKSPACE/pooled_ov0); PC=$(prefix_of $WORKSPACE/pooled_cov_ov0)
  $PYC -u scripts/training/export_detector_tokens.py --prefix $PD,$PC --detector $DET --m 20 --variant _det --max_range $OBJ_RANGE \
    > $WORKSPACE/ladder/export.log 2>&1 && [ -f $PD.boxtok20_det.f16 ] && [ -f $PC.boxtok20_det.f16 ] \
    && mark "EXPORT_DONE $(grep -oE 'objects per frame' -B0 $WORKSPACE/ladder/export.log | wc -l) caches, $(grep -oE '[0-9.]+ objects per frame' $WORKSPACE/ladder/export.log | tr '\n' ' ')" \
    || { mark EXPORT_FAILED; exit 1; } ;;
cutin)
  PC=$(prefix_of $WORKSPACE/pooled_cov_ov0)
  $PY scripts/analysis/cutin_token_check.py --prefix $PC --out $WORKSPACE/ladder/cutin_log_side.json > $WORKSPACE/ladder/cutin.log 2>&1 \
    && mark CUTIN_DONE || mark CUTIN_FAILED ;;
gate)
  $PY - <<'PYX' >> $S
import csv, statistics
def ade(arm):
    rows = [r for r in csv.DictReader(open(f"/workspace/checkpoints/r1d_{arm}/wor_training_telemetry.csv")) if (r.get("val_wp_ade_m") or "").strip()]
    return float(rows[-1]["val_wp_ade_m"]), rows[-1]["epoch"]
try:
    r1 = [ade(a) for a in ("r1_s0", "r1_s1")]
    c, ce = ade("r0ft")
    m = statistics.mean(v for v, _ in r1)
    verdict = "GATE_PASS" if m < c else "GATE_FAIL"
    print(f"{verdict} val ADE R1 {m:.4f} (s0 {r1[0][0]:.4f} ep {r1[0][1]}, s1 {r1[1][0]:.4f} ep {r1[1][1]}) vs r0ft {c:.4f} ep {ce}")
except Exception as e:
    print(f"GATE_ERROR {e!r}"[:200])
PYX
  ;;
train)
  ARM=${2:?arm}; PD=$(prefix_of $WORKSPACE/pooled_ov0); PC=$(prefix_of $WORKSPACE/pooled_cov_ov0)
  BASE=$COMMON; LAB=ladder_$ARM
  OBJ="--object_tokens 20 --object_detector $DET --object_max_range $OBJ_RANGE"
  case $ARM in
    r1_s0) BASE=$COMMON_RGB; LAB=r1d_$ARM; FL="--data_dir $DC --pooled_cache $PD,$PC $OBJ" ;;
    r1_s1) BASE=$COMMON_RGB; LAB=r1d_$ARM; FL="--data_dir $DC --pooled_cache $PD,$PC $OBJ --seed 1" ;;
    r0ft)  BASE=$COMMON_RGB; LAB=r1d_$ARM; FL="--data_dir $DC --pooled_cache $PD,$PC" ;;
    r1w4_s0) BASE=$COMMON_RGB; LAB=r1d_$ARM
             FL="--data_dir $DC --pooled_cache $PD,$PC $OBJ --moe_experts 16 --moe_hidden 32 --moe_top_k 4 --moe_aux_weight 0.01 --experiment_name r1_moe" ;;
    r0ftf) BASE=$COMMON_RGB; LAB=r1d_$ARM; FL="--data_dir $DC --pooled_cache $PD,$PC --experiment_name r1_moe --reset_optimizer 1" ;;   # the flag, not the env:
           # train_wor.py sets WOR_RESET_OPTIMIZER from --reset_optimizer and overwrote an exported 1 (2026-11-20: the first r0ftf run resumed E15's AdamW)
    mem_L[0-3]|mem_L[0-3]w)   # 2026-11-21 memorization ladder: r0ftf's recipe + training-only regularizers (+ w4's experts for ...w)
           k=${ARM:5:1}; NOISE=(0 0.1 0.25 0.5); DROP=(0 0.1 0.2 0.3); SD=${SEED:-0}
           BASE=$COMMON_RGB; LAB=$ARM$([ "$SD" = 0 ] || echo _s$SD)
           FL="--data_dir $DC --pooled_cache $PD,$PC --experiment_name memorization --reset_optimizer 1 --seed $SD --val_every 1
               --train_eval_frames 20000 --feature_noise ${NOISE[$k]} --token_drop ${DROP[$k]}"
           [ "${ARM: -1}" = w ] && FL="$FL --moe_experts 16 --moe_hidden 32 --moe_top_k 4 --moe_aux_weight 0.01"
           LOGN=$LAB ;;   # seed 1 must not overwrite seed 0's log
    a87_fm|a87_tp|a87_moelat)   # 2026-11-21 A87 round 2 (tfpp_chain.sh), mem_L0's settings; the caches come from cache_full (9x24 + exact 4x4)
           P9D=$(prefix_grid $WORKSPACE/pooled_ov0 9x24); P9C=$(prefix_grid $WORKSPACE/pooled_cov_ov0 9x24)
           P4D=$(prefix_grid $WORKSPACE/pooled_ov0 4x4); P4C=$(prefix_grid $WORKSPACE/pooled_cov_ov0 4x4)
           BASE=$COMMON_RGB; LAB=$ARM; LOGN=$ARM
           FL="--experiment_name a87_tfpp_parts --reset_optimizer 1 --seed 0 --val_every 1 --train_eval_frames 20000 --data_dir $DC"
           case $ARM in
             a87_fm)     FL="$FL --pooled_cache $P9D,$P9C --vision_grid 9x24" ;;   # the whole 9x24 map at 1512 channels: 216 camera tokens
             a87_tp)     FL="$FL --pooled_cache $P4D,$P4C --route_mode dense_tp --route_points 6" ;;   # + TF++'s target point and next one
             a87_moelat) FL="$FL --pooled_cache $P4D,$P4C --moe_experts 16 --moe_hidden 32 --moe_top_k 4 --moe_aux_weight 0.01 --moe_lateral_only 1" ;;
           esac ;;
    a87_path|a87_wd)   # 2026-11-21 A87: one TF++ policy choice each, on mem_L0's exact settings (mem_L0 = the same-box control)
           BASE=$COMMON_RGB; LAB=$ARM; LOGN=$ARM
           FL="--data_dir $DC --pooled_cache $PD,$PC --experiment_name a87_tfpp_parts --reset_optimizer 1 --seed 0 --val_every 1 --train_eval_frames 20000"
           [ $ARM = a87_path ] && FL="$FL --path_head 1"          # TF++'s space-indexed path (10 points); driven with WOR_PATH_STEER=1
           [ $ARM = a87_wd ] && FL="$FL --weight_decay 0.01" ;;   # TF++'s AdamW weight decay (ours: 1e-4)
    t_vis)   FL="--data_dir $D --pooled_cache $PD --privileged_variant _vis" ;;
    t_visns) FL="--data_dir $D --pooled_cache $PD --privileged_variant _visns" ;;
    t_cov)   FL="--data_dir $DC --pooled_cache $PD,$PC" ;;   # both caches hold keys relative to a wor_trajectories dir; all exist in DC
    *) mark "TRAIN_REFUSED unknown arm $ARM"; exit 1 ;;
  esac
  LOGN=${LOGN:-$ARM}
  mark "TRAIN_START $LOGN"
  $PYC -u scripts/training/train_wor.py $BASE $FL --num_workers ${NW:-4} --run_label $LAB --save_dir $WORKSPACE/checkpoints/$LAB \
    > $WORKSPACE/ladder/train_$LOGN.log 2>&1
  rc=$?
  CK=$WORKSPACE/checkpoints/$LAB
  [ -f $CK/frozen_backbone.pth ] || cp $ENC $CK/frozen_backbone.pth 2>/dev/null
  if [ $rc = 0 ] && [ -f $CK/model_epoch_030.pth ]; then touch $CK/TRAINED; fi
  mark "TRAIN_END $LOGN rc=$rc $(grep -aoE '\[memorization\].*' $WORKSPACE/ladder/train_$LOGN.log | tail -1)"; exit $rc ;;
cache_full)   # 2026-11-21 (tfpp_chain.sh): D and DX at 9x24 (TF++ camera-only, a87_fm) + the exact 4x4 (one encoder pass), sharded over every GPU
  NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1
  for W in D DX; do
    DD=$D; OUTD=$WORKSPACE/pooled_ov0; [ $W = DX ] && { DD=$DX; OUTD=$WORKSPACE/pooled_cov_ov0; }
    mkdir -p $OUTD
    pids=""
    for k in $(seq 0 $((NG - 1))); do
      ( $PYC scripts/training/build_pooled_cache.py --data_dir $DD --out_dir $OUTD --backbone regnety_032 --frozen_backbone $ENC --img_size 288x768 \
          --crop_bottom_frac 0.0 --route_overlay 0 --route_points 4 --route_key route_original --use_augmented_camera 1 --grid 9x24 --extra_grid 4 \
          --device cuda:$k --shard $k --num_shards $NG --num_workers $(( CORES / NG > 10 ? 10 : CORES / NG - 1 )) --batch_size 128 \
          > $WORKSPACE/ladder/cachefull_${W}_$k.log 2>&1 ) & pids="$pids $!"
      sleep 20   # shard 0 creates the layout; the others wait for its json
    done
    rc=0; for p in $pids; do wait $p || rc=1; done
    p9=$(prefix_grid $OUTD 9x24); p4=$(prefix_grid $OUTD 4x4)
    left=$($PY -c "import numpy as np,sys; print(sum(int((np.fromfile(p+'.done',dtype=np.uint8)==0).sum()) for p in sys.argv[1:]))" $p9 $p4 2>/dev/null)
    [ $rc = 0 ] && [ -n "$p9" ] && [ -n "$p4" ] && [ "$left" = 0 ] && mark "CACHE_FULL_DONE $W $p9 + 4x4" || { mark "CACHE_FULL_FAILED $W (rc $rc, frames left ${left:-?})"; exit 1; }
  done ;;
train_tfpp)   # 2026-11-21 A86: TF++ camera-only on the 9x24 maps (scripts/training/train_tfpp_cam.py; TF++'s recipe: 31 epochs, batch 128)
  P9D=$(prefix_grid $WORKSPACE/pooled_ov0 9x24); P9C=$(prefix_grid $WORKSPACE/pooled_cov_ov0 9x24); CK=$WORKSPACE/checkpoints/a86_tfpp_cam
  mark "TRAIN_START a86_tfpp_cam"
  $PYC -u scripts/training/train_tfpp_cam.py --data_dir $DC --pooled_cache $P9D,$P9C --frozen_backbone $ENC --save_dir $CK \
    --num_workers ${NW:-6} --experiment_name a86_tfpp_cam --epochs ${TFPP_EPOCHS:-31} ${TFPP_RESUME:+--resume $TFPP_RESUME}     >> $WORKSPACE/ladder/train_a86_tfpp_cam.log 2>&1   # TFPP_EPOCHS=15 + TFPP_RESUME=snapshot (2026-11-22): the first cosine cycles only (ends at epoch 15)
  rc=$?
  [ $rc = 0 ] && [ -f $CK/tfpp_cam/model_$(printf %04d ${TFPP_EPOCHS:-31}).pth ] && [ -f $CK/tfpp_cam/config.json ] && touch $CK/TRAINED
  mark "TRAIN_END a86_tfpp_cam rc=$rc $(grep -a '\[tfpp_cam\] epoch' $WORKSPACE/ladder/train_a86_tfpp_cam.log | tail -1 | cut -c1-150)"; exit $rc ;;
memtable)   # 2026-11-21: one MEMTABLE line per arm (latest validated epoch) + the experts' held-out effect per level
  $PY - <<'PYX' >> $S
import csv, glob, os
rows = {}
for f in sorted(glob.glob("/workspace/checkpoints/mem_L*/wor_training_telemetry.csv")):
    r = [x for x in csv.DictReader(open(f)) if (x.get("train_clean_wp_ade_m") or "").strip()]
    if r:
        rows[os.path.basename(os.path.dirname(f))] = r[-1]
for a, r in rows.items():
    tc, va = float(r["train_clean_wp_ade_m"]), float(r["val_wp_ade_m"])
    print(f"MEMTABLE {a} ep {r['epoch']} clean-train {tc:.4f} held-out {va:.4f} ratio {va / tc:.2f}")
for a in sorted(rows):
    if a.endswith("w") or a.endswith("w_s1"):
        continue
    w = a.replace("_s1", "w_s1") if a.endswith("_s1") else a + "w"
    if w in rows:
        p, q = float(rows[a]["val_wp_ade_m"]), float(rows[w]["val_wp_ade_m"])
        print(f"MEMTABLE experts at {a}: held-out {p:.4f} -> {q:.4f} ({(p - q) / p * 100:+.2f}% better with experts)")
PYX
  ;;
lanes)
  N=${2:?lanes}; shift 2
  mkdir -p $WORKSPACE/evlogs
  for k in $(seq 1 $N); do
    JOBS=()
    for spec in "$@"; do
      IFS=: read -r LABEL CK RF FLAGS <<< "$spec"
      R=$(tr ', ' '\n\n' < $RF | awk 'NF' | awk -v k=$k -v n=$N '(NR - 1) % n == k - 1' | paste -sd, -)
      [ -n "$R" ] && JOBS+=("CKE:${LABEL}_L$k:$CK:$R:$FLAGS")
    done
    P=$((2000 + 10 * k))
    (setsid nohup bash scripts/eval/queue_A9c.sh $P 0 "${JOBS[@]}" > $WORKSPACE/evlogs/ladder_L$k.out 2>&1 < /dev/null &)
    mark "LANE$k START port $P jobs ${#JOBS[@]}: $(printf '%s ' "${JOBS[@]}" | grep -oE 'CKE:[^:]+' | tr '\n' ' ')"
    sleep 20
  done ;;
*) echo "unknown step $1"; exit 2 ;;
esac
