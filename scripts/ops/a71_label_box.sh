#!/usr/bin/env bash
# A71 label box (2026-11-01): the 8 E15 towns, qcf_v4 labels on all cores (workers capped by RAM), ablations, then a tar of the label files.
set -uo pipefail
mkdir -p /workspace/a71; S=/workspace/a71/status
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis; export PYTHONPATH=/workspace/MThesis MLFLOW_ALLOW_FILE_STORE=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/venv/main/bin/python
D=/workspace/dataset/wor_trajectories
DENSE="Accident|Obstacle|HazardAtSideLane|OpensDoor|Construction|InvadingTurn|ParkedObstacle"
mark START
# src.training's __init__ imports the trainer -> logger -> torch.utils.tensorboard: the same packages as the box driver's setup (2026-11-02)
$PY -m pip install -q requests huggingface_hub mlflow tensorboard timm opencv-python-headless > /workspace/a71/pip.log 2>&1
grep -q DATA_DONE $S || { $PY scripts/setup/download_pdm_lite.py --towns Town01,Town02,Town03,Town04,Town05,Town10,Town12,Town13 --include-huge \
  --reserve-gb 30 --dest $D > /workspace/a71/data.log 2>&1 && mark DATA_DONE || { mark DATA_FAILED; exit 1; }; }
$PY -c "import sys; sys.path.insert(0, 'scripts/training'); import build_q_labels" > /workspace/a71/import_check.log 2>&1 \
  || { mark "IMPORT_FAILED $(tail -1 /workspace/a71/import_check.log)"; exit 1; }
CPUS=$($PY -c "import sys; sys.path.insert(0,'scripts/training'); from build_q_labels import _usable_cpus; print(_usable_cpus())")
RAMW=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 450000 ))          # ~450 MB per worker
W=$(( CPUS < RAMW ? CPUS : RAMW )); mark "WORKERS $W (cpus $CPUS, ram cap $RAMW)"
$PY scripts/training/build_q_labels.py --data_dir $D --tag qcf_v4 --stride 4 --dense_regex "$DENSE" --dense_stride 2 --workers $W \
  --summary /workspace/a71/qlabels_v4.json > /workspace/a71/qlabels_v4.log 2>&1
mark "LABELS_EXIT $?"
mkdir -p /workspace/dataset/abl/Town12/data
for sc in ParkedObstacleTwoWays HardBreakRoute; do ln -sfn $D/Town12/data/$sc /workspace/dataset/abl/Town12/data/$sc; done
for v in "hold8:--set hold=8" "osm0:--set overspeed_margin=0" "tail0:--set tail_s=0.25" "v4:"; do
  tag=abl_${v%%:*}; $PY scripts/training/build_q_labels.py --data_dir /workspace/dataset/abl --tag $tag ${v#*:} --stride 2 --workers $W \
    --summary /workspace/a71/$tag.json > /workspace/a71/$tag.log 2>&1; mark "ABL_${tag}_EXIT $?"
done
cd $D && find . -name qcf_v4.npz | tar czf /workspace/a71/qcf_v4_labels.tgz -T - && sha256sum /workspace/a71/qcf_v4_labels.tgz > /workspace/a71/qcf_v4_labels.sha
mark "TAR_DONE $(find . -name qcf_v4.npz | wc -l) files"
