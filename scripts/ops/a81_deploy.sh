#!/usr/bin/env bash
# TODO A81 deploy (run on the PC, Git Bash): ship the repo working tree (uncommitted changes included) and the inputs a81_box.sh needs,
# then check every file's sha256 on the box against the PC copy.
# usage: scripts/ops/a81_deploy.sh PORT HOST        (direct pair from the instance record, e.g. 40411 38.65.239.50)
set -uo pipefail
PORT=$1; HOST=$2; R="root@$HOST"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new $R"; SCP="scp -P $PORT -o StrictHostKeyChecking=accept-new"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP=/e/MThesis_EXP
E15=$EXP/archive_scratch_df280e34_2026-11-11/e15_noopt/model_epoch_015.pth
VF=$EXP/live_20261104_visionfix/checkpoints/vf_ctrl
LANES=$EXP/archive_scratch_df280e34_2026-11-11/obst_lanes.txt
XML=$REPO/Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml
for f in "$E15" "$VF/model_epoch_030.pth" "$VF/frozen_backbone.pth" "$VF/run_config.json" "$LANES" "$XML"; do
  [ -f "$f" ] || { echo "MISSING on the PC: $f"; exit 1; }
done
TMP=$(mktemp -d)
tar -C "$REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP/mthesis_code.tgz" src scripts tests $(cd "$REPO" && ls *.txt *.toml *.cfg 2>/dev/null)
$SSH "mkdir -p /workspace/MThesis /workspace/checkpoints/E15 /workspace/checkpoints/vf_ctrl /workspace/vf /workspace/a81" || exit 1
$SCP "$TMP/mthesis_code.tgz" "$R:/workspace/mthesis_code.tgz" && $SSH "tar -xzf /workspace/mthesis_code.tgz -C /workspace/MThesis" || exit 1
$SCP "$E15" "$R:/workspace/checkpoints/E15/model_epoch_015.pth" || exit 1
$SCP "$VF/model_epoch_030.pth" "$VF/frozen_backbone.pth" "$VF/run_config.json" "$R:/workspace/checkpoints/vf_ctrl/" || exit 1
$SCP "$LANES" "$R:/workspace/obst_lanes.txt" && $SCP "$XML" "$R:/workspace/vf/bench2drive220.xml" || exit 1
bad=0
check() {  # local remote
  local a b; a=$(sha256sum "$1" | cut -c1-64); b=$($SSH "sha256sum '$2'" 2>/dev/null | cut -c1-64)
  if [ "$a" = "$b" ]; then echo "OK   $2"; else echo "BAD  $2"; bad=$((bad + 1)); fi
}
check "$TMP/mthesis_code.tgz" /workspace/mthesis_code.tgz
check "$E15" /workspace/checkpoints/E15/model_epoch_015.pth
for f in model_epoch_030.pth frozen_backbone.pth run_config.json; do check "$VF/$f" /workspace/checkpoints/vf_ctrl/$f; done
check "$LANES" /workspace/obst_lanes.txt
check "$XML" /workspace/vf/bench2drive220.xml
rm -rf "$TMP"
echo "DEPLOY_VERIFY bad $bad"; [ $bad -eq 0 ]
