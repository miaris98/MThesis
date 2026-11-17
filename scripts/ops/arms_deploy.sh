#!/usr/bin/env bash
# TODO A81-A83 drives (2026-11-14): ship the repo and the six evaluated checkpoints to a fresh eval box, then check every file's sha256.
# Inputs come from the E: save of 2026-11-13 (live_20261113_arms): vf_ctrl (the paired baseline) and the five arms' epoch 30 + frozen backbone.
# Per checkpoint directory: model_epoch_030.pth, frozen_backbone.pth, run_config.json (the eval refuses without the frozen backbone).
# usage: scripts/ops/arms_deploy.sh PORT HOST        (direct pair from the instance record, e.g. 31752 ssh9.vast.ai)
set -uo pipefail
PORT=$1; HOST=$2; R="root@$HOST"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $R"; SCP="scp -P $PORT -o StrictHostKeyChecking=accept-new"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP=/e/MThesis_EXP/live_20261113_arms
LANES=$EXP/lanes220.txt
XML=$REPO/Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml
ARMS="vf_ctrl arm_s4 arm_w2 arm_w4 arm_ea arm_wr"
FILES="model_epoch_030.pth frozen_backbone.pth run_config.json"
for a in $ARMS; do for f in $FILES; do [ -f "$EXP/checkpoints/$a/$f" ] || { echo "MISSING on the PC: $EXP/checkpoints/$a/$f"; exit 1; }; done; done
[ -f "$LANES" ] || { echo "MISSING on the PC: $LANES"; exit 1; }
[ -f "$XML" ] || { echo "MISSING on the PC: $XML"; exit 1; }
TMP=$(mktemp -d)
tar -C "$REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP/mthesis_code.tgz" src scripts tests $(cd "$REPO" && ls *.txt *.toml *.cfg 2>/dev/null)
$SSH "mkdir -p /workspace/MThesis /workspace/vf /workspace/arms /workspace/evlogs /workspace/logs_prov" || exit 1
for a in $ARMS; do $SSH "mkdir -p /workspace/checkpoints/$a" || exit 1; done
$SCP "$TMP/mthesis_code.tgz" "$R:/workspace/mthesis_code.tgz" && $SSH "tar -xzf /workspace/mthesis_code.tgz -C /workspace/MThesis" || exit 1
$SCP "$LANES" "$R:/workspace/lanes220.txt" || exit 1
$SCP "$XML" "$R:/workspace/vf/bench2drive220.xml" || exit 1
for a in $ARMS; do
  for f in $FILES; do $SCP "$EXP/checkpoints/$a/$f" "$R:/workspace/checkpoints/$a/$f" || exit 1; done
done
bad=0; n=0
check() {  # local remote
  local x y; x=$(sha256sum "$1" | cut -c1-64); y=$($SSH "sha256sum '$2'" 2>/dev/null | cut -c1-64); n=$((n + 1))
  if [ "$x" = "$y" ]; then echo "OK   $2"; else echo "BAD  $2"; bad=$((bad + 1)); fi
}
check "$TMP/mthesis_code.tgz" /workspace/mthesis_code.tgz
check "$LANES" /workspace/lanes220.txt
check "$XML" /workspace/vf/bench2drive220.xml
for a in $ARMS; do for f in $FILES; do check "$EXP/checkpoints/$a/$f" "/workspace/checkpoints/$a/$f"; done; done
rm -rf "$TMP"
echo "DEPLOY_VERIFY files $n bad $bad"; [ $bad -eq 0 ]
