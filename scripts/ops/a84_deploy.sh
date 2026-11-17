#!/usr/bin/env bash
# TODO A84 (2026-11-17): ship code and inputs to the A84 boxes, then check every file's sha256 on both ends.
#   code PORT HOST           the repo (src scripts tests), the 220-route lane lists and bench2drive220.xml (no E: needed)
#   ckpt PORT HOST NAME...   checkpoint dirs from E: -> /workspace/checkpoints/NAME (model_epoch_030.pth or _015, frozen_backbone.pth,
#                            run_config.json). Names: E15 (resume point), vf_clean vf_clean_s1 vf_clean_s2 vf_ctrl vf_s2d (2026-11-04 vision fix)
#   push PORT HOST NAME      a trained A84 arm from one box to this PC's scratch dir and on to another box (use pull/put below)
# usage: scripts/ops/a84_deploy.sh code 31168 211.14.147.110
set -uo pipefail
MODE=$1; PORT=$2; HOST=$3; shift 3; R="root@$HOST"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $R"; SCP="scp -P $PORT -o StrictHostKeyChecking=accept-new"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
XML=$REPO/Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml
LANES=${A84_LANES:-/e/MThesis_EXP/live_20261113_arms/lanes220.txt}
VF=/e/MThesis_EXP/live_20261104_visionfix
E15SRC=/e/MThesis_EXP/live_20261006_box0_train_3090ti/checkpoints/carla_armE_aug1_hires
bad=0; n=0
check() {  # local remote
  # sed: GNU sha256sum escapes a path with backslashes (Windows temp dirs) and then prefixes the hash with '\'
  local x y; x=$(sha256sum "$1" | sed 's/^\\//' | cut -c1-64); y=$($SSH "sha256sum '$2'" 2>/dev/null | cut -c1-64); n=$((n + 1))
  if [ "$x" = "$y" ]; then echo "OK   $2"; else echo "BAD  $2"; bad=$((bad + 1)); fi
}
case $MODE in
code)
  [ -f "$XML" ] && [ -f "$LANES" ] || { echo "MISSING on the PC: $XML or $LANES"; exit 1; }
  TMP=$(mktemp -d)
  tar -C "$REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP/mthesis_code.tgz" src scripts tests $(cd "$REPO" && ls *.txt *.toml *.cfg 2>/dev/null)
  $SSH "mkdir -p /workspace/MThesis /workspace/vf /workspace/a84 /workspace/evlogs /workspace/checkpoints" || exit 1
  $SCP "$TMP/mthesis_code.tgz" "$R:/workspace/mthesis_code.tgz" && $SSH "tar -xzf /workspace/mthesis_code.tgz -C /workspace/MThesis" || exit 1
  $SCP "$LANES" "$R:/workspace/lanes220.txt" && $SCP "$XML" "$R:/workspace/vf/bench2drive220.xml" || exit 1
  check "$TMP/mthesis_code.tgz" /workspace/mthesis_code.tgz; check "$LANES" /workspace/lanes220.txt; check "$XML" /workspace/vf/bench2drive220.xml
  rm -rf "$TMP" ;;
ckpt)
  for NAME in "$@"; do
    if [ "$NAME" = E15 ]; then SRC=$E15SRC; EP=model_epoch_015.pth
    elif [ "$NAME" = vf_s2d ]; then SRC=$VF/vf_s2d/checkpoints; EP=model_epoch_030.pth
    elif [ "${NAME#a84_}" != "$NAME" ]; then SRC=/e/MThesis_EXP/live_20261117_a84/checkpoints/$NAME; EP=model_epoch_030.pth   # pulled by pull_a84.sh
    else SRC=$VF/checkpoints/$NAME; EP=model_epoch_030.pth; fi
    for f in $EP frozen_backbone.pth run_config.json; do [ -f "$SRC/$f" ] || { echo "MISSING on the PC: $SRC/$f (is E: connected?)"; exit 1; }; done
    $SSH "mkdir -p /workspace/checkpoints/$NAME" || exit 1
    for f in $EP frozen_backbone.pth run_config.json; do
      $SCP "$SRC/$f" "$R:/workspace/checkpoints/$NAME/$f" || exit 1
      check "$SRC/$f" "/workspace/checkpoints/$NAME/$f"
    done
  done ;;
*) echo "unknown mode $MODE"; exit 2 ;;
esac
echo "DEPLOY_VERIFY files $n bad $bad"; [ $bad -eq 0 ]
