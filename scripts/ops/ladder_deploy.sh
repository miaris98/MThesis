#!/usr/bin/env bash
# Teacher ladder deploy (2026-11-18), run on the PC (Git Bash): ship the repo working tree (uncommitted changes included), the route
# lists, the render test and the two checkpoints, check every file's sha256 on both ends, then mark each checkpoint dir READY.
# usage: scripts/ops/ladder_deploy.sh PORT HOST SCRATCH      (SCRATCH as a Git Bash path /c/..., never C:/... which scp reads as a host)
#   SCRATCH holds ids220.txt, fam_routes.txt, carla_render_test.sh and w8_teacher_local/ (model_epoch_030.pth, run_config.json, and
#   frozen_backbone.pth = E15's, tensor-identical)
# W8_SHIP=0 (R1 day, r1_chain.sh): E15 only, no w8_teacher.
set -uo pipefail
PORT=$1; HOST=$2; SCR=$3; R="root@$HOST"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $R"; SCP="scp -q -P $PORT -o StrictHostKeyChecking=accept-new"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
XML=$REPO/Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml
E15SRC=/e/MThesis_EXP/live_20261006_box0_train_3090ti/checkpoints/carla_armE_aug1_hires
W8=$SCR/w8_teacher_local
W8_E=/e/MThesis_EXP/live_20261108_spatial2/w8_teacher/model_epoch_030.pth
W8_SHIP=${W8_SHIP:-1}
bad=0; n=0
sha() { sha256sum "$1" | sed 's/^\\//' | cut -c1-64; }
check() {  # local remote
  local x y; x=$(sha "$1"); y=$($SSH "sha256sum '$2'" 2>/dev/null | cut -c1-64); n=$((n + 1))
  if [ "$x" = "$y" ]; then echo "OK   $2"; else echo "BAD  $2"; bad=$((bad + 1)); fi
}
NEED=("$XML" "$SCR/ids220.txt" "$SCR/fam_routes.txt" "$SCR/carla_render_test.sh" "$E15SRC/model_epoch_015.pth" "$E15SRC/frozen_backbone.pth" "$E15SRC/run_config.json")
[ "$W8_SHIP" = 1 ] && NEED+=("$W8/model_epoch_030.pth" "$W8/run_config.json" "$W8/frozen_backbone.pth")
for f in "${NEED[@]}"; do
  [ -f "$f" ] || { echo "MISSING on the PC: $f (is E: connected?)"; exit 1; }
done
[ "$W8_SHIP" = 0 ] || [ "$(sha "$W8/model_epoch_030.pth")" = "$(sha "$W8_E")" ] || { echo "w8_teacher scratch copy differs from E:"; exit 1; }
TMP=$(mktemp -d)
tar -C "$REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP/mthesis_code.tgz" src scripts tests $(cd "$REPO" && ls *.txt *.toml *.cfg 2>/dev/null)
$SSH "mkdir -p /workspace/MThesis /workspace/ladder /workspace/evlogs /workspace/checkpoints/E15 /workspace/checkpoints/w8_teacher" || exit 1
$SCP "$TMP/mthesis_code.tgz" "$R:/workspace/mthesis_code.tgz" && $SSH "tar -xzf /workspace/mthesis_code.tgz -C /workspace/MThesis" || exit 1
check "$TMP/mthesis_code.tgz" /workspace/mthesis_code.tgz
for pair in "$XML|/workspace/ladder/bench2drive220.xml" "$SCR/ids220.txt|/workspace/ladder/ids220.txt" "$SCR/fam_routes.txt|/workspace/ladder/fam_routes.txt" \
            "$SCR/carla_render_test.sh|/workspace/carla_render_test.sh"; do   # "|", not ":": a Windows path has a colon
  $SCP "${pair%%|*}" "$R:${pair#*|}" || exit 1; check "${pair%%|*}" "${pair#*|}"
done
SPECS=("E15|$E15SRC|model_epoch_015.pth frozen_backbone.pth run_config.json")
[ "$W8_SHIP" = 1 ] && SPECS+=("w8_teacher|$W8|model_epoch_030.pth run_config.json frozen_backbone.pth")
for spec in "${SPECS[@]}"; do
  NAME=${spec%%|*}; rest=${spec#*|}; SRC=${rest%%|*}; FILES=${rest#*|}; b0=$bad
  for f in $FILES; do $SCP "$SRC/$f" "$R:/workspace/checkpoints/$NAME/$f" || exit 1; check "$SRC/$f" "/workspace/checkpoints/$NAME/$f"; done
  [ $bad = $b0 ] && $SSH "touch /workspace/checkpoints/$NAME/READY" && echo "READY $NAME"
done
rm -rf "$TMP"
echo "DEPLOY_VERIFY files $n bad $bad"; [ $bad -eq 0 ]
