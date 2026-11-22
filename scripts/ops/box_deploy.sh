#!/usr/bin/env bash
# Ship the repo working tree (uncommitted changes included) and a manifest of files to a box, checking every file's sha256 on both
# ends (2026-11-20; the general form of ladder_deploy.sh). Run on the PC (Git Bash).
# usage: scripts/ops/box_deploy.sh PORT HOST MANIFEST
#   MANIFEST, one entry per line (blank lines and # comments skipped):
#     LOCAL|REMOTE          copy a file (LOCAL as a Git Bash path /c/... or /e/..., never C:/..., which scp reads as a host)
#     CP REMOTE1 REMOTE2    copy a file already on the box (e.g. one frozen backbone into a second checkpoint dir), hashes compared
#     READY REMOTE_DIR      touch REMOTE_DIR/READY when every entry so far verified (box chains wait for these markers)
set -uo pipefail
PORT=$1; HOST=$2; MAN=$3; R="root@$HOST"
SSH="ssh -n -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes $R"
SCP="scp -q -P $PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bad=0; n=0
sha() { sha256sum "$1" | sed 's/^\\//' | cut -c1-64; }
rsha() { $SSH "sha256sum '$1'" 2>/dev/null | cut -c1-64; }
verdict() {  # want got remote
  n=$((n + 1))
  if [ -n "$1" ] && [ "$1" = "$2" ]; then echo "OK   $3"; else echo "BAD  $3"; bad=$((bad + 1)); fi
}
[ -f "$MAN" ] || { echo "no manifest $MAN"; exit 1; }
while IFS= read -r line || [ -n "$line" ]; do   # every local file first: a missing one stops before anything is shipped
  case "$line" in ''|\#*|READY\ *|CP\ *) continue ;; esac
  [ -f "${line%%|*}" ] || { echo "MISSING on the PC: ${line%%|*} (is E: connected?)"; exit 1; }
done < "$MAN"
TMP=$(mktemp -d)
# carla_garage/team_code/*.py (2026-11-21): the TF++ camera-only trainer (src/baselines/tfpp_cam.py) builds TF++'s model from the repo copy
tar -C "$REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP/mthesis_code.tgz" src scripts tests $(cd "$REPO" && ls *.txt *.toml *.cfg 2>/dev/null) \
  $(cd "$REPO" && ls Carla-utils/carla_garage/team_code/*.py 2>/dev/null)
$SSH "mkdir -p /workspace/MThesis /workspace/ladder /workspace/evlogs /workspace/checkpoints" || exit 1
$SCP "$TMP/mthesis_code.tgz" "$R:/workspace/mthesis_code.tgz" && $SSH "tar -xzf /workspace/mthesis_code.tgz -C /workspace/MThesis" || exit 1
verdict "$(sha "$TMP/mthesis_code.tgz")" "$(rsha /workspace/mthesis_code.tgz)" /workspace/mthesis_code.tgz
rm -rf "$TMP"
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ''|\#*) ;;
    READY\ *) d=${line#READY }
      if [ $bad = 0 ]; then $SSH "touch '$d/READY'" && echo "READY $d"; else echo "NOT READY $d ($bad bad files)"; fi ;;
    CP\ *) set -- ${line#CP }
      $SSH "mkdir -p '$(dirname "$2")' && cp '$1' '$2'"
      verdict "$(rsha "$1")" "$(rsha "$2")" "$2" ;;
    *) src=${line%%|*}; dst=${line#*|}
      $SSH "mkdir -p '$(dirname "$dst")'" && $SCP "$src" "$R:$dst"
      verdict "$(sha "$src")" "$(rsha "$dst")" "$dst" ;;
  esac
done < "$MAN"
echo "DEPLOY_VERIFY files $n bad $bad"; [ $bad -eq 0 ]
