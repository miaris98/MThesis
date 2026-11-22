#!/usr/bin/env bash
# Ship the repo working tree (uncommitted changes included) and a manifest of files to a box, checking every file's sha256 on both
# ends (2026-11-20; the general form of the per-experiment deploy scripts, whose manifests are now in scripts/ops/manifests/).
# Run on the PC (Git Bash).
# usage: scripts/ops/box_deploy.sh PORT HOST MANIFEST
#        scripts/ops/box_deploy.sh --check MANIFEST      expand the manifest and check every local file exists; ships nothing
#   MANIFEST, one entry per line (blank lines and # comments skipped):
#     LOCAL|REMOTE          copy a file (LOCAL as a Git Bash path /c/... or /e/..., never C:/..., which scp reads as a host)
#     CP REMOTE1 REMOTE2    copy a file already on the box (e.g. one frozen backbone into a second checkpoint dir), hashes compared
#     READY REMOTE_DIR      touch REMOTE_DIR/READY when every entry so far verified (box chains wait for these markers)
#   ${NAME} in an entry is replaced by that variable; an unset one stops the deploy before anything is shipped. Always set:
#     ${EXP}   the experiment archive as a Git Bash path (MTHESIS_EXP_ROOT, scripts/lib/pc.sh)
#     ${REPO}  this repository          ${RWS}  the box workspace (REMOTE_WORKSPACE, default $RWS)
#   Others come from the environment, e.g. ${SCRATCH} for files prepared in a session folder.
RWS=${REMOTE_WORKSPACE:-/workspace}   # the boxes' workspace (the Vast layout); REMOTE_WORKSPACE overrides
set -uo pipefail
CHECK=0; [ "${1:-}" = --check ] && { CHECK=1; shift; set -- - - "$1"; }
PORT=$1; HOST=$2; MAN=$3; R="root@$HOST"
SSH="ssh -n -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes $R"
SCP="scp -q -P $PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
. "$REPO/scripts/lib/pc.sh"   # EXP
RWS=${REMOTE_WORKSPACE:-/workspace}
bad=0; n=0
sha() { sha256sum "$1" | sed 's/^\\//' | cut -c1-64; }
rsha() { $SSH "sha256sum '$1'" 2>/dev/null | cut -c1-64; }
verdict() {  # want got remote
  n=$((n + 1))
  if [ -n "$1" ] && [ "$1" = "$2" ]; then echo "OK   $3"; else echo "BAD  $3"; bad=$((bad + 1)); fi
}
expand() {  # ${NAME} -> its value; an unset NAME is an error (a manifest never ships from a guessed path)
  local s=$1 name i=0
  while [[ $s =~ \$\{([A-Za-z_][A-Za-z0-9_]*)\} ]] && [ $i -lt 20 ]; do
    name=${BASH_REMATCH[1]}; i=$((i + 1))
    [ -n "${!name+x}" ] || { echo "manifest line uses \${$name}, which is not set: $1" >&2; return 1; }
    s=${s//"\${$name}"/${!name}}
  done
  printf '%s' "$s"
}
[ -f "$MAN" ] || { echo "no manifest $MAN"; exit 1; }
nloc=0
while IFS= read -r line || [ -n "$line" ]; do   # every local file first: a missing one stops before anything is shipped
  line=${line%$'\r'}
  case "$line" in ''|\#*|READY\ *|CP\ *) continue ;; esac
  line=$(expand "$line") || exit 1
  [ -f "${line%%|*}" ] || { echo "MISSING on the PC: ${line%%|*} (is E: connected?)"; exit 1; }
  nloc=$((nloc + 1))
done < "$MAN"
[ $CHECK = 1 ] && { echo "MANIFEST_OK $MAN: $nloc local files present"; exit 0; }
TMP=$(mktemp -d)
# carla_garage/team_code/*.py (2026-11-21): the TF++ camera-only trainer (src/baselines/tfpp_cam.py) builds TF++'s model from the repo copy
tar -C "$REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP/mthesis_code.tgz" src scripts tests $(cd "$REPO" && ls *.txt *.toml *.cfg 2>/dev/null) \
  $(cd "$REPO" && ls Carla-utils/carla_garage/team_code/*.py 2>/dev/null)
$SSH "mkdir -p $RWS/MThesis $RWS/ladder $RWS/evlogs $RWS/checkpoints" || exit 1
$SCP "$TMP/mthesis_code.tgz" "$R:$RWS/mthesis_code.tgz" && $SSH "tar -xzf $RWS/mthesis_code.tgz -C $RWS/MThesis" || exit 1
verdict "$(sha "$TMP/mthesis_code.tgz")" "$(rsha $RWS/mthesis_code.tgz)" $RWS/mthesis_code.tgz
rm -rf "$TMP"
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%$'\r'}
  case "$line" in ''|\#*) continue ;; esac
  line=$(expand "$line") || exit 1
  case "$line" in
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
