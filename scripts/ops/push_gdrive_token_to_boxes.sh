#!/usr/bin/env bash
# RUN THIS YOURSELF (Claude must not read .gdrive_rclone.conf): puts the Google Drive rclone config (it holds the refresh token)
# into /dev/shm/gdrive_rclone.conf on each box and installs rclone there if missing. Prints only byte counts and the rclone version.
# Exit 0 only if EVERY box took the file (rotate_gdrive_token.sh relies on that before it revokes the old token).
#
# usage:  scripts/ops/push_gdrive_token_to_boxes.sh                 # every running Vast.ai instance (vastai show instances)
#         scripts/ops/push_gdrive_token_to_boxes.sh PORT HOST ...   # explicit boxes, e.g. 15895 ssh3.vast.ai 21207 ssh4.vast.ai
# Rotation: scripts/ops/rotate_gdrive_token.sh  (docs/ops/google_drive_sync.md)
RCLONE_VERSION=v1.75.1
cd "$(dirname "$0")/../.." || exit 1
[ -s .gdrive_rclone.conf ] || { echo "missing .gdrive_rclone.conf in repo root"; exit 1; }

# Git Bash on this PC: `python`/`python3` are Microsoft Store stubs, `py` is real; vastai lives in the graphtools conda env.
PY=""; for c in python3 python py; do "$c" -c 'import json' >/dev/null 2>&1 && { PY=$c; break; }; done
[ -n "$PY" ] || { echo "no working python (tried python3, python, py)"; exit 1; }
VAST=${VAST_CLI:-$(command -v vastai || echo /c/Users/miari/anaconda3/envs/graphtools/Scripts/vastai.exe)}

if [ "$#" -gt 0 ]; then
  BOXES=()
  while [ "$#" -ge 2 ]; do BOXES+=("$1 $2"); shift 2; done
else
  mapfile -t BOXES < <("$VAST" show instances --raw 2>/dev/null | "$PY" -c "
import json, sys
for i in json.load(sys.stdin):
    if i.get('actual_status') == 'running' and i.get('ssh_host'):
        print(i['ssh_port'], i['ssh_host'])" | tr -d '\r')
fi
[ "${#BOXES[@]}" -gt 0 ] || { echo "no boxes (vastai: $VAST)"; exit 1; }

FAIL=0
for hp in "${BOXES[@]}"; do
  set -- $hp
  OUT=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p "$1" root@"$2" "umask 077; cat > /dev/shm/gdrive_rclone.conf; chmod 600 /dev/shm/gdrive_rclone.conf
    if ! command -v rclone >/dev/null; then
      cd /tmp && curl -fsSL -o rclone.zip https://downloads.rclone.org/$RCLONE_VERSION/rclone-$RCLONE_VERSION-linux-amd64.zip \
        && (unzip -oq rclone.zip || python3 -c \"import zipfile; zipfile.ZipFile('rclone.zip').extractall('.')\") \
        && install -m 755 rclone-$RCLONE_VERSION-linux-amd64/rclone /usr/local/bin/rclone
    fi
    echo \"\$(hostname): conf bytes \$(wc -c < /dev/shm/gdrive_rclone.conf), \$(rclone version 2>&1 | head -1)\"" \
    < .gdrive_rclone.conf 2>&1 | grep -v "Welcome\|Have fun\|AI agents\|Warning: Permanently")
  echo "$2:$1 $OUT"
  case "$OUT" in *"conf bytes"*"rclone v"*) ;; *) FAIL=$((FAIL+1)); echo "  ^ FAILED (box not updated)";; esac
done
echo "boxes ${#BOXES[@]} failed $FAIL"
[ "$FAIL" = 0 ]
