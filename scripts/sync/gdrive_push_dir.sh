#!/usr/bin/env bash
# Push a folder to Google Drive under MThesis_backups/SUB and verify it by checksum (Drive stores MD5).
# usage: gdrive_push_dir.sh DIR SUB [FILES_FROM]
#   FILES_FROM: optional list of paths relative to DIR (one per line); only those are pushed and checked, so a box can push the
#   files the E: pull takes straight from /workspace without staging a second copy (fleet_monitor/sync_boxes.sh).
#   config: $GDRIVE_CONF or /dev/shm/gdrive_rclone.conf (shipped by scripts/ops/push_gdrive_token_to_boxes.sh)
# Prints one PUSH_VERIFY line (same shape as hf_push_dir.py); exit 0 only if every file matched.
DIR=$1; SUB=$2; LIST=$3
CONF=${GDRIVE_CONF:-/dev/shm/gdrive_rclone.conf}
[ -d "$DIR" ] && [ -n "$SUB" ] && [ -s "$CONF" ] || { echo "usage: gdrive_push_dir.sh DIR SUB [FILES_FROM] (and $CONF must exist)"; exit 2; }
DEST=gdrive:MThesis_backups/$SUB
RC="rclone --config $CONF --retries 4 --low-level-retries 10"
SEL=()
if [ -n "$LIST" ]; then
  [ -f "$LIST" ] || { echo "PUSH_VERIFY $SUB: files 0 bad unknown errors 1 (no list $LIST)"; exit 2; }
  CLEAN=$(mktemp)
  (cd "$DIR" && while IFS= read -r p; do p=${p#./}; [ -n "$p" ] && [ -f "$p" ] && printf '%s\n' "$p"; done < "$LIST" | sort -u) > "$CLEAN"
  SEL=(--files-from "$CLEAN")
  N=$(wc -l < "$CLEAN")
else
  N=$(find "$DIR" -type f | wc -l)
fi
$RC copy "$DIR" "$DEST" "${SEL[@]}" --checksum --transfers "${GDRIVE_TRANSFERS:-4}" --drive-chunk-size 128M --drive-upload-cutoff 128M --stats-one-line --stats 60s -v 2>&1 | grep -v "client_id"
OUT=$($RC check "$DIR" "$DEST" "${SEL[@]}" --one-way 2>&1)
[ -n "$LIST" ] && rm -f "$CLEAN"
BAD=$(printf '%s\n' "$OUT" | sed -n 's/.*: \([0-9]\+\) differences found.*/\1/p' | tail -1)
ERR=$(printf '%s\n' "$OUT" | sed -n 's/.*: \([0-9]\+\) errors while checking.*/\1/p' | tail -1)
BAD=${BAD:-unknown}; ERR=${ERR:-0}
echo "PUSH_VERIFY $SUB: files $N bad $BAD errors $ERR"
[ "$BAD" = 0 ] && [ "$ERR" = 0 ]
