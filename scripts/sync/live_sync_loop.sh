#!/usr/bin/env bash
# Periodic off-box backup of live runs to the external archive (stage 1 of the two-stage sync).
#
# usage: scripts/sync/live_sync_loop.sh <ssh_port> <user@host> <dest_dir> [interval_s] [remote paths...]
#   remote paths are relative to the box's /workspace (default: bench2drive_out guardian_logs checkpoints
#   MThesis/results MThesis/mlruns *.log); globs are expanded on the box.
#   env: MARGIN (s, default 900), FULL_EVERY (passes, default 12; 0 = only the first pass is full).
#
# Each pass streams a tar into a staging directory next to <dest_dir> and merges it in only when the stream
# arrived complete. It used to extract straight into <dest_dir>, so a connection drop mid-stream replaced a
# complete older file with a truncated one. The remote tar may report "file changed as we read it" (exit 1);
# that archive is still valid, and the trainers write checkpoints atomically (tmp + os.replace), so *.tmp is
# excluded. replay_latest.npz (~1.4 GB each) is excluded from the periodic pass: copy it once when a run you
# may resume finishes (S-064), after checking the box's bandwidth price.
#
# After the first successful pass, a pass sends only files modified since the previous successful pass
# started, minus MARGIN (clock skew between the box and this machine, files written during that pass);
# every FULL_EVERY-th pass is full again. Every pass used to re-send every checkpoint, uncompressed.
PORT="$1"; HOST="$2"; DEST="$3"; INTERVAL="${4:-600}"; shift 4 2>/dev/null || shift $#
PATHS="${*:-bench2drive_out guardian_logs checkpoints MThesis/results MThesis/mlruns *.log}"
MARGIN="${MARGIN:-900}"; FULL_EVERY="${FULL_EVERY:-12}"
mkdir -p "$DEST"
SINCE=""; PASS=0
while true; do
  START=$(date +%s)
  NEWER=""
  if [ -n "$SINCE" ] && ! { [ "$FULL_EVERY" -gt 0 ] && [ $((PASS % FULL_EVERY)) -eq 0 ]; }; then
    NEWER="--newer-mtime=@$SINCE"
  fi
  PASS=$((PASS + 1))
  STAGE=$(mktemp -d "$DEST/.staging.XXXXXX")
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" \
    "cd /workspace && tar -cf - --exclude='replay_latest.npz' --exclude='*.tmp' --exclude='*.tmp.*' \
       --exclude='*.tmp[0-9]*' $NEWER \$(ls -d $PATHS 2>/dev/null) 2>/dev/null" | tar -xf - -C "$STAGE"
  rc=("${PIPESTATUS[@]}")
  if [ "${rc[0]}" -le 1 ] && [ "${rc[1]}" -eq 0 ] && [ -n "$(ls -A "$STAGE")" ]; then
    cp -a "$STAGE/." "$DEST/"
    SINCE=$((START - MARGIN))
    echo "$(date '+%F %T') synced ${NEWER:+(incremental) }-> $DEST ($(du -sh "$DEST" 2>/dev/null | cut -f1))"
  else
    echo "$(date '+%F %T') pass FAILED (ssh/remote tar exit ${rc[0]}, local tar exit ${rc[1]}) - $DEST left unchanged"
  fi
  rm -rf "$STAGE"
  sleep "$INTERVAL"
done
