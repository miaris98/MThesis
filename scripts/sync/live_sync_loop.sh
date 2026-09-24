#!/usr/bin/env bash
# Periodic off-box backup of live runs to the external archive (stage 1 of the two-stage sync).
#
# usage: scripts/sync/live_sync_loop.sh <ssh_port> <user@host> <dest_dir> [interval_s]
#
# Streams a tar of the Bench2Drive results, guardian logs, Gate 2 Atari run dirs and logs every
# interval. replay_latest.npz (1.4 GB each) is excluded from the periodic pass; copy it once
# after a run finishes if the next gate should resume with its buffer. Runs until killed.
PORT="$1"; HOST="$2"; DEST="$3"; INTERVAL="${4:-600}"
mkdir -p "$DEST"
while true; do
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" "$HOST" \
    "cd /workspace && tar -cf - --exclude='replay_latest.npz' --exclude='*.tmp' --exclude='*.tmp.npz' \
       bench2drive_out guardian_logs \
       \$(ls -d MThesis/results/100k_benchmark/S053* 2>/dev/null) \
       \$(ls S053*.log b2d20_setup.log install_vk.log 2>/dev/null) 2>/dev/null" \
    2>/dev/null | tar -xf - -C "$DEST" 2>/dev/null
  echo "$(date '+%F %T') synced -> $DEST ($(du -sh "$DEST" 2>/dev/null | cut -f1))"
  sleep "$INTERVAL"
done
