#!/bin/bash
# Box side (2026-10-26, generic form of snapshot_b100k_pair.sh): take a consistent snapshot of one run's resumable pair
# (checkpoint_latest.pt + replay_latest.npz). The trainer rewrites both every 2,500 updates, so both are copied into a fresh
# directory and the copy is kept only if neither file's mtime or size changed while it ran.
# usage (on the box): snapshot_run_pair.sh CKPT_DIR SNAP_ROOT   -> prints the snapshot directory on success, FAILED otherwise
R=$1; SNAP=$2
rm -rf "$SNAP"; mkdir -p "$SNAP"
for try in 1 2 3 4 5 6; do
  a=$(stat -c %Y-%s $R/checkpoint_latest.pt $R/replay_latest.npz | tr '\n' ' ')
  T=$SNAP/$(date +%s)_$try
  mkdir -p "$T"
  cp $R/checkpoint_latest.pt $R/replay_latest.npz "$T/"
  b=$(stat -c %Y-%s $R/checkpoint_latest.pt $R/replay_latest.npz | tr '\n' ' ')
  if [ "$a" = "$b" ]; then
    echo "$T"
    exit 0
  fi
  rm -rf "$T"
done
echo FAILED
exit 1
