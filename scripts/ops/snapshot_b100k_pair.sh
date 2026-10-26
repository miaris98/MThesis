#!/bin/bash
# TODO B100K backup (2026-10-25): take a consistent snapshot of one seed's resumable pair on the 5090 box.
# The trainer rewrites checkpoint_latest.pt and replay_latest.npz every 2,500 updates (about every 9 min), and a 1 GB copy over the
# uplink takes longer, so copying them in place never verifies. Here both files are copied locally into a fresh directory, and the
# copy is kept only if neither file's mtime or size changed while it ran (so no save landed inside the copy window).
# usage (on the box): snapshot_b100k_pair.sh SEED   -> prints the snapshot directory on success, exit 1 otherwise
s=$1
R=/workspace/b100k/out/b100k_qnowd_s$s/checkpoints
# the previous pass's snapshot is no longer needed (the PC side keeps the last verified pair): clear it before taking a new one
rm -rf /workspace/b100k/snap/s$s
mkdir -p /workspace/b100k/snap/s$s
for try in 1 2 3 4 5 6; do
  a=$(stat -c %Y-%s $R/checkpoint_latest.pt $R/replay_latest.npz | tr '\n' ' ')
  T=/workspace/b100k/snap/s$s/$(date +%s)_$try
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
