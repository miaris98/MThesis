#!/bin/bash
# TODO B100K backup (2026-10-25): every 30 min, copy each live 100k seed's resumable pair (checkpoint_latest.pt + replay_latest.npz)
# from the 5090 box to E:\MThesis_EXP\live_20261024_b100k\b100k_qnowd_s<seed>\checkpoints, sha256-verified against the box.
# The box takes a consistent snapshot first (snapshot_b100k_pair.sh), so the transfer copies a file that cannot change under it.
# usage: backup_b100k_pairs.sh SEEDS...      (env: BOX, PORT, DST; logs to $DST/save.log)
SO="-o StrictHostKeyChecking=no -o ConnectTimeout=40 -o BatchMode=yes"
BOX=${BOX:-root@75.91.40.84}
PORT=${PORT:-59186}
DST=${DST:-/e/MThesis_EXP/live_20261024_b100k}
LOG=$DST/save.log
CH=104857600
SNAP_SCRIPT=$(dirname "$0")/snapshot_b100k_pair.sh
mkdir -p $DST
pull() {  # box path, local path: 100 MiB chunks, each retried until its length is right; then whole-file sha256 check
  local SRC=$1 OUT=$2
  local SIZE=$(ssh $SO -p $PORT $BOX "stat -c %s $SRC")
  local TOTAL=$(( (SIZE + CH - 1) / CH )); [ "$TOTAL" -lt 1 ] && TOTAL=1
  : > $OUT.part
  for ((i=0; i<TOTAL; i++)); do
    local WANT=$(( SIZE - i*CH )); (( WANT > CH )) && WANT=$CH
    for try in 1 2 3 4 5 6; do
      ssh $SO -p $PORT $BOX "dd if=$SRC bs=1M skip=$((i*100)) count=100 2>/dev/null" > /tmp/b100k_chunk.bin
      [ "$(stat -c %s /tmp/b100k_chunk.bin)" -eq "$WANT" ] && break
    done
    cat /tmp/b100k_chunk.bin >> $OUT.part
  done
  local BOXSHA=$(ssh $SO -p $PORT $BOX "sha256sum $SRC" | cut -c1-64)
  local LOCSHA=$(sha256sum $OUT.part | cut -c1-64)
  if [ "$BOXSHA" = "$LOCSHA" ] && [ "$(stat -c %s $OUT.part)" -eq "$SIZE" ]; then
    return 0   # left as .part: the pair is only moved into place once both files of the snapshot have verified
  fi
  rm -f $OUT.part
  return 1
}
while true; do
  for s in $@; do
    OUTDIR=$DST/b100k_qnowd_s$s/checkpoints
    mkdir -p $OUTDIR
    ssh $SO -p $PORT $BOX "cat > /workspace/b100k/snapshot_b100k_pair.sh && chmod +x /workspace/b100k/snapshot_b100k_pair.sh" < "$SNAP_SCRIPT"
    SNAP=$(ssh $SO -p $PORT $BOX "bash /workspace/b100k/snapshot_b100k_pair.sh $s" | tr -d '\r' | tail -1)
    if [ -z "$SNAP" ] || [ "$SNAP" = "FAILED" ]; then
      echo "$(date -u +%H:%M) s$s no consistent snapshot this pass" >> $LOG
      continue
    fi
    ck=0; rp=0
    pull $SNAP/checkpoint_latest.pt $OUTDIR/checkpoint_latest.pt && ck=1
    pull $SNAP/replay_latest.npz $OUTDIR/replay_latest.npz && rp=1
    if [ $ck = 1 ] && [ $rp = 1 ]; then
      mv $OUTDIR/checkpoint_latest.pt.part $OUTDIR/checkpoint_latest.pt
      mv $OUTDIR/replay_latest.npz.part $OUTDIR/replay_latest.npz
      echo "$(date -u +%H:%M) s$s VERIFIED pair from snapshot $(basename $SNAP)" >> $LOG
      ssh $SO -p $PORT $BOX "for d in /workspace/b100k/snap/s$s/*; do [ \"\$d\" = \"$SNAP\" ] || rm -rf \"\$d\"; done"
    else
      rm -f $OUTDIR/checkpoint_latest.pt.part $OUTDIR/replay_latest.npz.part
      echo "$(date -u +%H:%M) s$s pair transfer FAILED from snapshot $(basename $SNAP) (checkpoint ok=$ck, replay ok=$rp), nothing replaced, will retry next pass" >> $LOG
    fi
  done
  sleep 1800
done
