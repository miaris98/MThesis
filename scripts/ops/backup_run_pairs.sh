#!/bin/bash
# PC side (2026-10-26, generic form of backup_b100k_pairs.sh): every 30 min, copy each run's resumable pair (checkpoint_latest.pt +
# replay_latest.npz) from a box to E:, sha256-verified against a box-side snapshot (snapshot_run_pair.sh), so the copied files cannot
# change under the transfer. Both files are pulled to .part and moved into place only when both verify.
# usage: BOX=root@HOST PORT=N REMOTE_OUT=/workspace/b66/out DST=/e/MThesis_EXP/live_X backup_run_pairs.sh LABEL [LABEL ...]
SO="-o StrictHostKeyChecking=no -o ConnectTimeout=40 -o BatchMode=yes"
: "${BOX:?}" "${PORT:?}" "${REMOTE_OUT:?}" "${DST:?}"
LOG=$DST/save.log
CH=104857600
SNAP_SCRIPT=$(dirname "$0")/snapshot_run_pair.sh
mkdir -p $DST
pull() {  # box path, local path: 100 MiB chunks, each retried until its length is right; then whole-file sha256 check
  local SRC=$1 OUT=$2
  local SIZE=$(ssh $SO -p $PORT $BOX "stat -c %s $SRC")
  local TOTAL=$(( (SIZE + CH - 1) / CH )); [ "$TOTAL" -lt 1 ] && TOTAL=1
  : > $OUT.part
  for ((i=0; i<TOTAL; i++)); do
    local WANT=$(( SIZE - i*CH )); (( WANT > CH )) && WANT=$CH
    local ok=0
    for try in 1 2 3 4 5 6 7 8; do
      ssh $SO -p $PORT $BOX "dd if=$SRC bs=1M skip=$((i*100)) count=100 2>/dev/null" > $OUT.chunk
      [ "$(stat -c %s $OUT.chunk)" -eq "$WANT" ] && { ok=1; break; }
    done
    [ $ok = 1 ] || { rm -f $OUT.part $OUT.chunk; return 1; }
    cat $OUT.chunk >> $OUT.part
  done
  rm -f $OUT.chunk
  local BOXSHA=$(ssh $SO -p $PORT $BOX "sha256sum $SRC" | cut -c1-64)
  local LOCSHA=$(sha256sum $OUT.part | cut -c1-64)
  [ "$BOXSHA" = "$LOCSHA" ] && [ "$(stat -c %s $OUT.part)" -eq "$SIZE" ] && return 0
  rm -f $OUT.part
  return 1
}
ssh $SO -p $PORT $BOX "mkdir -p /workspace/bk && cat > /workspace/bk/snapshot_run_pair.sh && chmod +x /workspace/bk/snapshot_run_pair.sh" < "$SNAP_SCRIPT"
while true; do
  for label in "$@"; do
    OUTDIR=$DST/$label/checkpoints
    mkdir -p $OUTDIR
    if ! ssh $SO -p $PORT $BOX "test -f $REMOTE_OUT/$label/checkpoints/replay_latest.npz"; then
      echo "$(date -u +%H:%M) $label no pair on the box yet" >> $LOG
      continue
    fi
    SNAP=$(ssh $SO -p $PORT $BOX "bash /workspace/bk/snapshot_run_pair.sh $REMOTE_OUT/$label/checkpoints /workspace/bk/snap/$label" | tr -d '\r' | tail -1)
    if [ -z "$SNAP" ] || [ "$SNAP" = "FAILED" ]; then
      echo "$(date -u +%H:%M) $label no consistent snapshot this pass" >> $LOG
      continue
    fi
    ck=0; rp=0
    pull $SNAP/checkpoint_latest.pt $OUTDIR/checkpoint_latest.pt && ck=1
    pull $SNAP/replay_latest.npz $OUTDIR/replay_latest.npz && rp=1
    if [ $ck = 1 ] && [ $rp = 1 ]; then
      mv $OUTDIR/checkpoint_latest.pt.part $OUTDIR/checkpoint_latest.pt
      mv $OUTDIR/replay_latest.npz.part $OUTDIR/replay_latest.npz
      echo "$(date -u +%H:%M) $label VERIFIED pair from snapshot $(basename $SNAP)" >> $LOG
    else
      rm -f $OUTDIR/checkpoint_latest.pt.part $OUTDIR/replay_latest.npz.part
      echo "$(date -u +%H:%M) $label pair transfer FAILED (checkpoint ok=$ck, replay ok=$rp), nothing replaced, will retry next pass" >> $LOG
    fi
  done
  sleep 1800
done
