#!/usr/bin/env bash
# TODO A84 (2026-11-17), run on the training box: the whole training side in order, each step only after the one before succeeded.
#   data (parallel per-town downloaders finish, none FAILED) -> audit -> labels + cache in parallel -> wait for E15 (READY marker after the
#   sha256 check of the chunked upload) -> train rb and dg side by side on the one GPU -> READY markers in each arm's checkpoint dir.
# Status: $WORKSPACE/a84/status (a84_box.sh marks) + CHAIN_* marks.
set -uo pipefail
CHAIN_NAME=a84
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"   # L, S, B, mark, await, step, ... (paths: common.sh)
until [ "$(ps aux | grep -c '[d]ownload_pdm_lite')" = 0 ]; do sleep 30; done
nf=$(grep -h FAILED $WORKSPACE/vf/data_*.log | wc -l)
[ "$nf" = 0 ] || chain_stop "data: $nf archives FAILED"
echo "$(date -u +%H:%M:%S) DATA_DONE" >> $WORKSPACE/vf/status
mark "CHAIN data done: $(du -sh $WORKSPACE/dataset | cut -f1), $(grep -h 'done in' $WORKSPACE/vf/data_*.log | wc -l) archives"
bash scripts/ops/a84_box.sh audit || chain_stop audit
wait_files $CHECKPOINTS/E15/READY          # the cache encoder is E15's frozen backbone when the TF++ zip is absent
( bash scripts/ops/a84_box.sh labels ) & LP=$!
( bash scripts/ops/a84_box.sh cache ) & CP=$!
wait $LP; rl=$?; wait $CP; rc=$?
[ $rl = 0 ] && [ $rc = 0 ] || chain_stop "labels rc=$rl cache rc=$rc"
( bash scripts/ops/a84_box.sh train rb && touch $CHECKPOINTS/a84_rb/TRAINED ) &
sleep 60
( bash scripts/ops/a84_box.sh train dg && touch $CHECKPOINTS/a84_dg/TRAINED ) &
wait
mark "CHAIN_DONE rb $( [ -f $CHECKPOINTS/a84_rb/TRAINED ] && echo ok || echo FAILED ) dg $( [ -f $CHECKPOINTS/a84_dg/TRAINED ] && echo ok || echo FAILED )"
