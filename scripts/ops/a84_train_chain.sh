#!/usr/bin/env bash
# TODO A84 (2026-11-17), run on the training box: the whole training side in order, each step only after the one before succeeded.
#   data (parallel per-town downloaders finish, none FAILED) -> audit -> labels + cache in parallel -> wait for E15 (READY marker after the
#   sha256 check of the chunked upload) -> train rb and dg side by side on the one GPU -> READY markers in each arm's checkpoint dir.
# Status: /workspace/a84/status (a84_box.sh marks) + CHAIN_* marks.
set -uo pipefail
S=/workspace/a84/status; mkdir -p /workspace/a84
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
until [ "$(ps aux | grep -c '[d]ownload_pdm_lite')" = 0 ]; do sleep 30; done
nf=$(grep -h FAILED /workspace/vf/data_*.log | wc -l)
[ "$nf" = 0 ] || { mark "CHAIN_STOP data: $nf archives FAILED"; exit 1; }
echo "$(date -u +%H:%M:%S) DATA_DONE" >> /workspace/vf/status
mark "CHAIN data done: $(du -sh /workspace/dataset | cut -f1), $(grep -h 'done in' /workspace/vf/data_*.log | wc -l) archives"
bash scripts/ops/a84_box.sh audit || { mark "CHAIN_STOP audit"; exit 1; }
until [ -f /workspace/checkpoints/E15/READY ]; do sleep 30; done          # the cache encoder is E15's frozen backbone when the TF++ zip is absent
( bash scripts/ops/a84_box.sh labels ) & L=$!
( bash scripts/ops/a84_box.sh cache ) & C=$!
wait $L; rl=$?; wait $C; rc=$?
[ $rl = 0 ] && [ $rc = 0 ] || { mark "CHAIN_STOP labels rc=$rl cache rc=$rc"; exit 1; }
( bash scripts/ops/a84_box.sh train rb && touch /workspace/checkpoints/a84_rb/TRAINED ) &
sleep 60
( bash scripts/ops/a84_box.sh train dg && touch /workspace/checkpoints/a84_dg/TRAINED ) &
wait
mark "CHAIN_DONE rb $( [ -f /workspace/checkpoints/a84_rb/TRAINED ] && echo ok || echo FAILED ) dg $( [ -f /workspace/checkpoints/a84_dg/TRAINED ] && echo ok || echo FAILED )"
