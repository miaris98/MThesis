#!/usr/bin/env bash
# TODO A84 drives (2026-11-17), run on an eval box: 4 lanes, each owning one fixed chunk of the 220 routes (/workspace/chunks8.txt, 8 interleaved
# chunks; box A = chunks 1-4, box B = chunks 5-8), so every model drives every route on the same host (paired per route, no host effect).
# Each lane (scripts/ops/a84_lane.sh) runs vf_clean (the control, seed 0), then a84_rb and a84_dg with the danger gate once their checkpoints
# carry a READY marker (written after the sha256 check of the copy).
# usage: bash scripts/ops/a84_lanes.sh A|B          status -> /workspace/a84/status, lane logs -> /workspace/evlogs/
set -uo pipefail
BOX=${1:?A or B}; OFF=0; [ "$BOX" = B ] && OFF=4
S=/workspace/a84/status; mkdir -p /workspace/a84 /workspace/evlogs
mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd /workspace/MThesis
[ "$(wc -l < /workspace/chunks8.txt)" -ge 8 ] || { mark "LANES_REFUSED no chunks8.txt"; exit 1; }
grep -q PROVISION_DONE /workspace/setup_carla.log || { mark "LANES_REFUSED provisioning not done"; exit 1; }
if pgrep -f "a84_lan[e].sh" > /dev/null; then mark "LANES_REFUSED lanes already running"; exit 1; fi
for j in 1 2 3 4; do
  (setsid nohup bash scripts/ops/a84_lane.sh $j $(( OFF + j )) > /workspace/evlogs/lane_$j.out 2>&1 < /dev/null &)
  sleep 20
done
mark "LANES_STARTED box $BOX chunks $((OFF + 1))-$((OFF + 4))"
