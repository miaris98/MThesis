#!/usr/bin/env bash
# TODO A81-A83 drives (2026-11-14): lane 1's routes are the slowest (vf_ctrl on box 60611716: L1 3h51, L2 3h09, L3 2h18 per set), so
# lanes 3 and 2 would idle for hours while lane 1 drives its last sets. When lane 3's or lane 2's own queue has ended, start a queue on
# that port with lane 1's LAST jobs (wr, then ea) on lane 1's route list. queue_A9c.sh skips a label whose records are complete and
# attaches to a label whose guardian is running, so no set is driven twice and lane 1's queue needs no restart (it skips or attaches
# when it gets there). Same host, routes, checkpoints and flags: pairing is unchanged (a lane is only a CARLA port).
# The first lane to free up first reruns vf_ctrl on routes 3514 (Simulation crashed) and 3520 (never driven: queue_A9c's crash skip
# took the route after the crashed one, because the evaluator had already written a record for 3514).
# Run on the box:  cd $WORKSPACE; setsid nohup bash $WORKSPACE/MThesis/scripts/ops/arms_rebalance.sh > $WORKSPACE/arms/rebalance.out 2>&1 < /dev/null &
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
S=$WORKSPACE/arms/drive_status; mark() { echo "$(date -u +%H:%M:%S) $1" >> $S; }
cd $WORKSPACE/MThesis
F="CARLA_QUALITY=Epic"; L1=$(sed -n 1p $WORKSPACE/lanes220.txt)
J_RR="CKE:vf_ctrl_rr:vf_ctrl/model_epoch_030.pth:3514,3520:$F"
J_WR="CKE:arm_wr_L1:arm_wr/model_epoch_030.pth:$L1:$F"
J_EA="CKE:arm_ea_L1:arm_ea/model_epoch_030.pth:$L1:$F"
busy() { pgrep -f "[q]ueue_A9c.sh $1 " > /dev/null; }
mark "REBALANCE_WAITING for lanes 3 (2030) and 2 (2020) to finish their own sets"
started=""; n=0
while [ $n -lt 2 ]; do
  for P in 2030 2020; do
    case " $started " in *" $P "*) continue ;; esac
    busy $P && continue
    sleep 90; busy $P && continue                        # still free after 90 s: the lane's own queue has really ended
    if [ $n -eq 0 ]; then JOBS="$J_RR $J_WR $J_EA"; else JOBS="$J_WR $J_EA"; fi
    setsid nohup bash scripts/eval/queue_A9c.sh $P 0 $JOBS > $WORKSPACE/evlogs/rebalance_$P.out 2>&1 < /dev/null &
    mark "REBALANCE_LANE port $P jobs: $(echo $JOBS | grep -oE 'CKE:[a-z0-9_]+' | tr '\n' ' ')"
    started="$started $P"; n=$((n + 1))
  done
  sleep 60
done
mark "REBALANCE_ALL_STARTED"
