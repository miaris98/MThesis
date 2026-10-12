#!/usr/bin/env bash
# usage: rebal.sh BOX PORT HOST
SP=/c/Users/miari/AppData/Local/Temp/claude/c--Users-miari-Desktop-MThesis/e60dfb75-7a23-4e4b-948b-4746ad102d87/scratchpad
B=$1; P=$2; H=$3; O="-o ConnectTimeout=25 -o StrictHostKeyChecking=no"
scp -q $O -P $P $SP/w_r_$B.txt root@$H:/workspace/w_r_$B.txt
ssh -n $O -p $P root@$H "cp /workspace/w_r_$B.txt /workspace/w_r_${B}_bak.txt" 
ssh $O -p $P root@$H "bash -s" < $SP/stop_lanes.sh
ssh $O -p $P root@$H "bash -s $B /workspace/w_r_$B.txt" < /c/Users/miari/Desktop/MThesis/scripts/ops/launch_lanes_nokill.sh
