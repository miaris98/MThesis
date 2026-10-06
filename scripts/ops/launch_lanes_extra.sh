# S-131: extra lanes on a box that already runs lanes: like launch_lanes_nokill.sh but WITHOUT the step that stops running evaluators (that script kills them despite its name).
# usage (on the box): bash launch_lanes_extra.sh BOXNAME LANEFILE   (lane file ports must not clash with the running lanes: use gen_lanes.py --port0 >= 9000; size VRAM_GB for the free memory, 5.5 GB per lane)
BOX=$1; LANES=$2; cd /workspace
. MThesis/scripts/lib/common.sh
while read -r b PORT GPU JOBS; do
  [ "$b" = "$BOX" ] || continue
  echo "$(date) S-106 lane $PORT" >> lane_${BOX}b$PORT.log
  setsid nohup bash /workspace/MThesis/scripts/eval/queue_A9c.sh $PORT $GPU $JOBS >> lane_${BOX}b$PORT.log 2>&1 < /dev/null &
  sleep 25
done < "$LANES"
sleep 90
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "lanes $(pgrep -fc '[q]ueue_A9c.sh') carla $(pgrep -fc '[C]arlaUE4-Linux-Shipping') eval $(pgrep -fc '[l]eaderboard_evaluator') watchdog $(pgrep -fc '[e]val_watchdog')"
