# usage (stdin to bash -s): BOX LANEFILE - stop stray eval processes, then start one queue_A9c.sh lane per LANEFILE line for BOX
BOX=$1; LANES=$2; cd /workspace
. MThesis/scripts/lib/common.sh
for P in $(pgrep -f "[l]eaderboard_evaluator"); do kill_tree $P; done
sleep 2
while read -r b PORT GPU JOBS; do
  [ "$b" = "$BOX" ] || continue
  echo "$(date) S-106 lane $PORT" >> lane_${BOX}b$PORT.log
  setsid nohup bash /workspace/MThesis/scripts/eval/queue_A9c.sh $PORT $GPU $JOBS >> lane_${BOX}b$PORT.log 2>&1 < /dev/null &
  sleep 25
done < "$LANES"
sleep 90
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "lanes $(pgrep -fc '[q]ueue_A9c.sh') carla $(pgrep -fc '[C]arlaUE4-Linux-Shipping') eval $(pgrep -fc '[l]eaderboard_evaluator') watchdog $(pgrep -fc '[e]val_watchdog')"
