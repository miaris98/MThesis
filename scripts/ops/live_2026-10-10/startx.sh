cd /workspace; BOX=$1; LANES=$2
while read -r b PORT GPU JOBS; do
  [ "$b" = "$BOX" ] || continue
  echo "$(date) extra lane $PORT" >> lane_${BOX}b$PORT.log
  setsid nohup bash /workspace/MThesis/scripts/eval/queue_A9c.sh $PORT $GPU $JOBS >> lane_${BOX}b$PORT.log 2>&1 < /dev/null &
  sleep 25
done < "$LANES"
sleep 60; echo "lanes $(pgrep -fc '[q]ueue_A9c.sh') carla $(pgrep -fc '[C]arlaUE4-Linux-Shipping') eval $(pgrep -fc '[l]eaderboard_evaluator')"
