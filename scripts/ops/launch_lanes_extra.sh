# S-131: start one queue_A9c.sh lane per LANEFILE line for BOX (lane file from gen_lanes.py: `BOX PORT GPU JOBS...`).
# usage (on the box, or stdin to `bash -s`): bash launch_lanes_extra.sh [--kill] BOXNAME LANEFILE
#   default  extra lanes next to running ones: nothing is stopped. Lane ports must not clash with the running lanes (gen_lanes.py
#            --port0 >= 9000) and VRAM_GB must be sized for the free memory, 5.5 GB per lane.
#   --kill   first stop every running evaluator (kill_tree), then start the lanes: a full relaunch. This was launch_lanes_nokill.sh,
#            which killed evaluators despite its name (merged 2026-11-22).
KILL=0; [ "${1:-}" = --kill ] && { KILL=1; shift; }
BOX=$1; LANES=$2
. "${MTHESIS_ROOT:-${WORKSPACE:-/workspace}/MThesis}/scripts/lib/common.sh"
cd "$WORKSPACE" || exit 1
if [ $KILL = 1 ]; then
  for P in $(pgrep -f "[l]eaderboard_evaluator"); do kill_tree $P; done
  sleep 2
fi
while read -r b PORT GPU JOBS; do
  [ "$b" = "$BOX" ] || continue
  echo "$(date) S-106 lane $PORT" >> lane_${BOX}b$PORT.log
  setsid nohup bash "$MTHESIS_ROOT/scripts/eval/queue_A9c.sh" $PORT $GPU $JOBS >> lane_${BOX}b$PORT.log 2>&1 < /dev/null &
  sleep 25
done < "$LANES"
sleep 90
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "lanes $(pgrep -fc '[q]ueue_A9c.sh') carla $(pgrep -fc '[C]arlaUE4-Linux-Shipping') eval $(pgrep -fc '[l]eaderboard_evaluator') watchdog $(pgrep -fc '[e]val_watchdog')"
