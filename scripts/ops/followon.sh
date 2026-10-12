# S-115 follow-on lanes: start one single-route lane (E15 a, J20 s0..s3, E15 b) per route whenever a GPU has fewer
# than CAP running lanes. usage: followon.sh BOX CAP NGPU PORT0 ROUTE... (no evaluator kills: lanes already run)
BOX=$1; CAP=$2; NGPU=$3; PORT=$4; shift 4; cd /workspace
E=carla_armE_aug1_hires/model_epoch_015.pth
for R in "$@"; do
  while :; do
    G=""
    for g in $(seq 0 $((NGPU - 1))); do
      n=$(pgrep -fa "[q]ueue_A9c.sh [0-9]+ $g " | wc -l)
      [ "$n" -lt "$CAP" ] && { G=$g; break; }
    done
    [ -n "$G" ] && break
    sleep 60
  done
  JOBS="CKL:a9_E15_r${BOX}a_$PORT:$E:$R CKL:j20_r${BOX}_$PORT:carla_armJ_ft_obst/model_epoch_020.pth:$R"
  for s in 1 2 3; do JOBS="$JOBS CKL:j${s}s20_r${BOX}_$PORT:carla_armJ_ft_obst_s$s/model_epoch_020.pth:$R"; done
  JOBS="$JOBS CKL:a9_E15_r${BOX}b_$PORT:$E:$R"
  echo "$(date) follow-on lane $PORT GPU $G route $R"
  echo "$(date) S-115 follow-on lane $PORT" >> lane_${BOX}b$PORT.log
  setsid nohup bash /workspace/MThesis/scripts/eval/queue_A9c.sh $PORT $G $JOBS >> lane_${BOX}b$PORT.log 2>&1 < /dev/null &
  PORT=$((PORT + 400))
  sleep 120
done
echo "$(date) follow-on: all started"
