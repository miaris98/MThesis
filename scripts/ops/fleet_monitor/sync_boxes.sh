#!/usr/bin/env bash
# Per box: pull the run's small files + checkpoint_latest (+ gzipped replay with REPLAY=1) to a transit dir and copy them to
# E:/MThesis_EXP/live_DAY_NAME (stage 1, every file size-checked); then stage 2 FROM THE BOX: the same files to Google Drive
# MThesis_backups/live_DAY/NAME (scripts/sync/gdrive_push_dir.sh, MD5-checked, token in /dev/shm/gdrive_rclone.conf).
# The Drive push runs in the background on the box, so a slow uplink cannot be cut off here: each sync prints the last finished
# push's PUSH_VERIFY line (a backup is on Drive only after "bad 0 errors 0") and starts the next one if none is running.
# usage: sync_boxes.sh [nostage2]   env: ONLY=carla|atari, STAGE2=drive|hf|none (default drive; hf = the old HF relay push),
#                                        HFATARI=0 (no stage 2 for the Atari boxes), REPLAY=1, FLEET_DAY, FLEET_TRANSIT
S="$(cd "$(dirname "$0")" && pwd)"; DAY=${FLEET_DAY:-$(date +%Y%m%d)}; export DAY
T="${FLEET_TRANSIT:-$HOME/fleet_transit}"; mkdir -p "$T"
F='Welcome\|Have fun\|AI agents\|Warning: Permanently'
O="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o BatchMode=yes"
S2=${STAGE2:-drive}; { [ "$1" = nohf ] || [ "$1" = nostage2 ]; } && S2=none
RD=MThesis/results/100k_benchmark/S058_ezv2_match
while IFS='|' read -r name kind port host label; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  [ -n "$ONLY" ] && [ "$kind" != "$ONLY" ] && continue
  if [ "$kind" = atari ]; then
    [ "${REPLAY:-0}" = 1 ] && ssh -n $O -p $port root@$host "cd /workspace/$RD/$label/checkpoints 2>/dev/null && [ -f replay_latest.npz ] && gzip -c replay_latest.npz > replay_latest.npz.gz.tmp && mv replay_latest.npz.gz.tmp replay_latest.npz.gz" 2>&1 | sed "/$F/d"
    LIST="cd /workspace; ls $RD/$label/checkpoints/checkpoint_latest.pt $RD/$label/checkpoints/checkpoint_best.pt $RD/$label/checkpoints/eval*.json $RD/$label/checkpoints/replay_latest.npz.gz $RD/$label/*.json $label.log ${label}_eval.log 2>/dev/null"
  else
    LIST="cd /workspace; { find bench2drive_out -maxdepth 1 \( -name '${label}*_x.json*' -o -name 'epic220*_x.json*' \); ls lane_*.log old_epic_logs/lane_*.log launch_*.out w_*.txt a9_exclude.txt guardian_logs/${label}*_x.log 2>/dev/null; find guardian_logs -name '${label}*attempt*.log' -size -3M; }"
  fi
  D=$T/$name; mkdir -p "$D"
  ssh -n $O -p $port root@$host "$LIST | tar -czf - -T -" 2>/dev/null | tar -xzf - -C "$D" 2>/dev/null
  # A44 collision clips and the pre-launch result snapshots (queue_A9c.sh, TODO A60): to E:, and to Drive (5 TiB) but not to the HF relay
  CLIPS="cd /workspace; find bench2drive_out/clips bench2drive_out/snapshots -type f -path 'bench2drive_out/*/${label}*' 2>/dev/null"
  S2LIST=$LIST; [ "$kind" = atari ] || S2LIST="$LIST; $CLIPS"
  [ "$kind" = atari ] || ssh -n $O -p $port root@$host "$CLIPS | tar -czf - -T -" 2>/dev/null | tar -xzf - -C "$D" 2>/dev/null
  echo "pulled $name: $(find "$D" -type f | wc -l) files $(du -sh "$D" | cut -f1)"
  S2B=$S2; [ "$kind" = atari ] && [ "${HFATARI:-1}" != 1 ] && S2B=none
  if [ "$S2B" = drive ]; then
    # [g]: the pattern must not match the remote shell that runs this pgrep
    scp -q $O -P $port "$S/../../sync/gdrive_push_dir.sh" root@$host:/workspace/gdrive_push_dir.sh 2>&1 | sed "/$F/d"
    ssh -n $O -p $port root@$host "cd /workspace; [ -s /dev/shm/gdrive_rclone.conf ] || { echo 'DRIVE: no token on box (run scripts/ops/push_gdrive_token_to_boxes.sh)'; exit 0; }
      command -v rclone >/dev/null || { echo 'DRIVE: no rclone on box (push_gdrive_token_to_boxes.sh installs it)'; exit 0; }
      L=/workspace/s2_$name.log; last=\$(grep -a PUSH_VERIFY \$L 2>/dev/null | tail -1)
      if pgrep -f '[g]drive_push_dir.sh /workspace live_$DAY/$name ' > /dev/null; then echo \"DRIVE: push still running; last finished: \${last:-none}\"; exit 0; fi
      [ -n \"\$last\" ] && echo \"DRIVE: last push (finished \$(date -u -r \$L '+%F %T') UTC): \$last\"
      ($S2LIST) > /workspace/s2_$name.list 2>/dev/null
      setsid nohup bash /workspace/gdrive_push_dir.sh /workspace live_$DAY/$name /workspace/s2_$name.list > \$L 2>&1 < /dev/null &
      echo \"DRIVE: push started (\$(wc -l < /workspace/s2_$name.list) files)\"" 2>&1 | sed "/$F/d" | sed "s/^/drive $name: /"
  elif [ "$S2B" = hf ]; then
    scp -q $O -P $port $S/hf_push_dir.py root@$host:/workspace/ 2>&1 | sed "/$F/d"
    timeout 420 ssh -n $O -p $port root@$host "cd /workspace; [ -s /dev/shm/hf_token ] || { echo 'HF: no token on box'; exit 0; }; mkdir -p hfstage/$name; ($LIST | tail -n +1) | tar -cf - -T - 2>/dev/null | tar -xf - -C hfstage/$name; PY=/venv/main/bin/python; [ -x \$PY ] || PY=python3; HF_TOKEN=\$(cat /dev/shm/hf_token) timeout 1500 \$PY hf_push_dir.py hfstage/$name live_$DAY/$name 2>&1 | tail -1" 2>&1 | sed "/$F/d"
  fi
done < "$S/boxes.txt"
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w $S/to_e.ps1)" -Transit "$(cygpath -w $T)" -Day "$DAY" 2>&1 | grep -E "^E:|ERROR"
