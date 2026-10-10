#!/usr/bin/env bash
# Backup tick 2026-10-10: per box, pull the run's small files + checkpoint_latest/best (+ gzipped replay when REPLAY=1) to the scratchpad transit dir, copy to
# E:\MThesis_EXP\live_20261010_<name> (every pulled file size-checked), then push to HF FROM THE BOX with its /dev/shm/hf_token (sha-checked; skipped if no token).
# usage: sync3.sh [nohf]      env: REPLAY=1 also gzip + pull replay_latest.npz for the Atari runs
S=/c/Users/miari/AppData/Local/Temp/claude/c--Users-miari-Desktop-MThesis/e60dfb75-7a23-4e4b-948b-4746ad102d87/scratchpad
T=$S/transit10; mkdir -p $T
F='Welcome\|Have fun\|AI agents\|Warning: Permanently'
O="-o ConnectTimeout=25 -o StrictHostKeyChecking=no -o BatchMode=yes"
NOHF=0; [ "$1" = nohf ] && NOHF=1
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
  D=$T/$name; rm -rf "$D"; mkdir -p "$D"
  ssh -n $O -p $port root@$host "$LIST | tar -czf - -T -" 2>/dev/null | tar -xzf - -C "$D" 2>/dev/null
  echo "pulled $name: $(find "$D" -type f | wc -l) files $(du -sh "$D" | cut -f1)"
  DOHF=1; [ "$NOHF" = 1 ] && DOHF=0; [ "$kind" = atari ] && [ "${HFATARI:-1}" != 1 ] && DOHF=0
  if [ "$DOHF" = 1 ]; then
    scp -q $O -P $port $S/hf_push_dir.py root@$host:/workspace/ 2>&1 | sed "/$F/d"
    timeout 420 ssh -n $O -p $port root@$host "cd /workspace; [ -s /dev/shm/hf_token ] || { echo 'HF: no token on box'; exit 0; }; rm -rf hfstage/$name; mkdir -p hfstage/$name; ($LIST | tail -n +1) | tar -cf - -T - 2>/dev/null | tar -xf - -C hfstage/$name; PY=/venv/main/bin/python; [ -x \$PY ] || PY=python3; HF_TOKEN=\$(cat /dev/shm/hf_token) timeout 1500 \$PY hf_push_dir.py hfstage/$name live_20261010/$name 2>&1 | tail -1" 2>&1 | sed "/$F/d"
  fi
done < "$S/boxes.txt"
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w $S/to_e_10.ps1)" 2>&1 | grep -E "^E:|ERROR"
