#!/usr/bin/env bash
# Fleet check 2026-10-10: Atari 100k seeds (GTrXL, S058h recipe) + CARLA at Epic. One line per box: progress, pace, ETA, errors; then credit and burn.
# Box table is read from boxes.txt (name|kind|port|host|label), edited by hand when boxes change.
SP=/c/Users/miari/AppData/Local/Temp/claude/c--Users-miari-Desktop-MThesis/e60dfb75-7a23-4e4b-948b-4746ad102d87/scratchpad
F='Welcome\|Have fun\|AI agents\|Warning: Permanently'
O="-o ConnectTimeout=20 -o StrictHostKeyChecking=no -o BatchMode=yes"
echo "[$(date -u -d '+3 hours' +%H:%M) Athens]"
while IFS='|' read -r name kind port host label; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  if [ "$kind" = atari ]; then
    ssh -n $O -p $port root@$host "L=/workspace/\${LABEL}.log; L=/workspace/$label.log
      s=\$(grep '^env' \$L 2>/dev/null | tail -n1 | sed -E 's/^env +([0-9]+)\/100000 upd +([0-9]+).*train-life-ret\(20\) ([0-9.]+).*\| ([0-9.]+) min\$/\1 \3 \4/')
      set -- \$s; cur=\$1; ret=\$2; el=\$3
      first=\$(grep '^env' \$L 2>/dev/null | head -n1 | sed -E 's/^env +([0-9]+).*/\1/')
      ev=\$(grep -h '\[EVAL' /workspace/${label}_eval.log \$L 2>/dev/null | tail -n1 | sed -E 's/.*env([0-9]+) .*: ([0-9.]+) \+\/- ([0-9.]+) SE.*/env\1 \2±\3/')
      pace=\$(grep '^env' \$L 2>/dev/null | tail -n 4 | sed -E 's/^env +([0-9]+).*\| ([0-9.]+) min\$/\1 \2/' | awk 'NR==1{a=\$1;b=\$2} {c=\$1;d=\$2} END{ if(c>a) printf \"%.1f\", (d-b)/((c-a)/1000); else print \"?\"}')
      eta=\"?\"; [ -n \"\$cur\" ] && [ \"\$pace\" != \"?\" ] && eta=\$(awk -v c=\$cur -v p=\$pace 'BEGIN{printf \"%.1f\", (100000-c)/1000*p/60}')
      echo \"$name atari $label: env \${cur:-?}/100000 pace \${pace}min/1k ETA \${eta}h life-ret \${ret:-?} lastEval \${ev:-none} | tb \$(grep -c Traceback \$L /workspace/${label}_eval.log 2>/dev/null | paste -sd+ | bc 2>/dev/null) exits \$(grep -c 'exited rc=' \$L) | gpu \$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) load \$(cut -d' ' -f1 /proc/loadavg) | hf \$(tail -n1 /workspace/hfsync.log 2>/dev/null | cut -c1-60)\"" 2>&1 | sed "/$F/d" || echo "$name unreachable"
  else
    ssh -n $O -p $port root@$host "python3 - <<PY
import json,glob,re
n=0;ds=0;bad=0
for f in glob.glob('/workspace/bench2drive_out/$label*_x.json'):
    try: d=json.load(open(f))
    except Exception: continue
    for r in d.get('_checkpoint',{}).get('records',[]):
        n+=1; ds+=r['scores']['score_composed']; bad+=('set up' in r['status'])
print(f'$name carla $label: {n}/220 routes ({bad} INVALID agent-setup/OOM), mean DS {ds/max(n,1):.1f}')
PY
echo \"   lanes \$(pgrep -fc '[q]ueue_A9c.sh') carla \$(pgrep -fc '[C]arlaUE4-Linux-Shipping') eval \$(pgrep -fc '[l]eaderboard_evaluator') | crash-skips \$(cat /workspace/lane_*.log 2>/dev/null | grep -c 'CRASH ROUTE') | load \$(cut -d' ' -f1 /proc/loadavg) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ') | prov \$(cat /workspace/logs_prov/status 2>/dev/null | tr '\n' ' ')\"" 2>&1 | sed "/$F/d" || echo "$name unreachable"
  fi
done < "$SP/boxes.txt"
"/c/Users/miari/anaconda3/envs/graphtools/Scripts/vastai.exe" show instances --raw 2>/dev/null | py -c "
import json,sys
d=json.load(sys.stdin); print('boxes',len(d),'burn \$%.2f/h'%sum(i['dph_total'] for i in d), '| not running:', [ (i['id'],i['actual_status']) for i in d if i['actual_status']!='running'])"
"/c/Users/miari/anaconda3/envs/graphtools/Scripts/vastai.exe" show user --raw 2>/dev/null | py -c "
import json,sys; c=json.load(sys.stdin).get('credit',0); print('credit \$%.2f' % c)"
