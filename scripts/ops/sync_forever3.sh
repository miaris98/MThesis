#!/usr/bin/env bash
# sync_forever2.sh + modification time in the comparison (S-111): a file that is rewritten at the same size
# (checkpoint_latest.pt) is copied again. mtime is compared in whole epoch seconds (tar -x keeps it).
PORT=$1; HOST=$2; D=$3; PATHS=$4; INT=${5:-300}
mkdir -p "$D"
while true; do
  R=$(mktemp); L=$(mktemp); M=$(mktemp)
  timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=20 -p $PORT root@$HOST \
    "cd /workspace && find $PATHS -type f ! -name replay_latest.npz ! -path '*/wandb/*' ! -name '*.tmp*' -printf '%p\t%s\t%T@\n' 2>/dev/null" \
    2>>"$D/.sync_err.log" | grep -v -E "^Welcome|^Have fun|^AI agents" | awk -F'\t' '{split($3,t,"."); print $1"\t"$2"\t"t[1]}' | sort > "$R"
  NR=$(wc -l < "$R")
  if [ "$NR" -eq 0 ]; then
    echo "$(date '+%F %T') ERROR empty remote listing (box may be gone) - nothing verified" >> "$D/.sync_status.log"
  else
    (cd "$D" && find . -type f -printf '%P\t%s\t%T@\n' 2>/dev/null | awk -F'\t' '{split($3,t,"."); print $1"\t"$2"\t"t[1]}' | sort) > "$L"
    comm -23 "$R" "$L" | cut -f1 > "$M"
    NM=$(wc -l < "$M")
    if [ "$NM" -gt 0 ]; then
      scp -q -o BatchMode=yes -P $PORT "$M" root@$HOST:/tmp/sync_list_$$.txt 2>>"$D/.sync_err.log"
      timeout 1500 ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p $PORT root@$HOST "cd /workspace && tar -cf - -T /tmp/sync_list_$$.txt" 2>>"$D/.sync_err.log" | tar -xf - -C "$D" 2>>"$D/.sync_err.log"
    fi
    (cd "$D" && find . -type f -printf '%P\t%s\t%T@\n' 2>/dev/null | awk -F'\t' '{split($3,t,"."); print $1"\t"$2"\t"t[1]}' | sort) > "$L"
    LEFT=$(comm -23 "$R" "$L" | wc -l)
    echo "$(date '+%F %T') remote_files=$NR copied=$NM still_missing=$LEFT size=$(du -sh "$D" 2>/dev/null | cut -f1)" >> "$D/.sync_status.log"
  fi
  rm -f "$R" "$L" "$M"
  sleep $INT
done
