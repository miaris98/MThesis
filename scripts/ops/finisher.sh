# S-115 unattended finisher: while jobs matching PATTERN run, push the result PATHS every hour; once they are gone, push a final
# time and write FINISHED. Stage 2 is Google Drive (scripts/sync/gdrive_push_dir.sh, MD5-checked, token in /dev/shm/gdrive_rclone.conf,
# shipped with rclone by scripts/ops/push_gdrive_token_to_boxes.sh); FIN_TARGET=hf uses the old HF relay push (push_verify.py, token
# in /dev/shm/hf_token). usage: finisher.sh SUBDIR PATTERN PATH...   (SUBDIR under MThesis_backups/ on Drive, or the HF relay subdir;
# PATHS relative to /workspace, globs allowed; PATTERN must use the [x]yz bracket form so it does not match this script's own command line)
SUB=$1; PAT=$2; shift 2; PATHS="$*"
cd /workspace
LOG=/workspace/finisher.log
GD=${GDRIVE_PUSH:-/workspace/MThesis/scripts/sync/gdrive_push_dir.sh}; [ -f "$GD" ] || GD=/workspace/gdrive_push_dir.sh
push() {
  local line try
  rm -rf /workspace/stage_fin && mkdir -p /workspace/stage_fin
  tar -chf - --exclude='*replay*' --exclude='*.tmp*' $PATHS 2>/dev/null | tar -xf - -C /workspace/stage_fin
  echo "$(date -u) push $1: $(find /workspace/stage_fin -type f | wc -l) files, $(du -sh /workspace/stage_fin | cut -f1)" >> $LOG
  for try in 1 2 3; do
    if [ "${FIN_TARGET:-drive}" = hf ]; then
      line=$(/venv/main/bin/python /workspace/push_verify.py /workspace/stage_fin $SUB 2>&1 | grep -a "PUSH_VERIFY\|BAD\|retry" | tee -a $LOG | tail -1)
      case "$line" in *PUSH_VERIFY*"bad 0"*) return 0 ;; esac
    else
      line=$(bash "$GD" /workspace/stage_fin "$SUB" 2>&1 | grep -a "PUSH_VERIFY\|usage" | tail -1)
      echo "$(date -u) ${line:-no PUSH_VERIFY line from $GD}" >> $LOG
      case "$line" in *PUSH_VERIFY*"bad 0 errors 0"*) return 0 ;; esac
    fi
    sleep 120
  done
  return 1
}
last=0  # first push one minute in
while pgrep -f "$PAT" > /dev/null; do
  sleep 60
  if [ $(( $(date +%s) - last )) -ge 3600 ]; then push periodic; last=$(date +%s); fi
done
if push final; then echo "$(date -u) FINISHED (final push verified)" >> $LOG; else echo "$(date -u) FINAL PUSH FAILED" >> $LOG; fi
