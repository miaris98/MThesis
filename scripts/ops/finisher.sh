# S-115 unattended finisher: while jobs matching PATTERN run, push the result PATHS to the HF relay every hour; once
# they are gone, push a final time and write FINISHED. Pushes are sha256-verified (push_verify.py, token in
# /dev/shm/hf_token). usage: finisher.sh HF_SUBDIR PATTERN PATH...   (PATHS relative to /workspace, globs allowed;
# PATTERN must use the [x]yz bracket form so it does not match this script's own command line)
SUB=$1; PAT=$2; shift 2; PATHS="$*"
cd /workspace
LOG=/workspace/finisher.log
push() {
  rm -rf /workspace/stage_fin && mkdir -p /workspace/stage_fin
  tar -chf - --exclude='*replay*' --exclude='*.tmp*' $PATHS 2>/dev/null | tar -xf - -C /workspace/stage_fin
  echo "$(date -u) push $1: $(find /workspace/stage_fin -type f | wc -l) files, $(du -sh /workspace/stage_fin | cut -f1)" >> $LOG
  for try in 1 2 3; do
    /venv/main/bin/python /workspace/push_verify.py /workspace/stage_fin $SUB 2>&1 | grep -a "PUSH_VERIFY\|BAD\|retry" >> $LOG
    tail -1 $LOG | grep -q "bad 0" && return 0
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
