#!/usr/bin/env bash
# Launch a remote job, then verify it is alive and has not crashed before reporting success (2026-10-16: two evals died on
# "No module named 'src'" right after launch and nobody saw it for 40 min). Use for every remote launch.
#   usage: launch_checked.sh HOST PORT LOG_PATH "CMD ..."   (CMD runs in /workspace/MThesis with PYTHONPATH set)
# Prints LAUNCH_OK <pid> or LAUNCH_FAILED with the last log lines. Exit 0 only on LAUNCH_OK.
set -u
H=$1; P=$2; LOG=$3; CMD=$4
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p $P"
ssh $SSHO root@$H "cd /workspace/MThesis && PYTHONPATH=/workspace/MThesis setsid nohup bash -c '$CMD' > '$LOG' 2>&1 < /dev/null & echo \$! > $LOG.pid; sleep 50; \
  PID=\$(cat $LOG.pid); ERR=\$(awk '/Traceback|Error|No module named/' '$LOG' | head -n 1); \
  if [ -n \"\$ERR\" ]; then echo LAUNCH_FAILED \"\$ERR\"; tail -n 25 '$LOG'; \
  elif kill -0 \$PID 2>/dev/null; then echo LAUNCH_OK \$PID; \
  else echo LAUNCH_ENDED_NO_ERROR job_finished_within_50s_check_log_tail; tail -n 5 '$LOG'; fi" 2>&1 | grep --line-buffered -v -E "Welcome|Have fun|AI agents|Warning: Permanently"
