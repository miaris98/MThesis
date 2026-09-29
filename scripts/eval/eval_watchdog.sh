#!/usr/bin/env bash
# CARLA eval watchdog (boxes S, V, X2; 2026-09-26/27). Run on the eval box next to b2d_guardian.sh:
#   setsid nohup bash eval_watchdog.sh >> $WORKSPACE/eval_watchdog.log 2>&1 &
# A CARLA boot crash ("GameThread timed out waiting for RenderThread") or a mid-route CARLA segfault leaves
# leaderboard_evaluator.py sitting in 10-minute load_world time-outs, and b2d_guardian.sh only relaunches once the
# run script exits. When the newest attempt log of a running guardian ends in either, kill the whole process tree
# under that attempt's run script (found as the guardian's own child, never by a name pattern - S-077); CARLA runs
# under `su carlauser` and survives a plain kill, then blocks its ports for every later launch (S-091).
# Paths and the process-tree helpers come from scripts/lib/common.sh (one tested copy of desc/kill_tree; the copy
# that used to live here lacked `local` and never killed the evaluator, S-099).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
POLL="${POLL:-120}"
while true; do
  for G in $(pgrep -f "^bash $MTHESIS_ROOT/scripts/eval/b2d_guardian.sh "); do
    label=$(tr '\0' ' ' < /proc/$G/cmdline | awk '{print $3}')
    J=$B2D_OUT/${label}.json
    # keep the last parseable result file: a kill -9 mid-write left v2_armI_e15 truncated and the next attempt
    # restarted from route 1, overwriting 12 finished routes (2026-09-28)
    json_ok "$J" && cp -f "$J" "$J.bak"
    f=$(ls -t "$GUARDIAN_LOGS/${label}"_attempt*.log 2>/dev/null | head -1)
    [ -n "$f" ] || continue
    last=$(tail -c 2000 "$f" | tr '\r' '\n' | grep -v '^+' | grep -v '^\s*$' | tail -1)
    # also a CARLA segfault: the evaluator then waits out the same 10-min time-outs (WoR r2 on box X2, S-089).
    # Bash reports it as "...: line N: PID Segmentation fault (core dumped)", so match a substring, not the line.
    if [[ "$last" == *"time-out of 600000ms while waiting for the simulator"* || "$last" == *"Segmentation fault"* ]]; then
      for R in $(pgrep -P $G -f run_bench2drive.sh); do
        L=$(pgrep -P $R -f leaderboard_evaluator)
        log "$label: stuck (CARLA segfault or load_world time-out: $f) - killing evaluator $L and run script $R"
        kill_tree $R
        sleep 2
        if ! json_ok "$J" && [ -f "$J.bak" ]; then
          cp -f "$J.bak" "$J"; log "$label: result file unreadable after kill - restored $J.bak"
        fi
      done
      # an evaluator orphaned by an earlier incomplete kill (ppid 1, no run script above it) still matches the
      # guardian's `pgrep -f checkpoint=<json>`, so the guardian waits on it forever (S-099): kill it by that json
      if [ -z "$(pgrep -P $G -f run_bench2drive.sh)" ]; then
        for L in $(pgrep -f "leaderboard_evaluator.py.*checkpoint=$J( |$)"); do
          log "$label: stuck orphan evaluator $L (no run script) - killing its tree"
          kill_tree $L
        done
      fi
    fi
  done
  sleep "$POLL"
done
