#!/usr/bin/env bash
# A74 relay (2026-11-02): one PPO learner on this PC (its state on E:), the CARLA lanes on any number of boxes. Every A74_RELAY_POLL s
# (45), for each box: pull its finished episodes (/workspace/a74/RUN/rollouts/*.npz) into LOCAL/rollouts as <box>_<file> (extracted into
# LOCAL/incoming first and then renamed, so the learner never reads a partial file), park the pulled ones in the box's rollouts/relayed/,
# and push LOCAL/learner/head_latest.pth whenever it changed (to a .tmp, then renamed: the agents reload it at every route start).
# Ends when LOCAL/STOP exists or LOCAL/learner.log has A74_LEARNER_DONE, and then stops the boxes' lanes (their STOP file).
# usage: scripts/ops/a74_relay.sh LOCAL RUN name:port:host [name:port:host ...]          log: LOCAL/relay.log
#   learner: py scripts/training/train_maneuver_ppo.py --rollouts LOCAL/rollouts --out LOCAL/learner ... >> LOCAL/learner.log
set -u
L=$1; RUN=$2; shift 2
R=/workspace/a74/$RUN
mkdir -p "$L/rollouts" "$L/learner" "$L/incoming"
declare -A last
log() { echo "$(date -u +%H:%M:%S) $*" >> "$L/relay.log"; }
SSHO="-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
log "RELAY_START $RUN $*"
while :; do
  fin=0
  { [ -f "$L/STOP" ] || grep -q A74_LEARNER_DONE "$L/learner.log" 2>/dev/null; } && fin=1
  for b in "$@"; do
    IFS=: read -r name port host <<< "$b"
    S="ssh $SSHO -p $port root@$host"
    files=$($S "cd $R/rollouts 2>/dev/null && ls *.npz 2>/dev/null | head -300" 2>/dev/null | grep '\.npz$' | tr '\n' ' ')
    n=0; msg=""
    if [ -n "$files" ]; then
      inc="$L/incoming/$name"; mkdir -p "$inc"
      if $S "cd $R/rollouts && tar cf - $files" 2>/dev/null | tar xf - -C "$inc"; then
        for f in $files; do
          [ -s "$inc/$f" ] && mv "$inc/$f" "$L/rollouts/${name}_$f" && n=$(( n + 1 ))
        done
        $S "cd $R/rollouts && mkdir -p relayed && mv $files relayed/" 2>/dev/null || msg=" ACK_FAILED"
      else
        msg=" PULL_FAILED"
      fi
    fi
    if [ -f "$L/learner/head_latest.pth" ]; then
      h=$(md5sum < "$L/learner/head_latest.pth" | cut -c1-32)
      if [ "$h" != "${last[$name]:-}" ]; then
        if $S "mkdir -p $R/learner && cat > $R/learner/head_latest.pth.tmp && mv $R/learner/head_latest.pth.tmp $R/learner/head_latest.pth" \
             < "$L/learner/head_latest.pth" 2>/dev/null; then
          last[$name]=$h; msg="$msg head ${h:0:8}"
        else
          msg="$msg HEAD_PUSH_FAILED"
        fi
      fi
    fi
    [ $fin = 1 ] && { $S "touch $R/STOP" 2>/dev/null; msg="$msg STOP"; }
    log "$name episodes $n$msg"
  done
  [ $fin = 1 ] && { log RELAY_DONE; exit 0; }
  sleep "${A74_RELAY_POLL:-45}"
done
