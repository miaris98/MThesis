#!/usr/bin/env bash
# TODO A84 (2026-11-17), eval box A: write the rb and dg READY markers only after the seed-1 filler has finished on every lane (FILLER_DONE in
# $WORKSPACE/a84/status). Writing them earlier would start a lane's rb run on a CARLA port that a filler still holds.
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
S=$WORKSPACE/a84/status
until awk '/FILLER_DONE/{f=1} END{exit !f}' $S; do sleep 60; done
touch $WORKSPACE/checkpoints/a84_rb/READY $WORKSPACE/checkpoints/a84_dg/READY
echo "$(date -u +%H:%M:%S) READY_rb READY_dg written after FILLER_DONE" >> $S
