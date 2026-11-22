# Shared pieces of the box-side chain scripts (scripts/ops/*_chain.sh). Source it from a chain, do not run it:
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/chain.sh"
#
# A chain waits for the shipped inputs, runs ladder_box.sh steps in order and records each stage as a line in one status file.
# Every chain used to copy the same helpers (status file, wait-for-mark, skip-a-finished-step, render check, lane completion).
# This is the one copy. Paths come from common.sh, so `WORKSPACE=/data bash scripts/ops/r1_chain.sh` works.
# Sets: L = $WORKSPACE/$CHAIN_NAME (default ladder), S = $L/status, B = scripts/ops/ladder_box.sh; cds into $MTHESIS_ROOT.
# Tested by tests/test_scripts_chain.py.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
L=$WORKSPACE/${CHAIN_NAME:-ladder}; S=$L/status; mkdir -p "$L"
B=scripts/ops/ladder_box.sh
: "${PYC:=/venv/main/bin/python}"   # the trainer venv of the Vast PyTorch template
cd "$MTHESIS_ROOT" || exit 1

mark() { echo "$(date -u +%H:%M:%S) $1" >> "$S"; }

# chain_stop WHY: record "CHAIN_STOP WHY" and end the chain.
chain_stop() { mark "CHAIN_STOP $*"; exit 1; }

# wait_for OK FAIL: wait until the status file has a line matching OK or FAIL (extended regexes); 0 if OK matched.
wait_for() { until grep -q -E "$1|$2" "$S"; do sleep "${CHAIN_POLL:-30}"; done; grep -q -E "$1" "$S"; }

# await OK FAIL WHY: wait_for, and stop the chain with WHY unless OK.
await() { wait_for "$1" "$2" || chain_stop "$3"; }

# run_or_stop ARGS...: `ladder_box.sh ARGS`; a failure stops the chain ("CHAIN_STOP ARGS").
run_or_stop() { bash "$B" "$@" || chain_stop "$*"; }

# step DONE ARGS...: run_or_stop ARGS unless DONE (a regex) is already in the status file, so a restarted chain skips it.
step() { local done=$1; shift; grep -q -E "$done" "$S" || run_or_stop "$@"; }

# wait_files FILE...: until every file exists (box_deploy.sh touches DIR/READY once a checkpoint dir verified).
wait_files() { local f; for f in "$@"; do until [ -f "$f" ]; do sleep "${CHAIN_POLL:-20}"; done; done; }

# box_tests TEST...: the new tests on this box's GPU and packages before runs depend on them; once (BOX_TESTS_OK).
box_tests() {
  grep -q BOX_TESTS_OK "$S" && return 0
  if "$PYC" -m pytest -q -p no:cacheprovider "$@" > "$L/box_tests.log" 2>&1; then mark "BOX_TESTS_OK $(tail -1 "$L/box_tests.log")"
  else chain_stop "box tests ($(tail -1 "$L/box_tests.log"))"; fi
}

# render_test [stop]: can CARLA render? Marks "RENDER_OK frames N" (returns 0) or "RENDER_FAILED (see LOG)" (returns 1);
# with "stop" a failure ends the chain as "CHAIN_STOP render (see LOG)" instead.
render_test() {
  bash "$WORKSPACE/carla_render_test.sh" > "$L/render_test.log" 2>&1
  if grep -q RENDER_OK "$L/render_test.log"; then mark "$(grep -o 'RENDER_OK.*' "$L/render_test.log" | head -1)"; return 0; fi
  [ "${1:-}" = stop ] && chain_stop "render (see $L/render_test.log)"
  mark "RENDER_FAILED (see $L/render_test.log)"; return 1
}

# render_when_carla: (run in the background) wait for the CARLA install, then render_test; RENDER_SKIPPED if CARLA failed.
render_when_carla() {
  wait_for CARLA_PROV_END CARLA_PROV_FAILED || { mark "RENDER_SKIPPED carla"; return 1; }
  render_test
}

# cgroup_cores: the container's CPU quota, not nproc (S-086: a 128-thread host with a quota of 20); nproc when unlimited.
cgroup_cores() {
  local c; c=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null)
  if [ "${c:-0}" -gt 0 ] 2>/dev/null; then echo "$c"; else nproc; fi
}

# lanes_done NL: every ladder lane 1..NL logged "lane 20k0/GPU0 done" (ladder_box.sh lanes numbers them 2010, 2020, ...).
lanes_done() {
  local k; for k in $(seq 1 "$1"); do
    grep -q "lane $((2000 + 10 * k))/GPU0 done" "$WORKSPACE/evlogs/ladder_L$k.out" 2>/dev/null || return 1
  done
}

# trained_list PREFIX ARM...: " ARM" for every $CHECKPOINTS/PREFIX ARM/TRAINED, for the TRAINED: / *_DONE marks.
trained_list() { local p=$1 a; shift; for a in "$@"; do [ -f "$CHECKPOINTS/$p$a/TRAINED" ] && printf ' %s' "$a"; done; return 0; }
