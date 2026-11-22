#!/usr/bin/env bash
# 2026-11-13: package installs into a box venv that survive a silent network stall (box 60465887: uv sat for 30-45 min on an idle
# connection to the PyPI CDN, with no CPU and no output, and the setup went on without it, so the venv ended with no torch).
#   * one install at a time per venv (flock), so two setup steps never write the same venv;
#   * every attempt is capped (CAP seconds, default 1500); a stalled attempt is killed and retried (ATTEMPTS, default 3);
#   * uv's HTTP timeout and retry count are set, so a dead connection fails fast instead of hanging;
#   * if uv fails every attempt, the same packages are installed with pip (bootstrapped into the venv with uv);
#   * exit status 0 only if the install succeeded; the caller must stop on a nonzero status.
# usage: scripts/ops/install_guard.sh LABEL VENV_DIR -- uv-pip-install-args...
#   e.g. scripts/ops/install_guard.sh carla_torch $WORKSPACE/venv_carla -- torch torchvision --index-url https://download.pytorch.org/whl/cu128
set -uo pipefail
LABEL=${1:?label}; VENV=${2:?venv dir}; shift 2
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "install_guard: no packages given" >&2; exit 2; }
LOG=${GUARD_LOG:-/workspace/logs_prov/install_guard.log}
mkdir -p "$(dirname "$LOG")"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10} UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
ATTEMPTS=${ATTEMPTS:-3}; CAP=${CAP:-1500}
say() { echo "$(date -u +%H:%M:%S) [$LABEL] $*" | tee -a "$LOG"; }

exec 9>"$VENV.lock"
flock 9                                   # wait for any other install into this venv to finish
for a in $(seq 1 "$ATTEMPTS"); do
  say "uv attempt $a of $ATTEMPTS: $*"
  VIRTUAL_ENV="$VENV" timeout --kill-after=30 "$CAP" uv pip install "$@" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then say "OK (uv)"; exit 0; fi
  say "uv attempt $a failed (rc=$rc)"
done

say "uv failed $ATTEMPTS times; installing the same packages with pip"
VIRTUAL_ENV="$VENV" uv pip install pip >> "$LOG" 2>&1 || say "could not bootstrap pip into $VENV"
timeout --kill-after=30 "$CAP" "$VENV/bin/python" -m pip install --retries 10 --timeout 120 "$@" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say "OK (pip)"; exit 0; fi
say "FAILED (pip rc=$rc)"
exit 1
