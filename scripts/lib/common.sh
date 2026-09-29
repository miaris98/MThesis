# Shared paths and helpers for the shell scripts. Source it, do not run it:
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
#
# Every path is a default that the environment can override, so a box with a different layout needs
# env vars, not edits (the Python side does the same through MTHESIS_EXP_ROOT, src/config/paths.py).
# The helpers below used to be copied into each script; the copy in eval_watchdog.sh and queue_A9c.sh
# lacked `local` and never killed the evaluator (S-099). One copy here, tested by
# tests/test_scripts_common.py.

: "${WORKSPACE:=/workspace}"
: "${MTHESIS_ROOT:=$WORKSPACE/MThesis}"
: "${CARLA_DIR:=$WORKSPACE/carla}"            # the real CARLA install (CARLA_ROOT may point at a shim)
: "${GARAGE:=$WORKSPACE/carla_garage}"
: "${VENV_CARLA:=$WORKSPACE/venv_carla}"
: "${B2D_OUT:=$WORKSPACE/bench2drive_out}"
: "${GUARDIAN_LOGS:=$WORKSPACE/guardian_logs}"
: "${CHECKPOINTS:=$WORKSPACE/checkpoints}"
: "${PYTHON3:=python3}"                   # interpreter for the small JSON helpers below
export WORKSPACE MTHESIS_ROOT CARLA_DIR GARAGE VENV_CARLA B2D_OUT GUARDIAN_LOGS CHECKPOINTS PYTHON3

# CUDA numbers GPUs fastest-first by default; CARLA's -graphicsadapter and nvidia-smi use PCI order. On a
# box with mixed cards, CUDA_VISIBLE_DEVICES=1 and -graphicsadapter=1 would then be different GPUs.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

log() { echo "$(date -u '+%F %T') $*"; }

# All descendants of a pid, deepest first. `local c` matters: without it the recursion overwrote c and
# returned only the deepest leaf (S-099).
desc() { local c; for c in $(ps -eo pid=,ppid= | awk -v p="$1" '$2==p {print $1}'); do desc "$c"; echo "$c"; done; }

# SIGKILL each pid and its whole tree. CARLA runs under `su carlauser` below the evaluator and survives a
# plain kill of its parent, then holds its ports for every later launch (S-091).
kill_tree() { local p t; for p in "$@"; do t=$(desc "$p"); kill -9 $t "$p" 2>/dev/null; done; return 0; }

# pids of CARLA processes (CarlaUE4.sh wrapper or CarlaUE4-Linux-Shipping) serving an RPC port. Matches the
# comm prefix through /proc: `pgrep -x` truncates comm to 15 chars and never matches (S-014), and
# `pkill -f <port string>` can match the caller's own command line (S-013).
carla_pids_on_port() {
  local port=$1 pid cmd
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    case "$(cat "/proc/$pid/comm" 2>/dev/null)" in CarlaUE4*) ;; *) continue ;; esac
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) "
    case "$cmd" in *"-carla-rpc-port=$port "*|*"-carla-port=$port "*) echo "$pid" ;; esac
  done
}

# Kill the CARLA servers in a lane's port range [port-2, port+4]: RPC, streaming and secondary ports.
clear_carla_ports() {
  local q
  for q in $(seq $(( $1 - 2 )) $(( $1 + 4 ))); do kill_tree $(carla_pids_on_port "$q"); done
}

# 0 if the file is valid JSON (a kill -9 mid-write leaves a truncated result file, 2026-09-28).
json_ok() { "$PYTHON3" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null; }

# Leaderboard progress[0] of a result file, 0 if missing/unreadable. Paths go through argv, not string
# interpolation, so a path with quotes cannot break the Python snippet.
b2d_progress() {
  "$PYTHON3" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("_checkpoint", {}).get("progress", [0])[0])
except Exception:
    print(0)' "$1" 2>/dev/null || echo 0
}

# Number of route records in a result file, 0 if missing/unreadable.
b2d_records() {
  "$PYTHON3" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["_checkpoint"]["records"]))' "$1" 2>/dev/null || echo 0
}

# Sizes for this box (see scripts/lib/machine.py): exports MACHINE_* and REC_* variables.
machine_env() { eval "$("$PYTHON3" "$MTHESIS_ROOT/scripts/lib/machine.py" --export 2>/dev/null)"; }
