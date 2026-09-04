"""Detection and cleanup of abandoned training processes still holding GPU memory.

A run stopped with Ctrl+Z is only SUSPENDED, not terminated: it keeps its CUDA
context and every byte of VRAM it had allocated, so the next run starts against
a GPU that looks almost full. This is easy to misread as a driver-level leak,
because `nvidia-smi` reports host-namespace PIDs that don't exist inside a
container's PID namespace - `kill -9 <that pid>` answers "No such process"
whether the process is dead or perfectly alive. Matching on the command line via
`ps` finds the container-local PIDs that can actually be signalled.
"""
from typing import Dict, List, Optional, Sequence
import os
import signal
import subprocess
import time


def find_matching_processes(pattern: str, exclude_pids: Optional[Sequence[int]] = None) -> List[Dict]:
    """Returns processes whose command line contains `pattern`, as dicts of
    pid/stat/etime/cmd. Always excludes the calling process itself."""
    if os.name == "nt":
        return []

    exclude = set(exclude_pids or []) | {os.getpid()}
    found: List[Dict] = []
    try:
        out = subprocess.run(["ps", "-eo", "pid,stat,etime,cmd"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return found

    for line in out.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid_str, stat, etime, cmd = parts
        if pattern not in cmd:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid in exclude:
            continue
        found.append({"pid": pid, "stat": stat, "etime": etime, "cmd": cmd})
    return found


def cleanup_stale_processes(pattern: str, kill_suspended: bool = True) -> int:
    """Reports other instances of `pattern` and terminates the suspended ones.

    Only processes in state T (stopped) are killed: those are abandoned by
    definition - a suspended trainer makes no progress while pinning its VRAM,
    which is exactly the state a Ctrl+Z leaves behind. Running instances are
    reported but never touched, since those may be deliberate concurrent runs.

    SIGKILL rather than SIGTERM: a stopped process never gets scheduled to run a
    signal handler, so a SIGTERM would just sit pending and free nothing.
    """
    procs = find_matching_processes(pattern)
    if not procs:
        return 0

    suspended = [p for p in procs if p["stat"].startswith("T")]
    running = [p for p in procs if not p["stat"].startswith("T")]

    if running:
        print(f"[gpu_cleanup] {len(running)} other '{pattern}' process(es) are RUNNING and were left alone:")
        for p in running[:5]:
            print(f"    PID {p['pid']} (stat={p['stat']}, up {p['etime']})")

    if not suspended:
        return 0
    if not kill_suspended:
        print(f"[gpu_cleanup] {len(suspended)} suspended '{pattern}' process(es) are holding VRAM "
              f"(kill_stale is off - free them with: pkill -9 -f {pattern})")
        return 0

    print(f"[gpu_cleanup] Found {len(suspended)} SUSPENDED '{pattern}' process(es) still holding "
          f"their CUDA context (likely a previous run stopped with Ctrl+Z). Terminating them:")
    killed = 0
    for p in suspended:
        try:
            os.kill(p["pid"], signal.SIGKILL)
            killed += 1
        except ProcessLookupError:
            pass  # Already gone between the ps snapshot and now.
        except PermissionError:
            print(f"    PID {p['pid']}: permission denied - not ours to kill, leaving it.")

    if killed:
        print(f"[gpu_cleanup] Terminated {killed} suspended process(es) (up to {suspended[0]['etime']} old); "
              f"waiting for the driver to release their VRAM...")
        time.sleep(3)
    return killed
