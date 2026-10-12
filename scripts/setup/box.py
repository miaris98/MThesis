#!/usr/bin/env python3
"""One tool for looking at a remote box, instead of a new check_*.py per run.

Reads host/port/paths from `.remote` via remote_config.py (env overrides: REMOTE_HOST, REMOTE_PORT, ...).

    python scripts/setup/box.py probe                 # ssh reachable? OS, GPUs, disk, container limits
    python scripts/setup/box.py status [-p PATTERN]... [-l LOG]...
                                                      # GPU use, matching processes, log tails, disk, limits
    python scripts/setup/box.py env                   # can each venv import torch/CUDA (and carla, gym)?

The container limits come from running scripts/lib/machine.py on the box (piped over ssh, so the repo need not
be there): the cgroup CPU quota and pids.max, not the host's CPU count, which is what every launcher should size
itself by (S-086). Process patterns use the bracket trick, so the remote shell never matches itself (S-013).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from remote_config import load_config  # noqa: E402

MACHINE_PY = HERE.parent / "lib" / "machine.py"


def bracket(pattern: str) -> str:
    """'train_wor' -> '[t]rain_wor': still matches the target, never the grep/pgrep command line itself."""
    return f"[{pattern[0]}]{pattern[1:]}" if pattern and pattern[0].isalnum() else pattern


def run(cfg, cmd: str, stdin: str | None = None, timeout: int = 60) -> str:
    argv = cfg.ssh_T(cmd) if stdin is not None else cfg.ssh(cmd)
    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    return (r.stdout + (f"\n[stderr] {r.stderr.strip()}" if r.returncode and r.stderr.strip() else "")).rstrip()


def limits(cfg) -> str:
    return run(cfg, "python3 - --export", stdin=MACHINE_PY.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    st = sub.add_parser("status")
    st.add_argument("-p", "--pattern", action="append", default=["train_", "leaderboard_evaluator", "CarlaUE4-Linux",
                                                                 "b2d_guardian", "ez/train.py"])
    st.add_argument("-l", "--log", action="append", default=[], help="remote log to tail (repeatable)")
    st.add_argument("-n", "--lines", type=int, default=15)
    sub.add_parser("env")
    args = ap.parse_args()
    cfg = load_config()
    print(f"== {cfg.user}@{cfg.host}:{cfg.port}")

    if args.cmd == "probe":
        print(run(cfg, "uname -srm; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader;"
                       f" df -h / {shlex.quote(cfg.workspace)} 2>/dev/null | sort -u", timeout=30))
        print(limits(cfg))
    elif args.cmd == "status":
        print(run(cfg, "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
                       "--format=csv,noheader"))
        pats = "|".join(bracket(p) for p in args.pattern)
        print("-- processes"); print(run(cfg, f"ps -eo pid,etime,pcpu,rss,args | grep -E {shlex.quote(pats)} | cut -c1-200"))
        for log in args.log:
            print(f"-- tail {log}"); print(run(cfg, f"tail -n {args.lines} {shlex.quote(log)}"))
        print("-- disk"); print(run(cfg, f"df -h {shlex.quote(cfg.workspace)} | tail -1"))
        print("-- limits"); print(limits(cfg))
    elif args.cmd == "env":
        check = ("import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                 "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')")
        for label, py in (("main", cfg.python()), ("carla", cfg.python_carla())):
            print(f"-- {label}: {py}")
            print(run(cfg, f"{shlex.quote(py)} -c {shlex.quote(check)} 2>&1 | tail -1"))
        print(run(cfg, f"{shlex.quote(cfg.python_carla())} -c 'import carla; print(\"carla ok\")' 2>&1 | tail -1"))
        print(run(cfg, f"{shlex.quote(cfg.python())} -c 'import gymnasium, ale_py; print(\"gymnasium/ale ok\")' 2>&1 | tail -1"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
