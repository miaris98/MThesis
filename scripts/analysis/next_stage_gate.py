"""Stage gate (2026-11-03): run the next Atari stage only if the previous stage's log passes the Gate 1 / Gate 2 check.
usage: py scripts/analysis/next_stage_gate.py PREV_LOG -- COMMAND [ARGS...]
The gate's verdict is atari_gate_check.gate_verdict on PREV_LOG. On a fail the command is not started and the exit code is 1."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.analysis.atari_gate_check import gate_verdict  # noqa: E402


def main(argv) -> int:
    if "--" not in argv or argv.index("--") != 1:
        print("usage: next_stage_gate.py PREV_LOG -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    log, cmd = argv[0], argv[2:]
    with open(log, encoding="utf-8", errors="replace") as f:
        ok, reasons = gate_verdict(f.read().splitlines())
    if not ok:
        print(f"GATE FAIL {log}: " + "; ".join(reasons) + " -- next stage not started", file=sys.stderr)
        return 1
    print(f"GATE PASS {log}: starting the next stage", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
