"""scripts/lib/ez_train.sh: the resumable retry loop shared by the Atari launchers, run with a stub trainer."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

# stub python: record the argv, write checkpoint_latest.pt on the first call and fail; succeed from then on
STUB = r'''#!/usr/bin/env bash
echo "$*" >> "$CALLS"
out=""; label=""; prev=""
for a in "$@"; do [ "$prev" = --log-dir ] && out=$a; [ "$prev" = --run-label ] && label=$a; prev=$a; done
ck=$out/$label/checkpoints/checkpoint_latest.pt
if [ ! -f "$ck" ]; then touch "$ck"; exit 3; fi
exit 0
'''


def test_train_resumable_retries_and_resumes(tmp_path):
    root = tmp_path / "MThesis"
    root.mkdir()
    py = tmp_path / "py.sh"
    py.write_text(STUB, newline="\n")
    py.chmod(0o755)
    calls = tmp_path / "calls.txt"
    out, logs = tmp_path / "out", tmp_path / "logs"
    env = {**os.environ, "WORKSPACE": tmp_path.as_posix(), "MTHESIS_ROOT": root.as_posix(), "PY_MAIN": py.as_posix(),
           "PYTHON3": sys.executable, "CALLS": calls.as_posix(), "EZ_RETRY_SLEEP": "0"}
    lib = (REPO / "scripts" / "lib" / "ez_train.sh").as_posix()
    r = subprocess.run([BASH, "-c", f". '{lib}'; ez_env 1; ez_train_resumable lab '{out.as_posix()}' '{logs.as_posix()}' "
                                    "--trunk qwen --seed 4 --graph-search; echo rc=$? gpu=$CUDA_VISIBLE_DEVICES"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert "rc=0 gpu=1" in r.stdout, r.stdout + r.stderr
    first, second = calls.read_text().splitlines()
    assert "--trunk qwen --seed 4 --graph-search" in first and "--eval-mode deferred" in first
    assert "--resume-from" not in first
    assert f"--resume-from {out.as_posix()}/lab/checkpoints/checkpoint_latest.pt" in second
    log = (logs / "lab.log").read_text()
    assert "try 1 fresh on GPU 1" in log and "exited rc=3" in log and "exited rc=0" in log
