"""scripts/lib/chain.sh and the chains built on it (scripts/ops/*_chain.sh), run against a fake workspace.

The fake workspace has a stub ladder_box.sh that writes the status marks the real one writes, so a whole chain runs here
in seconds: the order of the stages, the CHAIN_STOP marks and which arms reach `train` are what the tests check.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHAIN = REPO / "scripts" / "lib" / "chain.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")

STUB_BOX = r'''#!/usr/bin/env bash
# stub ladder_box.sh: log the call, write the mark the real step writes, fail when FAIL_STEP names the step
S=$WORKSPACE/${CHAIN_NAME:-ladder}/status
echo "$*" >> "$WORKSPACE/calls.log"
[ "${FAIL_STEP:-}" = "$1" ] && exit 1
case $1 in
  setup) echo "00:00:00 SETUP_START" >> $S; echo "00:00:00 DATA_DONE" >> $S; echo "00:00:00 CARLA_PROV_END" >> $S ;;
  audit) echo "00:00:00 AUDIT_DONE" >> $S ;;
  cache) echo "00:00:00 CACHE_DONE $2 " >> $S ;;
  cache_full) echo "00:00:00 CACHE_FULL_DONE DX" >> $S ;;
  train) mkdir -p "$CHECKPOINTS/$2"; touch "$CHECKPOINTS/$2/TRAINED" ;;
  train_tfpp) mkdir -p "$CHECKPOINTS/a86_tfpp_cam"; touch "$CHECKPOINTS/a86_tfpp_cam/TRAINED" ;;
esac
exit 0
'''


@pytest.fixture
def ws(tmp_path):
    """A workspace with the repo's scripts/lib and the stub ladder_box.sh under $WORKSPACE/MThesis."""
    w = tmp_path / "ws"
    root = w / "MThesis"
    (root / "scripts" / "ops").mkdir(parents=True)
    shutil.copytree(REPO / "scripts" / "lib", root / "scripts" / "lib")
    for name in ("tfpp_chain.sh", "mem_chain.sh"):
        shutil.copy(REPO / "scripts" / "ops" / name, root / "scripts" / "ops" / name)
    (root / "scripts" / "ops" / "ladder_box.sh").write_text(STUB_BOX, newline="\n")
    pyc = w / "pyc.sh"   # the trainer python: only `-m pytest ...` is called, print a pytest-like last line
    pyc.write_text("#!/usr/bin/env bash\necho '1 passed in 0.01s'\n", newline="\n")
    pyc.chmod(0o755)
    (w / "checkpoints" / "E15").mkdir(parents=True)
    (w / "checkpoints" / "E15" / "READY").touch()
    (w / "ladder").mkdir()
    (w / "ladder" / "bench2drive220.xml").touch()
    return w


def env_for(w, **extra):
    return {**os.environ, "WORKSPACE": w.as_posix(), "MTHESIS_ROOT": (w / "MThesis").as_posix(),
            "CHECKPOINTS": (w / "checkpoints").as_posix(), "PYC": (w / "pyc.sh").as_posix(), "PYTHON3": sys.executable,
            "CHAIN_POLL": "0.1", "CHAIN_STAGGER": "0", **extra}


def run_chain(w, name, **extra):
    script = (w / "MThesis" / "scripts" / "ops" / name).as_posix()
    return subprocess.run([BASH, script], capture_output=True, text=True, env=env_for(w, **extra), timeout=120)


def lib(w, snippet, **extra):
    """Source the repo's chain.sh in the fake workspace and run SNIPPET; returns (rc, status file text)."""
    r = subprocess.run([BASH, "-c", f". '{CHAIN.as_posix()}'; {snippet}"], capture_output=True, text=True,
                       env=env_for(w, MTHESIS_ROOT=(w / "MThesis").as_posix(), **extra), timeout=60)
    status = w / "ladder" / "status"
    return r, status.read_text() if status.exists() else ""


def marks(w, name="ladder"):
    return [ln.split(" ", 1)[1] for ln in (w / name / "status").read_text().splitlines()]


def test_await_stops_the_chain_on_the_failure_mark(ws):
    (ws / "ladder" / "status").write_text("00:00:00 DATA_FAILED\n")
    r, status = lib(ws, 'await DATA_DONE DATA_FAILED data; echo after')
    assert r.returncode == 1 and "after" not in r.stdout
    assert status.splitlines()[-1].endswith("CHAIN_STOP data")


def test_step_skips_a_finished_step_and_runs_a_new_one(ws):
    (ws / "ladder" / "status").write_text("00:00:00 AUDIT_DONE\n")
    r, status = lib(ws, 'step AUDIT_DONE audit; step "CACHE_DONE D " cache D; echo after')
    assert r.returncode == 0 and "after" in r.stdout
    assert (ws / "calls.log").read_text().split("\n")[0] == "cache D"   # audit was not run again
    r, status = lib(ws, 'run_or_stop audit; echo after', FAIL_STEP="audit")
    assert r.returncode == 1 and status.splitlines()[-1].endswith("CHAIN_STOP audit")


def test_trained_list_and_lanes_done(ws):
    for a in ("r1d_r1_s0", "r1d_r0ft"):
        (ws / "checkpoints" / a).mkdir()
        (ws / "checkpoints" / a / "TRAINED").touch()
    r, _ = lib(ws, 'echo "[$(trained_list r1d_ r1_s0 r1_s1 r0ft)]"')
    assert r.stdout.strip() == "[ r1_s0 r0ft]"
    (ws / "evlogs").mkdir()
    (ws / "evlogs" / "ladder_L1.out").write_text("lane 2010/GPU0 done\n")
    (ws / "evlogs" / "ladder_L2.out").write_text("still running\n")
    r, _ = lib(ws, 'lanes_done 1 && echo one; lanes_done 2 || echo not-two')
    assert r.stdout.split() == ["one", "not-two"]


def test_render_test_marks_and_stops(ws):
    render = ws / "carla_render_test.sh"
    render.write_text("echo RENDER_OK frames 12\n", newline="\n")
    r, status = lib(ws, "render_test stop && echo rendered")
    assert "rendered" in r.stdout and status.splitlines()[-1].endswith("RENDER_OK frames 12")
    render.write_text("echo RENDER_FAILED frames 0\n", newline="\n")
    r, status = lib(ws, "render_test stop; echo after")
    assert r.returncode == 1 and "CHAIN_STOP render (see" in status.splitlines()[-1]


def test_tfpp_chain_runs_every_stage_and_trains_the_arms(ws):
    r = run_chain(ws, "tfpp_chain.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    m = marks(ws)
    order = [m.index(x) for x in ("TFPP_CHAIN_START", "CKPTS_READY", "AUDIT_DONE", "CACHE_FULL_DONE DX")]
    assert order == sorted(order)
    assert any(x.startswith("BOX_TESTS_OK 1 passed") for x in m)
    assert m[-1] == "TFPP_DONE trained: a87_fm a86_tfpp_cam a87_tp a87_moelat"
    calls = (ws / "calls.log").read_text().splitlines()
    assert "train_tfpp" in calls and "train a87_tp" in calls


def test_tfpp_chain_single_arm_resume_replaces_a86_chain(ws):
    r = run_chain(ws, "tfpp_chain.sh", ARMS="a86_tfpp_cam")
    assert r.returncode == 0, r.stdout + r.stderr
    assert marks(ws)[-1] == "TFPP_DONE trained: a86_tfpp_cam"
    assert [c for c in (ws / "calls.log").read_text().splitlines() if c.startswith("train")] == ["train_tfpp"]


def test_mem_chain_stops_when_the_audit_fails(ws):
    r = run_chain(ws, "mem_chain.sh", FAIL_STEP="audit")
    assert r.returncode == 1
    assert marks(ws)[-1] == "CHAIN_STOP audit"
    assert not any(c.startswith("train") for c in (ws / "calls.log").read_text().splitlines())


def test_mem_chain_skips_trained_arms(ws):
    (ws / "checkpoints" / "mem_L0").mkdir()
    (ws / "checkpoints" / "mem_L0" / "TRAINED").touch()
    r = run_chain(ws, "mem_chain.sh", ARMS="mem_L0 mem_L1")
    assert r.returncode == 0, r.stdout + r.stderr
    trains = [c for c in (ws / "calls.log").read_text().splitlines() if c.startswith("train")]
    assert trains == ["train mem_L1"]
    assert marks(ws)[-1] == "MEM_DONE trained: mem_L0 mem_L1"
