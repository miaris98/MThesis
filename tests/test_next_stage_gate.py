"""TODO (2026-11-03): stage gating. A launcher runs the next Atari stage only when the previous stage's log passes the gate. The gate's
exit code is the verdict (0 pass, 1 fail), so no script can start Gate N+1 from a log that failed Gate N."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEALTHY = ("upd 1000/1000 | loss v 0.5 p 7.0 | attn H 0.86/0.84/0.87/0.85 pol 0.90 | 1.0 min\n"
           "[play] life len 42 | act share 0.24/0.26/0.25/0.25 | act-pi H 1.386\n")
COLLAPSED = HEALTHY.replace("attn H 0.86/0.84/0.87/0.85", "attn H 0.50/0.23/0.71/0.73")


def _run(tmp_path, log_text, marker):
    log = tmp_path / "stage.log"
    log.write_text(log_text, encoding="utf-8")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "analysis", "next_stage_gate.py"), str(log), "--",
           sys.executable, "-c", f"open(r'{marker}', 'w').write('started')"]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_the_gate_exits_0_on_pass_and_1_on_fail(tmp_path):
    ok = tmp_path / "ok.log"
    ok.write_text(HEALTHY, encoding="utf-8")
    bad = tmp_path / "bad.log"
    bad.write_text(COLLAPSED, encoding="utf-8")
    gate = os.path.join(ROOT, "scripts", "analysis", "atari_gate_check.py")
    assert subprocess.run([sys.executable, gate, str(ok)], capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, gate, str(bad)], capture_output=True).returncode == 1


def test_the_next_stage_starts_after_a_passing_gate(tmp_path):
    marker = tmp_path / "started.txt"
    r = _run(tmp_path, HEALTHY, marker)
    assert r.returncode == 0 and marker.read_text() == "started"


def test_the_next_stage_never_starts_after_a_failing_gate(tmp_path):
    marker = tmp_path / "started.txt"
    r = _run(tmp_path, COLLAPSED, marker)
    assert r.returncode == 1 and not marker.exists()
    assert "attention" in r.stdout + r.stderr
