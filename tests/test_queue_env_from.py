"""2026-11-20: queue_A9c.sh's CKE job exports its K=V flags and, with ENV_FROM=<file>, waits for the file and exports its lines
(LANE_SKIP=1 skips the job). The CKE case is cut out of the real script and run with a stub eval_job."""
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def _cke_case() -> str:
    lines = (ROOT / "scripts" / "eval" / "queue_A9c.sh").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("CKE:*)"))
    j = next(k for k in range(i, len(lines)) if lines[k].rstrip().endswith(";;"))
    return "\n".join(lines[i:j + 1])


def _run(job: str, tmp_path: Path, timeout=20):
    script = tmp_path / "q.sh"
    script.write_text("C=/ck\n"
                      'eval_job() { echo "EVAL $1 $2 $3 CTL=${WOR_CONTROLLER:-} Q=${CARLA_QUALITY:-} X=${EXTRA:-}"; }\n'
                      f'job="$1"\ncase $job in\n{_cke_case()}\nesac\n', newline="\n")
    return subprocess.run([BASH, str(script), job], capture_output=True, text=True, timeout=timeout,
                          env={"PATH": "/usr/bin:/bin", "ENV_FROM_POLL_S": "0.2"})


def _posix(p: Path) -> str:
    s = p.as_posix()
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s     # C:/x -> /c/x for Git Bash


pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def test_plain_flags_are_exported(tmp_path):
    r = _run("CKE:lab:ck/m.pth:1,2,3:CARLA_QUALITY=Epic,WOR_CONTROLLER=pdm", tmp_path)
    assert "EVAL lab /ck/ck/m.pth 1,2,3 CTL=pdm Q=Epic" in r.stdout, r.stdout + r.stderr


def test_env_from_file_is_exported(tmp_path):
    f = tmp_path / "winner.env"
    f.write_text("# picked by the chain\nWOR_CONTROLLER=pdm_lat\nEXTRA=a=b\nnot a line\n", newline="\n")
    r = _run(f"CKE:lab:m.pth:5:CARLA_QUALITY=Epic,ENV_FROM={_posix(f)}", tmp_path)
    assert "EVAL lab /ck/m.pth 5 CTL=pdm_lat Q=Epic X=a=b" in r.stdout, r.stdout + r.stderr


def test_lane_skip_skips_the_job(tmp_path):
    f = tmp_path / "go"
    f.write_text("LANE_SKIP=1\n", newline="\n")
    r = _run(f"CKE:lab:m.pth:5:ENV_FROM={_posix(f)}", tmp_path)
    assert "EVAL" not in r.stdout and "skip lab (LANE_SKIP=1" in r.stdout, r.stdout + r.stderr


def test_waits_until_the_file_appears(tmp_path):
    f = tmp_path / "late.env"
    threading.Timer(1.0, lambda: f.write_text("WOR_CONTROLLER=brake\n", newline="\n")).start()
    t0 = time.time()
    r = _run(f"CKE:lab:m.pth:5:ENV_FROM={_posix(f)}", tmp_path)
    assert time.time() - t0 >= 0.9
    assert "waits for" in r.stdout and "CTL=brake" in r.stdout, r.stdout + r.stderr
