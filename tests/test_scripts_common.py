"""scripts/lib/common.sh: paths, JSON helpers and the process-tree kill used by the eval scripts."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "common.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def sh(script, **env):
    full_env = {**os.environ, "PYTHON3": sys.executable, **env}
    out = subprocess.run([BASH, "-c", f". '{COMMON.as_posix()}'; {script}"], capture_output=True, text=True, env=full_env)
    return out.stdout.strip()


def test_path_defaults_and_overrides():
    assert sh('echo "$MTHESIS_ROOT $B2D_OUT"', WORKSPACE="/ws") == "/ws/MThesis /ws/bench2drive_out"
    assert sh('echo "$GARAGE"', WORKSPACE="/ws", GARAGE="/opt/garage") == "/opt/garage"
    assert sh('echo "$CUDA_DEVICE_ORDER"') == "PCI_BUS_ID"


def test_json_helpers(tmp_path):
    good = tmp_path / "r.json"
    good.write_text(json.dumps({"_checkpoint": {"progress": [7, 20], "records": [{}, {}, {}]}}))
    bad = tmp_path / "bad.json"
    bad.write_text('{"_checkpoint": {"progr')  # a kill -9 mid-write
    p = good.as_posix()
    assert sh(f"json_ok '{p}' && echo ok") == "ok"
    assert sh(f"json_ok '{bad.as_posix()}' || echo broken") == "broken"
    assert sh(f"b2d_progress '{p}'") == "7"
    assert sh(f"b2d_records '{p}'") == "3"
    assert sh(f"b2d_progress '{bad.as_posix()}'") == "0"
    assert sh(f"b2d_records '{(tmp_path / 'missing.json').as_posix()}'") == "0"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux ps/proc")
def test_desc_and_kill_tree_reach_grandchildren():
    # parent -> child -> grandchild, like run script -> evaluator -> su/CARLA (S-091, S-099)
    parent = subprocess.Popen([BASH, "-c", "bash -c 'sleep 300 & wait' & wait"])
    time.sleep(0.5)
    tree = sh(f"desc {parent.pid}").split()
    assert len(tree) >= 2, tree  # the child bash and its sleep, not only the deepest leaf
    sh(f"kill_tree {parent.pid}")
    time.sleep(0.3)
    assert parent.poll() is not None
    for pid in tree:
        assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]
