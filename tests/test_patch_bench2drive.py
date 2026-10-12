"""scripts/setup/patch_bench2drive.py: the evaluator must stop killing other lanes' CARLA servers."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "patch_bench2drive", Path(__file__).resolve().parents[1] / "scripts" / "setup" / "patch_bench2drive.py")
pb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pb)

HEAD = "import os, signal, subprocess, atexit\n\nclass LeaderboardEvaluator:\n    def _run(self, args, crashed):\n"
TAIL = "\n        return crashed\n"


def _garage(tmp_path, body):
    f = tmp_path / "Bench2Drive" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    f.parent.mkdir(parents=True)
    f.write_text(HEAD + body + TAIL)
    return f


def test_patch_replaces_gpu_wide_kill_and_is_idempotent(tmp_path):
    f = _garage(tmp_path, pb.OLD)
    assert pb.patch(tmp_path).startswith("patched")
    src = f.read_text()
    assert "graphicsadapter=" not in src.split("MThesis patch")[1]
    assert "psutil.Process(self.server.pid)" in src
    compile(src, str(f), "exec")
    assert pb.patch(tmp_path).startswith("already patched")
    assert f.read_text() == src


def test_patch_refuses_when_upstream_changed(tmp_path):
    _garage(tmp_path, "        if crashed:\n            pass\n")
    with pytest.raises(SystemExit):
        pb.patch(tmp_path)


def test_real_vendored_evaluator_if_present(tmp_path):
    real = Path(__file__).resolve().parents[1] / "Carla-utils" / "carla_garage" / "Bench2Drive" / "leaderboard" / \
        "leaderboard" / "leaderboard_evaluator.py"
    if not real.is_file():
        pytest.skip("vendored carla_garage not present")
    src = real.read_text()
    if pb.MARKER in src:
        pytest.skip("local copy already patched")
    assert pb.OLD in src  # the patch still matches upstream
