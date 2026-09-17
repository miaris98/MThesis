"""`run_carla_evaluation` (eval_wor.py) and `run_route` (eval_wor_closed_loop.py) each spawn
their own agent-facing camera instead of calling `agent.sensors()` - the one place the
camera-parity fix (11.4) actually lives, per `WorldOnRailsAgent.sensors()`'s own docstring.

Both were still spawning the *exact* pre-fix geometry that docstring names as already
resolved: `x=1.3, z=1.3, fov=100, 256x256`, against the true data-collection camera at
`x=-1.5, z=2.0, fov=110, 1024x512`. Neither file imported `PDM_LITE_CAMERA` at all. Every
video `eval_wor.py` ever recorded, and every route score `eval_wor_closed_loop.py` ever
produced, was measured through the wrong camera - `agent.run_step`'s own `preprocess_rgb`
crop/resize is only the other half of parity and cannot correct a frame already rendered
from the wrong pose.

Both files hard-import `carla` (or things that transitively require it) at module scope, so
they can't be imported in this environment - checked via source inspection instead, the same
approach `test_route_lookahead_parity.py` uses and for the same reason.
"""
import re

from pathlib import Path

from src.config.camera import PDM_LITE_CAMERA


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    """Read a source file addressed relative to the repository root."""
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_eval_wor_imports_the_canonical_camera():
    assert "from src.config.camera import" in _read("scripts/eval/eval_wor.py")
    assert "PDM_LITE_CAMERA" in _read("scripts/eval/eval_wor.py")


def test_eval_wor_closed_loop_imports_the_canonical_camera():
    assert "PDM_LITE_CAMERA" in _read("scripts/eval/eval_wor_closed_loop.py")


def test_neither_file_hardcodes_the_old_broken_geometry():
    """The literal values from the pre-fix camera - if any of these reappear as a spawn
    parameter, someone has reverted to the broken sensor by hand."""
    for path in ("scripts/eval/eval_wor.py", "scripts/eval/eval_wor_closed_loop.py"):
        text = _read(path)
        assert '"image_size_x", "256"' not in text, f"{path} still spawns a 256px-wide camera"
        assert '"fov", "100"' not in text, f"{path} still spawns a 100deg-fov camera"
        assert "carla.Location(x=1.3, z=1.3)" not in text, (
            f"{path} still spawns the camera at the pre-parity-fix pose")


def test_reshape_targets_are_no_longer_the_stale_256_square():
    for path in ("scripts/eval/eval_wor.py", "scripts/eval/eval_wor_closed_loop.py"):
        text = _read(path)
        assert ".reshape((256, 256, 4))" not in text, (
            f"{path} still reshapes the raw buffer as a 256x256 image")


def test_camera_spawn_parameters_are_read_from_pdm_lite_camera():
    """Pins the actual mechanism, not just the absence of the old literals: the width/height/fov
    passed to set_attribute must trace to PDM_LITE_CAMERA, so a future edit that reads from some
    other, possibly-drifted source is still caught."""
    for path in ("scripts/eval/eval_wor.py", "scripts/eval/eval_wor_closed_loop.py"):
        text = _read(path)
        assert re.search(r'set_attribute\("image_size_x",\s*str\(PDM_LITE_CAMERA\["width"\]\)\)',
                         text), f"{path}'s camera width is not sourced from PDM_LITE_CAMERA"
        assert re.search(r'set_attribute\("image_size_y",\s*str\(PDM_LITE_CAMERA\["height"\]\)\)',
                         text), f"{path}'s camera height is not sourced from PDM_LITE_CAMERA"
        assert re.search(r'set_attribute\("fov",\s*str\(PDM_LITE_CAMERA\["fov"\]\)\)',
                         text), f"{path}'s camera fov is not sourced from PDM_LITE_CAMERA"


def test_pdm_lite_camera_itself_is_the_expected_geometry():
    """Sanity check on the source of truth these files now defer to, so a change there is
    visible here too rather than only in src/config/camera.py's own tests."""
    assert PDM_LITE_CAMERA["x"] == -1.5
    assert PDM_LITE_CAMERA["z"] == 2.0
    assert PDM_LITE_CAMERA["fov"] == 110
    assert (PDM_LITE_CAMERA["width"], PDM_LITE_CAMERA["height"]) == (1024, 512)
