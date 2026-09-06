"""Single source of truth for where experiment output goes.

Three machines write experiment output in this project and each wants a different
root:

  * the vast.ai instance, which keeps the historical ``/workspace`` layout so
    ``run_multi_carla_training.sh`` and ``download_from_vastai.py`` keep working;
  * this PC, which should write onto the external disk (``E:\MThesis_EXP``) so runs
    survive a repo wipe and are not carried around in the working tree;
  * anything else (CI, a laptop without the disk), which falls back to
    ``<repo>/experiments``.

``MTHESIS_EXP_ROOT`` overrides all of it. Every consumer asks this module rather
than hardcoding a path, so pointing the whole stack at a different disk is one
environment variable.
"""
import os
import sys
from pathlib import Path
from typing import Optional

#: Set this to force an experiment root regardless of machine.
ENV_VAR = "MTHESIS_EXP_ROOT"

#: External disk this thesis archives runs onto when it is mounted.
WINDOWS_EXTERNAL_ROOT = Path("E:/MThesis_EXP")

#: Repo root (this file is <repo>/src/config/paths.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The vast.ai layout, kept verbatim so remote tooling is unaffected. ``mlruns``
#: sits under /workspace/MThesis because that is the store the tmux MLflow server
#: in run_multi_carla_training.sh is launched against.
_VASTAI_LAYOUT = {
    "runs": Path("/workspace/runs"),
    "checkpoints": Path("/workspace/checkpoints"),
    "mlruns": Path("/workspace/MThesis/mlruns"),
    "videos": Path("/workspace"),
    "root": Path("/workspace"),
}


def _is_vastai() -> bool:
    """True on the cloud instance, where /workspace is the persistent volume."""
    return os.name != "nt" and Path("/workspace").is_dir()


def _external_disk_available() -> bool:
    """True when the E: drive is mounted, whether or not MThesis_EXP exists yet."""
    return os.name == "nt" and Path(WINDOWS_EXTERNAL_ROOT.anchor).exists()


def exp_root() -> Path:
    """Resolve the experiment root for this machine."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    if _is_vastai():
        return _VASTAI_LAYOUT["root"]
    if _external_disk_available():
        return WINDOWS_EXTERNAL_ROOT
    return REPO_ROOT / "experiments"


def _resolve(kind: str) -> Path:
    """Map a logical output kind onto a concrete directory for this machine."""
    override = os.environ.get(ENV_VAR)
    if not override and _is_vastai():
        return _VASTAI_LAYOUT[kind]
    root = exp_root()
    return root if kind == "root" else root / kind


def runs_dir() -> Path:
    """TensorBoard event files and per-run telemetry CSVs."""
    return _resolve("runs")


def checkpoints_dir() -> Path:
    """Model weights and train_state.json."""
    return _resolve("checkpoints")


def mlruns_dir() -> Path:
    """MLflow file-store backend (metrics, params, artifacts)."""
    return _resolve("mlruns")


def videos_dir() -> Path:
    """Recorded evaluation videos."""
    return _resolve("videos")


def imports_dir() -> Path:
    """Raw snapshots pulled off remote instances by sync_experiments.py."""
    return exp_root() / "imports"


def mlruns_uri() -> str:
    """``mlruns_dir`` as a ``file://`` URI MLflow accepts on Windows and Linux."""
    return path_to_uri(mlruns_dir())


def path_to_uri(path) -> str:
    """Absolute filesystem path -> a file:// URI MLflow accepts on both platforms."""
    return Path(path).resolve().as_uri()


def ensure(*paths: Optional[Path]) -> None:
    """mkdir -p a set of output directories, ignoring None entries."""
    for p in paths:
        if p is not None:
            Path(p).mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable summary of the resolved layout, for startup banners."""
    source = (
        f"${ENV_VAR}" if os.environ.get(ENV_VAR)
        else "vast.ai /workspace" if _is_vastai()
        else "external disk" if _external_disk_available()
        else "repo fallback"
    )
    return (
        f"Experiment root [{source}]: {exp_root()}\n"
        f"  mlruns      : {mlruns_dir()}\n"
        f"  runs        : {runs_dir()}\n"
        f"  checkpoints : {checkpoints_dir()}"
    )


if __name__ == "__main__":
    print(describe())
    sys.exit(0)
