"""The vastai CLI for the scripts that run on the PC (fleet listing, credit checks, instance selection).

One lookup instead of a per-script path: ``$VAST_CLI`` (or the older ``$VAST``), else ``vastai`` on PATH, else a conda
install under the home folder (pip puts ``vastai.exe`` in ``<env>/Scripts`` on Windows, which is not on Git Bash's PATH).
The shell side is ``vast_cli`` in scripts/lib/pc.sh, with the same order.
"""
import glob
import os
import shutil
from pathlib import Path

_CONDA_PATTERNS = (
    "anaconda3/envs/*/Scripts/vastai.exe",
    "anaconda3/Scripts/vastai.exe",
    "miniconda3/envs/*/Scripts/vastai.exe",
    "miniconda3/Scripts/vastai.exe",
)


def vast_cli(home: Path = None) -> str:
    """Path of the vastai executable; FileNotFoundError when there is none."""
    env = os.environ.get("VAST_CLI") or os.environ.get("VAST")
    if env:
        return env
    found = shutil.which("vastai")
    if found:
        return found
    home = Path(home) if home is not None else Path.home()
    for pattern in _CONDA_PATTERNS:
        hits = sorted(glob.glob(str(home / pattern)))
        if hits:
            return hits[0]
    raise FileNotFoundError("vastai CLI not found: set VAST_CLI or put vastai on PATH")
