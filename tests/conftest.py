"""Put the repository root on `sys.path` for the whole test session.

Tests import both `src.*` and the operational entry points under `scripts.*`; both
are resolved relative to the repository root rather than to `tests/`, so the root has
to be importable no matter which directory pytest is invoked from.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
