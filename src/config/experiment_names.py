"""Display names of the headline CARLA experiments (alias map, 2026-11-26).

The names live in experiment_names.json next to this file. Old IDs (run labels, E: folders, log entries) are never
renamed; use `display()` wherever a name is shown to a reader (thesis tables, figures, reports).

    from src.config.experiment_names import display, old_id
    display("E15")        -> "QwenPolicy-FE-WP-Overlay-PID"
    display("r1d_r0ftf")  -> "QwenPolicy-FE-WP-NoOverlay-Cov-FreshAdamW-PID"
    old_id("QwenPolicy-FE-WP-NoOverlay-PID") -> "vf_clean"
"""
import json
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).with_name("experiment_names.json")


@lru_cache(maxsize=1)
def _table():
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    by_any = {}
    for e in data["experiments"]:
        for k in (e["old_id"], e["run_label"], e["display"]):
            by_any[k] = e
    return data, by_any


def experiments():
    """All entries, in thesis order."""
    return list(_table()[0]["experiments"])


def display(name: str, default=None) -> str:
    """Display name for an old ID, run label or display name; `default` (or the input) if unknown."""
    e = _table()[1].get(name)
    return e["display"] if e else (name if default is None else default)


def old_id(name: str) -> str:
    """Old ID for a display name, run label or old ID."""
    e = _table()[1].get(name)
    if e is None:
        raise KeyError(f"unknown experiment name: {name!r}")
    return e["old_id"]


def tags():
    """Meaning of every suffix tag."""
    return dict(_table()[0]["tags"])


def short(name: str, default=None) -> str:
    """Short name for figures and narrow tables, e.g. short("E15") -> "QP-OV-PID"."""
    e = _table()[1].get(name)
    return e["short"] if e else (name if default is None else default)


def short_tags():
    """Short tag -> long tag (QP = QwenPolicy-FE-WP)."""
    return dict(_table()[0]["short_tags"])
