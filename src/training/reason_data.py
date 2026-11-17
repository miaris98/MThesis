"""TODO A84: the dataset side of the reason labels - per-route lookup at index time, per-sample tensors, and the coverage check.

Switched on by train_wor.py --reason_labels TAG (environment WOR_REASON_LABELS, so DataLoader workers inherit it). Kept out of
wor_dataset.py, which is far over the repository's 500-line limit already; the dataset calls three functions from here.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch

from src.training.reason_labels import label_path, load_labels

# Stored per sample as one small tuple (cause, dist, closing, brake, danger, bucket): ~400k samples, so no per-sample dict.
_NONE = (-1, float("nan"), float("nan"), -1, -1, -1)


def reason_tag() -> str:
    return os.environ.get("WOR_REASON_LABELS", "") or ""


def route_lookup(route_dir: str) -> Optional[Tuple[Dict[str, int], Dict]]:
    """The route's label file as (frame id -> row, columns), or None when labels are off or the route has no file."""
    tag = reason_tag()
    if not tag:
        return None
    path = label_path(route_dir, tag)
    return load_labels(path) if os.path.exists(path) else None


def sample_reason(lookup, frame_id: str):
    """The label tuple of one frame, or None (no labels for this route / frame)."""
    if lookup is None:
        return None
    idx, cols = lookup
    j = idx.get(frame_id)
    if j is None:
        return None
    return (int(cols["cause"][j]), float(cols["dist"][j]), float(cols["closing"][j]), int(cols["brake"][j]), int(cols["danger"][j]),
            int(cols["bucket"][j]))


def reason_tensors(reason) -> Dict[str, torch.Tensor]:
    """Batchable tensors of one sample; frames without a label get -1 / 0 with a zero mask, so the losses skip them."""
    c, d, cl, b, dg, bk = reason if reason is not None else _NONE
    dist_ok = c > 0 and math.isfinite(d)
    close_ok = c > 0 and math.isfinite(cl)
    return {
        "reason_cause": torch.tensor(c, dtype=torch.long),
        "reason_dist": torch.tensor(math.log1p(max(d, 0.0)) if dist_ok else 0.0, dtype=torch.float32),
        "reason_dist_valid": torch.tensor(1.0 if dist_ok else 0.0),
        "reason_closing": torch.tensor(cl if close_ok else 0.0, dtype=torch.float32),
        "reason_closing_valid": torch.tensor(1.0 if close_ok else 0.0),
        "reason_brake": torch.tensor(float(b), dtype=torch.float32),
        "reason_danger": torch.tensor(float(dg), dtype=torch.float32),
        "bucket": torch.tensor(bk, dtype=torch.long),
    }


def check_coverage(samples, min_share: float = 0.5) -> None:
    """With labels switched on, refuse a dataset where most frames have none (a wrong tag would train the heads on nothing)."""
    tag = reason_tag()
    if not tag:
        return
    real = [s for s in samples if isinstance(s, dict)]
    if not real:
        return
    have = sum(1 for s in real if s.get("reason") is not None)
    share = have / len(real)
    print(f"--> reason labels '{tag}': {have}/{len(real)} frames labelled ({100 * share:.1f}%)", flush=True)
    if share < min_share:
        raise RuntimeError(f"only {100 * share:.1f}% of the frames have reason labels '{tag}': run scripts/training/build_reason_labels.py "
                           f"--tag {tag} on this data dir first (or fix the tag)")
