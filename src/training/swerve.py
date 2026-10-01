"""Swerve frames and their sampler (TODO A16, arm K = arm J + moderate obstacle oversampling).

A swerve frame is one where PDM-Lite's expert bent its `route` around an obstacle - the first route points (what the
policy sees) sit more than SWERVE_SHIFT_M laterally off the leaderboard plan `route_original`, the measure of
scripts/analysis/route_shift_stats.py - or one of the WOR_SWERVE_WINDOW frames before it (the approach).
WorldOnRailsDataset marks them; the train loader oversamples them when WOR_SWERVE_FRAC > 0 (train_wor.py --swerve_frac).
"""
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler

from src.training.seeding import make_generator

SWERVE_SHIFT_M = 0.5


def route_shift(route, route_original, points: int = 4) -> float:
    """Largest lateral (ego y, metres) gap between the shifted route and the plan over the first `points` points."""
    if not route or not route_original:
        return 0.0
    n = min(points, len(route), len(route_original))
    if n == 0:
        return 0.0
    return float(np.abs(np.asarray(route[:n], float)[:, 1] - np.asarray(route_original[:n], float)[:, 1]).max())


def swerve_flags(shifts, window: int) -> np.ndarray:
    """Per-frame shift (metres, None for unreadable frames) -> bool mask of shifted frames plus `window` frames before."""
    shifted = np.array([s is not None and s > SWERVE_SHIFT_M for s in shifts], dtype=bool)
    flags = np.zeros(len(shifted), dtype=bool)
    for j in np.flatnonzero(shifted):
        flags[max(0, j - window):j + 1] = True
    return flags


def swerve_sampler(dataset, frac: float, seed: Optional[int] = None) -> WeightedRandomSampler:
    """Draws swerve frames as `frac` of the epoch, everything else uniformly.

    Moderate on purpose (the TODO's ~10%): training mostly on obstacle frames risks false overtakes (swerving
    around queued traffic, head-on collisions in the TwoWays variants) and forgetting other skills. Draws are
    with replacement and the epoch keeps its length, so the LR schedule and epoch counts stay comparable to arm J.
    Never shrinks swerve frames below their natural share.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    idx = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    flag = np.array([bool(isinstance(base.samples[i], dict) and base.samples[i].get("swerve")) for i in idx])
    n, k = len(flag), int(flag.sum())
    if k == 0 or k == n or k / n >= frac:
        w = np.ones(n)
        note = "natural share already at or above target" if k else "no swerve frames found"
    else:
        w = np.where(flag, frac / k, (1.0 - frac) / (n - k))
        note = f"x{(frac / k) / ((1.0 - frac) / (n - k)):.1f} weight on swerve frames"
    print(f"--> Swerve sampler: {k}/{n} train frames flagged ({100 * k / max(n, 1):.1f}%), "
          f"target {100 * frac:.0f}% of draws ({note}).", flush=True)
    gen = make_generator(seed) if seed is not None else None
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=n, replacement=True,
                                 generator=gen)
