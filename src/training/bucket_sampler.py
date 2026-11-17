"""TODO A84: re-balance training frames over the reason-label buckets (src/training/reason_labels.BUCKETS).

Uniform sampling spends almost every step on cruising; the frames that decide collisions - a brake onset, a yield, a pedestrian, a
start from a stop - are a few percent each. CarLLaVA / SimLingo sample from such buckets; this is the same idea with buckets built
from our own logs. A frame in bucket b is drawn with weight n_b ** -alpha (alpha 0 = uniform, 1 = every non-empty bucket equally
often), the cruise bucket keeps at least `min_cruise` of the draws so ordinary driving is not forgotten (vf_ctrl's lesson), draws are
with replacement and the epoch keeps its length, so schedules and epoch counts stay comparable to the uniform arms. Frames without a
label (bucket -1) are treated as cruise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler

from src.training.seeding import make_generator


def bucket_weights(buckets: np.ndarray, alpha: float, min_cruise: float = 0.3, n_buckets: Optional[int] = None) -> np.ndarray:
    """Per-frame draw probabilities (sum 1). Bucket 0 is cruise; -1 counts as cruise."""
    b = np.where(np.asarray(buckets) < 0, 0, np.asarray(buckets)).astype(int)
    k = int(n_buckets or (b.max() + 1 if len(b) else 1))
    counts = np.bincount(b, minlength=k).astype(float)
    mass = np.where(counts > 0, counts ** (1.0 - alpha), 0.0)                       # bucket mass = n_b * n_b^-alpha
    mass /= mass.sum()
    # The floor only stops cruise from being starved: it never lifts cruise above its natural share. On the full A84 data cruise is 10.7% of
    # the frames (a vehicle binds the expert's speed on 59%), and a flat 0.3 floor tripled cruise and halved the danger / brake draws (2026-11-17).
    min_cruise = min(min_cruise, counts[0] / max(1, len(b)))
    if counts[0] > 0 and mass[0] < min_cruise and counts[0] < len(b):
        rest = mass[1:].sum()
        mass[1:] *= (1.0 - min_cruise) / rest
        mass[0] = min_cruise
    w = mass[b] / counts[b]
    return w / w.sum()


def bucket_sampler(dataset, alpha: float, seed: Optional[int] = None, min_cruise: float = 0.3) -> WeightedRandomSampler:
    from src.training.reason_labels import BUCKETS
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    idx = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    buckets = np.array([(base.samples[i].get("reason") or (0, 0, 0, 0, 0, -1))[5] if isinstance(base.samples[i], dict) else -1 for i in idx])
    w = bucket_weights(buckets, alpha, min_cruise, n_buckets=len(BUCKETS))
    b = np.where(buckets < 0, 0, buckets)
    nat = np.bincount(b, minlength=len(BUCKETS)) / max(1, len(b))
    drawn = np.bincount(b, weights=w, minlength=len(BUCKETS))
    print(f"--> bucket sampler (alpha {alpha}, cruise floor {min_cruise}): " + "  ".join(
        f"{n} {100 * p:.1f}%->{100 * q:.1f}%" for n, p, q in zip(BUCKETS, nat, drawn) if p > 0), flush=True)
    gen = make_generator(seed) if seed is not None else None
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True, generator=gen)
