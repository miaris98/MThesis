"""Simulate the B66/B67 sampler: how many reanalysis searches a per-target-window cache would need, and in-batch duplicates."""
import numpy as np
rng = np.random.default_rng(0)
E, B, K, TD = 4, 256, 5, 5
HOR = K + TD
def window_unique(size_rows, W):
    n = size_rows - HOR
    flat = rng.integers(0, n * E, size=W * B)          # replay.sample, alpha 0: uniform over (t, e) window starts
    t, e = flat // E, flat % E
    roots = ((t[:, None] + np.arange(K + 1)) * E + e[:, None]).reshape(-1)
    return np.unique(roots).size, roots.size
def batch_dup(size_rows):
    n = size_rows - HOR
    flat = rng.integers(0, n * E, size=B); t, e = flat // E, flat % E
    roots = ((t[:, None] + np.arange(K + 1)) * E + e[:, None]).reshape(-1)
    return 1 - np.unique(roots).size / roots.size
for W in (200, 400):
    fresh = draws = 0; rows = []
    for u0 in range(0, 120_000, W):
        trans = min(100_000, 2_000 + u0)                 # replay ratio 1: one update per transition, then 22k offline updates
        uq, dr = window_unique(trans // E, W)
        fresh += uq; draws += dr
        if u0 in (0, 8_000, 18_000, 48_000, 98_000, 119_800) or (W == 400 and u0 in (7_600, 17_600, 47_600, 97_600, 119_600)):
            N = trans; D = 6 * B * W; f = (N / D) * (1 - np.exp(-D / N))
            rows.append(f"  update {u0:>6} buffer {trans:>6}: fresh {uq / dr:6.1%}   formula (N/D)(1-e^-D/N) {f:6.1%}")
    print(f"W = {W}: run-average fraction of searches needed with a per-window cache = {fresh / draws:.3f}  (speed-up of the search x{draws / fresh:.1f})")
    print("\n".join(rows))
print("in-batch duplicate roots:", ", ".join(f"buffer {N}: {np.mean([batch_dup(N // E) for _ in range(200)]):.1%}" for N in (5_000, 20_000, 50_000, 100_000)))
