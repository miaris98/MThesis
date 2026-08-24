"""Online running mean and standard deviation normalizer using Welford's algorithm."""
import numpy as np


class RunningMeanStd:
    """Tracks running mean and variance for online reward normalization."""
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: float) -> None:
        """Update online running mean and variance with a new scalar observation."""
        x = float(x)
        delta = x - self.mean
        self.count += 1
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    @property
    def std(self) -> float:
        """Return running standard deviation clamped to epsilon."""
        return max(float(np.sqrt(self.var)), 1e-4)
