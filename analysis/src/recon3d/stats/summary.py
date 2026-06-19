"""Summary-statistics helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def calc_ci(
    x: np.ndarray | Sequence[float],
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate a t-distribution confidence interval."""
    values = np.asarray(x, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    n = len(values)
    alpha = 1 - confidence_level
    t_value = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = t_value * (std / np.sqrt(n))
    return mean - margin, mean + margin
