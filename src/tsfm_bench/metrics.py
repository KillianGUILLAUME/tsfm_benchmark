"""Scoring rules and tests for quantile forecasts."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .config import LEVELS


def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """Pinball (quantile) loss averaged over the quantile levels; y has shape [n], q has shape [n, K].

    The mean of the pinball loss over a grid of levels is a discretised CRPS (up to a factor 2)."""
    diff = y[:, None] - q
    return np.maximum(levels[None, :] * diff, (levels[None, :] - 1.0) * diff).mean(axis=1)


def central_coverage(y: np.ndarray, q: np.ndarray) -> float:
    """Share of observations inside the central interval [q_first, q_last] (nominal 80% for levels 0.1..0.9)."""
    return float(np.mean((y >= q[:, 0]) & (y <= q[:, -1])))


def calibration(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Empirical frequency of y <= q_k for every level k (equals the level when calibrated)."""
    return (y[:, None] <= q).mean(axis=0)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 5) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy with a Newey-West (Bartlett) long-run variance.

    Returns (mean loss differential a - b, two-sided p-value); a negative mean means that a is better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    m = d.mean()
    dc = d - m
    var = float(dc @ dc) / n
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        var += 2.0 * w * float(dc[j:] @ dc[:-j]) / n
    if var <= 0:
        return float(m), float("nan")
    stat = m / np.sqrt(var / n)
    return float(m), float(2.0 * (1.0 - norm.cdf(abs(stat))))
