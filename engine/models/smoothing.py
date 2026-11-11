"""Exponential smoothing: SES, Holt (optional damping) and multiplicative
Holt-Winters (optional damping). Slides 32-43.

Indexing: start values describe time 0 (just before the first training period).
For each training period t = 1..n the one-step forecast is made first, then the
states are updated with the actual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..init import StartValues, start_values
from .base import FitResult, damped_sum


def ses(train: pd.Series, alpha: float, start: StartValues | None = None) -> FitResult:
    y = train.to_numpy(dtype=float)
    start = start or start_values(y)
    level = start.level
    fitted = np.empty(len(y))
    levels = np.empty(len(y))
    for t, a in enumerate(y):
        fitted[t] = level
        level = alpha * a + (1 - alpha) * level
        levels[t] = level
    final = level
    return FitResult("ses", train, pd.Series(fitted, index=train.index), {"alpha": alpha}, start,
                     {"level": pd.Series(levels, index=train.index)},
                     forecaster=lambda h: np.full(h, final))


def holt(train: pd.Series, alpha: float, beta: float, phi: float = 1.0,
         start: StartValues | None = None) -> FitResult:
    y = train.to_numpy(dtype=float)
    start = start or start_values(y, trend=True)
    level, trend = start.level, start.trend
    n = len(y)
    fitted, levels, trends = np.empty(n), np.empty(n), np.empty(n)
    for t, a in enumerate(y):
        fitted[t] = level + phi * trend
        new_level = alpha * a + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
        levels[t], trends[t] = level, trend
    L, T = level, trend
    idx = train.index
    return FitResult("holt", train, pd.Series(fitted, index=idx),
                     {"alpha": alpha, "beta": beta, "phi": phi}, start,
                     {"level": pd.Series(levels, index=idx), "trend": pd.Series(trends, index=idx)},
                     forecaster=lambda h: L + damped_sum(phi, h) * T)


def winters(train: pd.Series, s: int, alpha: float, beta: float, gamma: float, phi: float = 1.0,
            start: StartValues | None = None) -> FitResult:
    y = train.to_numpy(dtype=float)
    if (y <= 0).any():
        raise ValueError("Multiplicative seasonality needs demand above zero.")
    start = start or start_values(y, s=s, trend=True, seasonal=True)
    level, trend = start.level, start.trend
    season = np.array(start.season, dtype=float)
    n = len(y)
    fitted, levels, trends, seasons = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for t, a in enumerate(y):
        pos = t % s
        fitted[t] = (level + phi * trend) * season[pos]
        new_level = alpha * a / season[pos] + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
        season[pos] = gamma * a / level + (1 - gamma) * season[pos]
        levels[t], trends[t], seasons[t] = level, trend, season[pos]
    L, T, S = level, trend, season.copy()

    def forecaster(h: int) -> np.ndarray:
        positions = (n + np.arange(h)) % s
        return (L + damped_sum(phi, h) * T) * S[positions]

    idx = train.index
    return FitResult("winters", train, pd.Series(fitted, index=idx),
                     {"alpha": alpha, "beta": beta, "gamma": gamma, "phi": phi, "s": s}, start,
                     {"level": pd.Series(levels, index=idx), "trend": pd.Series(trends, index=idx),
                      "season": pd.Series(seasons, index=idx)},
                     forecaster=forecaster)
