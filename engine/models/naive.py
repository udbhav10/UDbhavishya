"""Naive and seasonal naive benchmarks (slide 19)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FitResult


def naive(train: pd.Series) -> FitResult:
    y = train.to_numpy(dtype=float)
    fitted = pd.Series(np.r_[np.nan, y[:-1]], index=train.index)
    last = y[-1]
    return FitResult("naive", train, fitted, forecaster=lambda h: np.full(h, last))


def seasonal_naive(train: pd.Series, s: int | None) -> FitResult:
    """Each future period = the same season in the most recent complete season.
    Falls back to naive when there is no season or less than one full season."""
    if not s or len(train) < s:
        res = naive(train)
        res.extras["fallback"] = True
        res.method = "snaive"
        return res
    y = train.to_numpy(dtype=float)
    fitted = pd.Series(np.r_[np.full(s, np.nan), y[:-s]], index=train.index)
    last_season = y[-s:]

    def forecaster(h: int) -> np.ndarray:
        return np.array([last_season[i % s] for i in range(h)])

    return FitResult("snaive", train, fitted, params={"s": s}, forecaster=forecaster)
