"""Test-period error metrics (slide 47, Annexure B2). Error e = actual - forecast."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Scores:
    n: int
    mad: float
    mape: float | None  # None when any actual is zero
    wmape: float | None  # None when actuals sum to zero
    rmse: float
    bias: float
    nbias: float
    ts: float
    mase: float | None

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def rank_value(self) -> float:
        """Value used to rank methods: wMAPE, or MAD when wMAPE is undefined."""
        return self.wmape if self.wmape is not None else self.mad


def naive_scale(train: pd.Series, s: int | None) -> float | None:
    """In-sample MAD of the seasonal-naive (or naive) forecast over the training period."""
    y = train.to_numpy(dtype=float)
    lag = s if s and len(y) > s else 1
    if len(y) <= lag:
        return None
    value = float(np.mean(np.abs(y[lag:] - y[:-lag])))
    return value if value > 0 else None


def score(actual, forecast, scale: float | None = None) -> Scores:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape or len(a) == 0:
        raise ValueError("Actual and forecast must be non-empty and the same length.")
    e = a - f
    abs_e = np.abs(e)
    n = len(a)
    mad = float(abs_e.mean())
    mape = None if (a == 0).any() else float(np.mean(abs_e / np.abs(a)))
    wmape = None if a.sum() == 0 else float(abs_e.sum() / np.abs(a).sum())
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(e.mean())
    nbias = float(e.sum() / abs_e.sum()) if abs_e.sum() > 0 else 0.0
    ts = float(e.sum() / mad) if mad > 0 else 0.0
    mase = float(mad / scale) if scale else None
    return Scores(n, mad, mape, wmape, rmse, bias, nbias, ts, mase)


def bias_direction(bias: float, mad: float) -> tuple[str, str]:
    """Arrow and label for the bias direction (not colour alone)."""
    if mad == 0 or abs(bias) < 0.05 * mad:
        return "●", "unbiased"
    return ("▲", "under-forecasting") if bias > 0 else ("▼", "over-forecasting")
