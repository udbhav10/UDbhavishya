"""One start-value method for every smoothing model.

Based on the heuristic in Hyndman, Koehler, Ord & Snyder (2008), Forecasting with
Exponential Smoothing, section 2.6.1 (also the starting point used by R's
forecast::ets and Python's statsmodels "heuristic" initialisation):

1. Seasonal indices (seasonal models only): centred moving average over the first
   two seasons of training data; ratio = actual / moving average; one ratio per
   position in the season; scaled so the indices average 1.00.
2. Deseasonalise the data (actual / index), or use it as is.
3. Fit a straight line to the first ten (deseasonalised) values, time 1..10:
   intercept -> starting level L0, slope -> starting trend T0.
   Models without a trend use a flat line: L0 = mean of those values, T0 = 0.

Only the training data is ever passed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

WARMUP_POINTS = 10


@dataclass
class StartValues:
    level: float
    trend: float = 0.0
    season: np.ndarray | None = None  # index per season position; position 0 = first training period
    notes: dict[str, str] = field(default_factory=dict)  # how each value was found

    def with_overrides(self, level=None, trend=None, season=None) -> "StartValues":
        return replace(
            self,
            level=self.level if level is None else float(level),
            trend=self.trend if trend is None else float(trend),
            season=self.season if season is None else np.asarray(season, dtype=float),
        )


def centred_moving_average(y: np.ndarray, s: int) -> np.ndarray:
    """2 x s moving average for even s, s moving average for odd s. NaN where undefined."""
    n = len(y)
    out = np.full(n, np.nan)
    if s % 2 == 0:
        half = s // 2
        weights = np.r_[0.5, np.ones(s - 1), 0.5] / s
        for t in range(half, n - half):
            out[t] = float(np.dot(weights, y[t - half: t + half + 1]))
    else:
        half = (s - 1) // 2
        for t in range(half, n - half):
            out[t] = float(y[t - half: t + half + 1].mean())
    return out


def seasonal_indices(y: np.ndarray, s: int) -> np.ndarray:
    if len(y) < 2 * s:
        raise ValueError(f"Need at least two seasons ({2 * s} periods) to compute seasonal indices.")
    warm = y[: 2 * s]
    ma = centred_moving_average(warm, s)
    sums = np.zeros(s)
    counts = np.zeros(s)
    for t in range(len(warm)):
        if not np.isnan(ma[t]) and ma[t] != 0:
            sums[t % s] += warm[t] / ma[t]
            counts[t % s] += 1
    raw = np.where(counts > 0, sums / np.maximum(counts, 1), 1.0)
    return raw / raw.mean()


def start_values(y, s: int | None = None, trend: bool = False, seasonal: bool = False) -> StartValues:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        raise ValueError("No training data.")
    notes: dict[str, str] = {}
    season = None
    adjusted = y
    if seasonal:
        season = seasonal_indices(y, s)
        notes["season"] = (f"Average of actual ÷ centred {s}-period moving average for each position, "
                           f"over the first {2 * s} training periods; scaled to average 1.00.")
        adjusted = y / season[np.arange(len(y)) % s]
    k = min(WARMUP_POINTS, len(adjusted))
    first = adjusted[:k]
    what = "deseasonalised values" if seasonal else "values"
    if trend and k >= 2:
        t = np.arange(1, k + 1)
        slope, intercept = np.polyfit(t, first, 1)
        level, tr = float(intercept), float(slope)
        notes["level"] = f"Intercept of a straight line fitted to the first {k} training {what}."
        notes["trend"] = f"Slope of the same line: the average change per period."
    else:
        level, tr = float(first.mean()), 0.0
        notes["level"] = f"Average of the first {k} training {what}."
    return StartValues(level, tr, season, notes)
