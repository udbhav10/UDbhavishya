"""Seasonal period: default by frequency, checked with autocorrelation (slide 24)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .frequency import DEFAULT_SEASON


@dataclass
class SeasonCheck:
    default: int | None
    acf: pd.Series  # index = lag 1..max_lag
    peak_lag: int | None  # highest local ACF peak (lag >= 2), if clear
    threshold: float  # rough significance line, 2 / sqrt(n)

    @property
    def default_is_peak(self) -> bool:
        """The ACF at the default lag (±1) is a significant local peak."""
        s = self.default
        if not s or s + 1 > len(self.acf):
            return False
        r = self.acf
        for k in (s - 1, s, s + 1):
            if 2 <= k < len(r) and r[k] > self.threshold and r[k] >= r[k - 1] and r[k] >= r[k + 1]:
                return True
        return False

    @property
    def agrees(self) -> bool:
        if self.default is None:
            return False
        p = self.peak_lag
        if p is not None and p < self.default and self.default % p == 0:
            return False  # a shorter cycle that fits inside the default explains the data better
        if self.default_is_peak:
            return True
        return p is not None and p % self.default == 0

    @property
    def suggestion(self) -> int | None:
        """A different s to suggest, when the ACF clearly disagrees with the default."""
        if self.peak_lag is None or self.agrees:
            return None
        return self.peak_lag


def acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Correlation of the series with itself shifted by k periods, using the
    periods where both have values (Excel =CORREL, slide 24)."""
    x = np.asarray(values, dtype=float)
    out = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        a, b = x[k:], x[:-k]
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            continue
        out[k - 1] = float(np.corrcoef(a, b)[0, 1])
    return out


def check(train: pd.Series, freq_code: str) -> SeasonCheck:
    """ACF of the linearly detrended training data and its highest local peak."""
    default = DEFAULT_SEASON[freq_code]
    y = train.to_numpy(dtype=float)
    n = len(y)
    span = 2 * default if default else 24
    max_lag = max(1, min(span + 2, n // 2))
    t = np.arange(n)
    slope, intercept = np.polyfit(t, y, 1) if n > 2 else (0.0, y.mean())
    detrended = y - (intercept + slope * t)
    r = acf(detrended, max_lag)
    threshold = 2 / np.sqrt(n) if n else 1.0
    # Local peaks above the significance line; report the first one that is nearly
    # as strong as the strongest (multiples of the season also peak).
    peaks = [(k, r[k - 1]) for k in range(2, max_lag)
             if r[k - 1] > r[k - 2] and r[k - 1] >= r[k] and r[k - 1] > threshold]
    peak = None
    if peaks:
        best = max(v for _, v in peaks)
        peak = min(k for k, v in peaks if v >= 0.9 * best)
    return SeasonCheck(default, pd.Series(r, index=np.arange(1, max_lag + 1)), peak, float(threshold))


def enough_seasons(train_length: int, season: int | None) -> bool:
    return bool(season) and train_length >= 2 * season
