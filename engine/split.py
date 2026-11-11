"""Time-ordered training/test split."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.Series
    test: pd.Series

    @property
    def boundary(self) -> pd.Period:
        """Last training period."""
        return self.train.index[-1]


def default_test_length(n: int, season: int | None, freq_code: str) -> int:
    """One full season, or 20% of the series for yearly/non-seasonal data (at least 1)."""
    if season and freq_code != "Y":
        length = season
    else:
        length = round(0.2 * n)
    length = max(1, length)
    return min(length, max(1, n - 3))  # always leave a few training periods


def split(series: pd.Series, test_length: int) -> Split:
    """Last `test_length` periods form the test set. Order is never changed."""
    if not 1 <= test_length < len(series):
        raise ValueError("Test length must be at least 1 and shorter than the series.")
    series = series.sort_index()
    return Split(series.iloc[:-test_length], series.iloc[-test_length:])
