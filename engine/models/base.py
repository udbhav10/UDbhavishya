"""Shared result type for all forecasting models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..init import StartValues

METHOD_NAMES = {
    "naive": "Naive",
    "snaive": "Seasonal naive",
    "ses": "SES",
    "holt": "Holt",
    "winters": "Winters",
    "regression": "Regression",
}


@dataclass
class FitResult:
    method: str
    train: pd.Series
    fitted: pd.Series  # one-step-ahead forecasts over training (NaN where none exist)
    params: dict = field(default_factory=dict)
    start: StartValues | None = None
    components: dict[str, pd.Series] = field(default_factory=dict)
    forecaster: Callable[[int], np.ndarray] | None = None
    extras: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return METHOD_NAMES.get(self.method, self.method)

    def future_index(self, h: int) -> pd.PeriodIndex:
        last = self.train.index[-1]
        return pd.PeriodIndex([last + i for i in range(1, h + 1)], freq=self.train.index.freq)

    def forecast(self, h: int, **kwargs) -> pd.Series:
        values = self.forecaster(h, **kwargs) if kwargs else self.forecaster(h)
        return pd.Series(np.asarray(values, dtype=float), index=self.future_index(h), name=self.name)

    @property
    def residuals(self) -> pd.Series:
        return self.train - self.fitted

    @property
    def sse(self) -> float:
        r = self.residuals.dropna()
        return float((r ** 2).sum())


def damped_sum(phi: float, h: int) -> np.ndarray:
    """phi + phi^2 + ... + phi^k for k = 1..h."""
    return np.cumsum(phi ** np.arange(1, h + 1))
