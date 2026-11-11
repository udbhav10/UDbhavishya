"""Running tracking signal over the test period (slide 47): RSFE / MAD to date."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LIMIT = 4.0


@dataclass
class Tracking:
    signal: pd.Series
    breach: pd.Period | None  # first period beyond +/- LIMIT
    breach_value: float | None

    @property
    def ok(self) -> bool:
        return self.breach is None

    @property
    def verdict(self) -> str:
        if self.ok:
            return f"Within ±{LIMIT:g}: no persistent bias"
        direction = "consistently over-forecasting" if self.breach_value < 0 else "consistently under-forecasting"
        return f"{self.breach}: tracking signal {self.breach_value:+.1f}, {direction}"


def running(actual: pd.Series, forecast: pd.Series) -> Tracking:
    e = (actual - forecast).astype(float)
    rsfe = e.cumsum()
    mad = e.abs().expanding().mean()
    signal = pd.Series(np.where(mad > 0, rsfe / mad.replace(0, np.nan), 0.0), index=actual.index)
    beyond = signal[signal.abs() > LIMIT]
    if beyond.empty:
        return Tracking(signal, None, None)
    return Tracking(signal, beyond.index[0], float(beyond.iloc[0]))
