"""Confidence range around a forecast: F x (1 +/- z x r), r = 1.25 x MAPE (slide 74)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

LEVELS = (20, 60, 80)


def z_value(level: int) -> float:
    """Two-sided z for a central range covering `level` percent: 20 -> 0.253, 60 -> 0.842, 80 -> 1.282."""
    if level not in LEVELS:
        raise ValueError(f"Range level must be one of {LEVELS}")
    return float(norm.ppf(0.5 + level / 200))


@dataclass
class RangeBasis:
    r: float  # relative spread (fraction)
    source: str  # "MAPE" or "wMAPE"
    error: float  # the test error the spread is based on (fraction)


def basis(mape: float | None, wmape: float | None) -> RangeBasis:
    if mape is not None:
        return RangeBasis(1.25 * mape, "MAPE", mape)
    if wmape is not None:
        return RangeBasis(1.25 * wmape, "wMAPE", wmape)
    raise ValueError("A percentage error is needed to build a range.")


def build(forecast: pd.Series, r: float, level: int) -> pd.DataFrame:
    z = z_value(level)
    f = forecast.astype(float)
    lower = np.maximum(0.0, f * (1 - z * r))
    upper = f * (1 + z * r)
    return pd.DataFrame({"forecast": f, "lower": lower, "upper": upper}, index=forecast.index)
