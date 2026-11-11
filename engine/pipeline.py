"""Fit any method by name with its parameters and start-value overrides."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import init
from .models import FitResult, naive, regression, smoothing

DEFAULT_PARAMS = {
    "ses": {"alpha": 0.3},
    "holt": {"alpha": 0.3, "beta": 0.1, "phi": 1.0},
    "winters": {"alpha": 0.3, "beta": 0.1, "gamma": 0.1, "phi": 1.0},
    "regression": {"trend": False, "seasonal": False},
}


@dataclass
class MethodSettings:
    params: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)  # level / trend / season


def computed_start(method: str, train: pd.Series, s: int | None) -> init.StartValues | None:
    y = train.to_numpy(dtype=float)
    if method == "ses":
        return init.start_values(y)
    if method == "holt":
        return init.start_values(y, trend=True)
    if method == "winters":
        return init.start_values(y, s=s, trend=True, seasonal=True)
    return None


def fit_method(method: str, train: pd.Series, s: int | None, settings: MethodSettings | None = None,
               driver: pd.Series | None = None, driver_name: str = "driver") -> FitResult:
    settings = settings or MethodSettings()
    p = {**DEFAULT_PARAMS.get(method, {}), **settings.params}
    if method == "naive":
        return naive.naive(train)
    if method == "snaive":
        return naive.seasonal_naive(train, s)
    if method == "regression":
        if driver is None:
            raise ValueError("Regression needs a driver.")
        return regression.regression(train, driver, p["trend"], p["seasonal"], s, driver_name)
    start = computed_start(method, train, s)
    if settings.overrides:
        start = start.with_overrides(**settings.overrides)
    if method == "ses":
        return smoothing.ses(train, p["alpha"], start)
    if method == "holt":
        return smoothing.holt(train, p["alpha"], p["beta"], p["phi"], start)
    if method == "winters":
        return smoothing.winters(train, s, p["alpha"], p["beta"], p["gamma"], p["phi"], start)
    raise ValueError(f"Unknown method: {method}")


def forecast_ahead(fit: FitResult, h: int, driver_future: pd.Series | np.ndarray | None = None) -> pd.Series:
    """Forecast h periods from the end of the fitted data (origin = last fitted period)."""
    if fit.method == "regression":
        values = driver_future.to_numpy() if isinstance(driver_future, pd.Series) else driver_future
        return fit.forecast(h, x_future=values)
    return fit.forecast(h)
