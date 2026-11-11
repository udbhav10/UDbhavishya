"""Linear regression on one causal driver, with optional trend and seasonal dummies
(slides 60-70). Least squares, with the statistics Excel's regression tool reports."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .base import FitResult


@dataclass
class RegressionStats:
    names: list[str]
    coef: np.ndarray
    std_err: np.ndarray
    t_stat: np.ndarray
    r2: float
    adj_r2: float
    se: float  # standard error of the regression
    n: int
    df: int

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names, "coefficient": self.coef,
                             "std. error": self.std_err, "t-stat": self.t_stat})

    @property
    def obs_per_coef(self) -> float:
        return self.n / len(self.coef)


def _season_label(period: pd.Period, freq: str) -> str:
    if freq.startswith("M"):
        return period.strftime("%b")
    if freq.startswith("Q"):
        return f"Q{period.quarter}"
    if freq.startswith("D"):
        return period.strftime("%a")
    return f"position {period}"


def _design(n: int, x: np.ndarray, offset: int, trend: bool, s: int | None) -> np.ndarray:
    cols = [np.ones(n), x]
    if trend:
        cols.append(np.arange(offset + 1, offset + n + 1, dtype=float))
    if s:
        pos = (offset + np.arange(n)) % s
        for p in range(s - 1):  # last position is the base
            cols.append((pos == p).astype(float))
    return np.column_stack(cols)


def regression(train: pd.Series, driver: pd.Series, trend: bool = False, seasonal: bool = False,
               s: int | None = None, driver_name: str = "driver") -> FitResult:
    y = train.to_numpy(dtype=float)
    x = driver.reindex(train.index).to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("Driver values are missing for some training periods.")
    s_used = s if (seasonal and s) else None
    X = _design(len(y), x, 0, trend, s_used)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid = y - fitted
    n, k = X.shape
    df = n - k
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 0.0
    se = float(np.sqrt(sse / df)) if df > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / df if df > 0 else float("nan")
    try:
        cov = se ** 2 * np.linalg.inv(X.T @ X)
        std_err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std_err = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = coef / std_err

    names = ["Intercept", driver_name]
    if trend:
        names.append("Trend (per period)")
    if s_used:
        freq = train.index.freqstr
        labels = [_season_label(train.index[p], freq) if p < len(train) else f"position {p}" for p in range(s_used)]
        names += [f"Season: {labels[p]}" for p in range(s_used - 1)]
    stats_ = RegressionStats(names, coef, std_err, t_stat, r2, adj, se, n, df)
    n_train = len(y)

    def forecaster(h: int, x_future=None) -> np.ndarray:
        if x_future is None:
            raise ValueError("Future driver values are needed for a regression forecast.")
        xf = np.asarray(x_future, dtype=float)[:h]
        if len(xf) < h or np.isnan(xf).any():
            raise ValueError("Future driver values are needed for every forecast period.")
        return _design(h, xf, n_train, trend, s_used) @ coef

    params = {"trend": trend, "seasonal": bool(s_used)}
    return FitResult("regression", train, pd.Series(fitted, index=train.index), params,
                     forecaster=forecaster, extras={"stats": stats_, "t_crit": float(stats.t.ppf(0.975, df))
                                                    if df > 0 else float("nan")})
