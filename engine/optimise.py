"""Parameter tuning on training data only: coarse grid, then bounded L-BFGS-B
minimising the sum of squared one-step-ahead training errors. Start values stay fixed."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .init import StartValues
from .models import smoothing

BOUNDS = {"alpha": (0.01, 0.99), "beta": (0.01, 0.99), "gamma": (0.01, 0.99), "phi": (0.80, 1.00)}
GRID = np.round(np.arange(0.1, 1.0, 0.1), 2)
PHI_GRID = (0.8, 0.9, 0.98)


def _names(method: str, damped: bool) -> list[str]:
    names = {"ses": ["alpha"], "holt": ["alpha", "beta"], "winters": ["alpha", "beta", "gamma"]}[method]
    return names + (["phi"] if damped and method != "ses" else [])


def _sse(method, train, s, start, params) -> float:
    if method == "ses":
        fit = smoothing.ses(train, params["alpha"], start)
    elif method == "holt":
        fit = smoothing.holt(train, params["alpha"], params["beta"], params.get("phi", 1.0), start)
    else:
        fit = smoothing.winters(train, s, params["alpha"], params["beta"], params["gamma"],
                                params.get("phi", 1.0), start)
    return fit.sse


def optimise(method: str, train: pd.Series, start: StartValues, s: int | None = None,
             damped: bool = False) -> dict[str, float]:
    names = _names(method, damped)
    grids = [PHI_GRID if n == "phi" else GRID for n in names]
    best, best_sse = None, np.inf
    for combo in itertools.product(*grids):
        value = _sse(method, train, s, start, dict(zip(names, combo)))
        if value < best_sse:
            best, best_sse = combo, value

    def objective(vec):
        return _sse(method, train, s, start, dict(zip(names, vec)))

    result = minimize(objective, np.array(best, dtype=float), method="L-BFGS-B",
                      bounds=[BOUNDS[n] for n in names])
    chosen = result.x if result.fun <= best_sse else np.array(best, dtype=float)
    out = {n: float(np.clip(v, *BOUNDS[n])) for n, v in zip(names, chosen)}
    if method != "ses" and not damped:
        out["phi"] = 1.0
    return out
