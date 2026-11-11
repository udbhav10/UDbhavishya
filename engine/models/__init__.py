"""Forecasting methods and their availability rules."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import METHOD_NAMES, FitResult  # noqa: F401

ORDER = ("naive", "snaive", "ses", "holt", "winters", "regression")


@dataclass
class Availability:
    available: bool
    reason: str = ""


def availability(train: pd.Series, s: int | None, has_driver: bool) -> dict[str, Availability]:
    n = len(train)
    out = {
        "naive": Availability(n >= 1),
        "snaive": Availability(n >= 1, "" if s and n >= s else "No seasonal pattern: same as naive"),
        "ses": Availability(n >= 2, "" if n >= 2 else "Needs at least 2 periods"),
        "holt": Availability(n >= 3, "" if n >= 3 else "Needs at least 3 periods"),
    }
    if not s:
        out["winters"] = Availability(False, "No seasonal period for this frequency")
    elif n < 2 * s:
        out["winters"] = Availability(False, f"Needs at least two seasons ({2 * s} periods) of training data")
    elif (train <= 0).any():
        out["winters"] = Availability(False, "Multiplicative seasonality needs demand above zero")
    else:
        out["winters"] = Availability(True)
    if not has_driver:
        out["regression"] = Availability(False, "No driver column mapped")
    elif n < 3:
        out["regression"] = Availability(False, "Needs at least 3 periods")
    else:
        out["regression"] = Availability(True)
    return out
