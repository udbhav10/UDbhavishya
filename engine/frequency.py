"""Frequency detection, period alignment and roll-up."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ORDER = ("D", "W", "M", "Q", "Y")

LABELS = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"}
UNIT = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
DEFAULT_SEASON = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": None}

_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


@dataclass(frozen=True)
class Frequency:
    code: str  # one of ORDER
    week_start: int = 0  # weekday the week starts on (0 = Monday); weekly only

    @property
    def pandas(self) -> str:
        if self.code == "W":
            week_end = _WEEKDAYS[(self.week_start - 1) % 7]
            return f"W-{week_end}"
        return self.code

    @property
    def label(self) -> str:
        if self.code == "W":
            return f"weekly (weeks start {_WEEKDAYS[self.week_start].title()})"
        return LABELS[self.code]

    @property
    def unit(self) -> str:
        return UNIT[self.code]

    @property
    def default_season(self) -> int | None:
        return DEFAULT_SEASON[self.code]


def detect(dates: pd.Series) -> Frequency | None:
    """Detect frequency from the median gap between sorted unique dates.

    Returns None when the spacing is irregular or there are too few dates.
    """
    unique = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(unique) < 3:
        return None
    gaps = unique.diff().dropna().dt.total_seconds() / 86400
    median = float(np.median(gaps))
    if median <= 1.5:
        return Frequency("D")
    if 6 <= median <= 8:
        weekday = int(unique.dt.day_of_week.mode().iloc[0])
        return Frequency("W", week_start=weekday)
    if 27 <= median <= 32:
        return Frequency("M")
    if 85 <= median <= 95:
        return Frequency("Q")
    if 355 <= median <= 370:
        return Frequency("Y")
    return None


def to_periods(dates: pd.Series, freq: Frequency) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(dates), freq=freq.pandas)


def coarser_options(freq: Frequency) -> list[str]:
    return list(ORDER[ORDER.index(freq.code) + 1:])


def rollup(demand: pd.Series, driver: pd.Series | None, to_code: str,
           driver_how: str = "mean") -> tuple[pd.Series, pd.Series | None]:
    """Aggregate a period-indexed series to a coarser frequency.

    Demand is summed. The driver is summed or averaged (driver_how = "sum" | "mean").
    Each source period is assigned to the coarser period containing its start date.
    """
    target = demand.index.to_timestamp(how="start").to_period(to_code)
    new_demand = demand.groupby(target).sum(min_count=1)
    new_driver = None
    if driver is not None:
        dtarget = driver.index.to_timestamp(how="start").to_period(to_code)
        grouped = driver.groupby(dtarget)
        new_driver = grouped.sum(min_count=1) if driver_how == "sum" else grouped.mean()
    return new_demand, new_driver


def label(period: pd.Period) -> str:
    """Readable period text: weeks as 'wk 2024-01-01' (start date), others as pandas prints them."""
    if period.freqstr.startswith("W"):
        return f"wk {period.start_time:%Y-%m-%d}"
    return str(period)


def labels(index) -> list[str]:
    return [label(p) for p in index]
