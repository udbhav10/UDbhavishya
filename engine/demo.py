"""Realistic, reproducible mock products for users without data.

Each generator returns a table in the same layout a user would upload
(date, demand, driver), including a few future rows with driver values only,
so demo data flows through exactly the same validation path as real files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoProduct:
    key: str
    name: str
    unit: str
    frequency: str
    pattern: str
    driver: str
    build: Callable[[], pd.DataFrame]


def _frame(dates, demand, driver, driver_name, decimals=0) -> pd.DataFrame:
    demand = np.round(np.asarray(demand, dtype=float), decimals)
    return pd.DataFrame({"date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
                         "demand": demand, driver_name: driver})


def _bread() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    hist, future = 730, 14
    dates = pd.date_range("2023-01-02", periods=hist + future, freq="D")
    t = np.arange(hist + future)
    weekday = np.array([0.90, 0.88, 0.92, 0.95, 1.08, 1.30, 1.25])[dates.day_of_week]
    festival = (rng.random(hist + future) < 0.03).astype(int)
    level = 220 * (1 + 0.05 * t / 365)
    demand = level * weekday * (1 + 0.35 * festival) * rng.lognormal(0, 0.07, hist + future)
    demand[200] *= 2.2  # bulk order from a caterer
    demand[455] *= 0.3  # oven breakdown: stock-out
    demand[hist:] = np.nan
    return _frame(dates, demand, festival, "festival_day")


def _shampoo() -> pd.DataFrame:
    rng = np.random.default_rng(202)
    hist, future = 182, 8
    n = hist + future
    dates = pd.date_range("2022-01-03", periods=n, freq="7D")  # Mondays
    t = np.arange(n)
    season = 1 + 0.12 * np.sin(2 * np.pi * (t - 10) / 52)
    promo = np.zeros(n, dtype=int)
    promo[rng.choice(np.arange(3, n), size=n // 7, replace=False)] = 1
    level = 1800 * (1 + 0.06 * t / 52)
    demand = level * season * (1 + 0.30 * promo) * rng.lognormal(0, 0.05, n)
    demand[60] *= 1.9  # institutional order
    demand[121] *= 0.45  # supply disruption
    demand[hist:] = np.nan
    return _frame(dates, demand, promo, "promotion_week")


_PAINT_INDEX = np.array([0.87, 0.87, 1.09, 0.86, 0.95, 0.83, 0.51, 1.43, 1.29, 1.42, 0.94, 0.95])


def _paint() -> pd.DataFrame:
    rng = np.random.default_rng(303)
    hist, future = 96, 12
    n = hist + future
    dates = pd.date_range("2017-01-01", periods=n, freq="MS")
    t = np.arange(n)
    promo = np.zeros(n, dtype=int)
    for year in range(n // 12 + 1):  # two promotion months a year, never in July
        months = rng.choice([1, 2, 4, 5, 10, 11], size=2, replace=False)
        for m in months:
            if year * 12 + m < n:
                promo[year * 12 + m] = 1
    level = 210 + 1.1 * t
    demand = level * _PAINT_INDEX[t % 12] * (1 + 0.12 * promo) * rng.lognormal(0, 0.05, n)
    demand[26] *= 1.6  # bulk order for a housing project (Mar 2019)
    demand[52] *= 0.45  # stock-out after a plant shutdown (May 2021)
    demand[hist:] = np.nan
    return _frame(dates, demand, promo, "promotion_month", decimals=1)


_NORMAL_MAX_TEMP = np.array([24, 27, 32, 37, 40, 38, 34, 33, 33, 32, 29, 25], dtype=float)


def _air_conditioners() -> pd.DataFrame:
    rng = np.random.default_rng(404)
    hist, future = 108, 12
    n = hist + future
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    t = np.arange(n)
    temp = _NORMAL_MAX_TEMP[t % 12] + rng.normal(0, 1.2, n)
    temp[hist:] = _NORMAL_MAX_TEMP[t[hist:] % 12]  # future: seasonal normals
    level = 8000 * (1 + 0.09) ** (t / 12)
    demand = level * np.exp(0.075 * (temp - 32)) * rng.lognormal(0, 0.06, n)
    demand[40] *= 1.5  # dealer channel fill ahead of a price increase (May 2019)
    demand[51] *= 0.2  # lockdown: shops closed (Apr 2020)
    demand[hist:] = np.nan
    return _frame(dates, demand, np.round(temp, 1), "avg_max_temp_c")


_GDP_GROWTH = [4.8, 3.8, 7.9, 7.9, 7.9, 8.1, 7.7, 3.1, 7.9, 8.5, 5.2, 5.5, 6.4, 7.4, 8.0, 8.3,
               6.8, 6.5, 3.9, -5.8, 9.7, 7.0, 8.2, 8.2]


def _two_wheelers() -> pd.DataFrame:
    rng = np.random.default_rng(505)
    gdp = np.array(_GDP_GROWTH + [6.5, 6.5, 6.3])
    hist = len(_GDP_GROWTH)
    n = len(gdp)
    dates = pd.date_range("2001-01-01", periods=n, freq="YS")
    demand = np.empty(n)
    demand[0] = 4800.0
    for i in range(1, n):
        growth = (1.4 * gdp[i] - 2.5 + rng.normal(0, 2.0)) / 100
        demand[i] = demand[i - 1] * (1 + growth)
    demand[11] *= 1.12  # emission-norm pre-buying (2012)
    demand[hist:] = np.nan
    return _frame(dates, demand, gdp, "gdp_growth_pct")


PRODUCTS: dict[str, DemoProduct] = {p.key: p for p in [
    DemoProduct("bread", "Bakery bread loaves", "loaves", "daily", "weekday pattern, weekend peak",
                "festival day (1 = yes)", _bread),
    DemoProduct("shampoo", "Shampoo 180 ml pack", "packs", "weekly", "yearly season, growth, promotions",
                "promotion week (1 = yes)", _shampoo),
    DemoProduct("paint", "Decorative paint", "KL", "monthly", "trend + monsoon dip and festive peak",
                "promotion month (1 = yes)", _paint),
    DemoProduct("ac", "Room air conditioners", "units", "monthly", "summer peak, strong growth",
                "average max temperature (°C)", _air_conditioners),
    DemoProduct("two_wheelers", "Two-wheeler sales", "'000 units", "yearly", "trend, follows the economy",
                "GDP growth (%)", _two_wheelers),
]}


def load(key: str) -> pd.DataFrame:
    """Return the demo table for a product key."""
    return PRODUCTS[key].build()


def mapping(key: str) -> dict[str, str | None]:
    frame = load(key)
    return {"date": "date", "demand": "demand", "driver": frame.columns[2], "sku": None}
