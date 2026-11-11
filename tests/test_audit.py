"""The slide-83 audit checklist as automated tests.

1. Random instead of time-ordered train/test split
2. Centred averages that use future (test) data
3. Parameters tuned on the test period
4. Wrong seasonal period
5. MAPE on zero-demand months
6. App numbers must match the Excel workbook (see test_workbook_parity.py)
"""
import time

import numpy as np
import pandas as pd
import pytest

from engine import demo, init, optimise, results
from engine import validate as v
from engine.pipeline import MethodSettings

SEASONAL = ["bread", "shampoo", "paint", "ac"]


def demo_series(key):
    frame = demo.load(key)
    raw = v.parse(frame, demo.mapping(key)).skus[v.SINGLE_SKU]
    return v.build_series(raw, v.diagnose(raw), raw.freq, driver_name=frame.columns[2])


def with_new_test_values(series, factor=3.0):
    """Same series, but every test-period value multiplied (test = default length)."""
    first = results.run(series)
    changed = series.demand.copy()
    changed.loc[first.split.test.index] *= factor
    return v.SeriesData(series.sku, series.freq, changed, series.driver, series.future_driver,
                        series.notes, series.messages, series.driver_name)


# ---------------------------------------------------------------- 1. time-ordered split

@pytest.mark.parametrize("key", list(demo.PRODUCTS))
def test_split_is_time_ordered(key):
    res = results.run(demo_series(key))
    train, test = res.split.train, res.split.test
    assert train.index.is_monotonic_increasing and test.index.is_monotonic_increasing
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(res.series.demand)
    assert test.index[-1] == res.series.demand.index[-1]  # test = the latest periods


# ---------------------------------------------------------------- 2. no future data in averages

@pytest.mark.parametrize("key", list(demo.PRODUCTS))
def test_start_values_and_outlier_limits_ignore_test_data(key):
    series = demo_series(key)
    settings = results.RunSettings(outlier_replacement="reference")
    a = results.run(series, settings)
    b = results.run(with_new_test_values(series), settings)
    for method, sv in a.start_values.items():
        other = b.start_values[method]
        assert sv.level == other.level and sv.trend == other.trend
        if sv.season is not None:
            assert np.allclose(sv.season, other.season)
    pd.testing.assert_series_equal(a.outliers.reference, b.outliers.reference)
    pd.testing.assert_series_equal(a.train, b.train)


@pytest.mark.parametrize("key", list(demo.PRODUCTS))
def test_test_forecasts_ignore_test_actuals(key):
    series = demo_series(key)
    a = results.run(series)
    b = results.run(with_new_test_values(series))
    for method, fc in a.test_forecasts.items():
        pd.testing.assert_series_equal(fc, b.test_forecasts[method])


# ---------------------------------------------------------------- 3. tuning on training only

@pytest.mark.parametrize("method,key", [("ses", "two_wheelers"), ("holt", "two_wheelers"), ("winters", "paint")])
def test_optimiser_ignores_test_period(method, key):
    series = demo_series(key)
    tuned = []
    for s in (series, with_new_test_values(series)):
        res = results.run(s)
        start = res.fits[method].start
        tuned.append(optimise.optimise(method, res.train, start, s=res.s))
    assert tuned[0] == tuned[1]


# ---------------------------------------------------------------- 4. seasonal period

@pytest.mark.parametrize("key", SEASONAL)
def test_seasonal_period_confirmed_by_acf(key):
    res = results.run(demo_series(key))
    assert res.s == res.series.freq.default_season
    assert res.season_check.agrees


# ---------------------------------------------------------------- 5. MAPE on zero demand

def test_mape_disabled_with_zero_demand():
    series = demo_series("bread")
    res0 = results.run(series)
    demand = series.demand.copy()
    demand.loc[res0.split.test.index[2]] = 0.0
    zeroed = v.SeriesData(series.sku, series.freq, demand, series.driver, series.future_driver,
                          series.notes, series.messages, series.driver_name)
    res = results.run(zeroed)
    assert all(sc.mape is None for sc in res.comparison.scores.values())
    assert all(sc.wmape is not None for sc in res.comparison.scores.values())
    assert res.final.basis.source == "wMAPE"


# ---------------------------------------------------------------- 10.3 performance

def test_recompute_on_1000_periods_is_fast():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2022-01-01", periods=1000, freq="D")
    t = np.arange(1000)
    demand = pd.Series(200 * (1 + 0.2 * np.sin(2 * np.pi * t / 7)) * rng.lognormal(0, 0.05, 1000) + 0.05 * t,
                       index=idx)
    driver = pd.Series(rng.integers(0, 2, 1000).astype(float), index=idx)
    series = v.SeriesData("x", __import__("engine.frequency", fromlist=["Frequency"]).Frequency("D"),
                          demand, driver, None, [], [], "promo")
    from ui import charts  # import (one-off, at app start) is not part of a slider response

    results.run(series)  # warm-up
    settings = results.RunSettings(methods={"holt": MethodSettings({"alpha": 0.5})})
    start = time.perf_counter()
    res = results.run(series, settings)

    charts.fit_chart(res.train, res.split.test, res.fits["holt"].fitted, res.test_forecasts["holt"], None, "Holt")
    charts.compare_chart(res.comparison.table(), res.comparison.benchmark, res.comparison.recommended)
    charts.forecast_chart(res.series.demand.iloc[-60:], res.final.table, 80, res.final.name)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"recompute took {elapsed:.2f}s"
