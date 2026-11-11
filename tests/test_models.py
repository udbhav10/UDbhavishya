import numpy as np
import pandas as pd
import pytest

from engine import init, optimise, split
from engine.models import availability, naive, regression, smoothing


def series(values, start="2020-01", freq="M"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.period_range(start, periods=len(values), freq=freq))


def seasonal_trend(n=60, s=12, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = 1 + 0.3 * np.sin(2 * np.pi * t / s)
    return series((100 + 2 * t) * idx * rng.lognormal(0, 0.02, n))


# ---------------------------------------------------------------- 5.1 start values

def test_flat_start_for_ses():
    sv = init.start_values(np.arange(1, 21, dtype=float))
    assert sv.level == pytest.approx(5.5) and sv.trend == 0 and sv.season is None
    assert "first 10" in sv.notes["level"]


def test_trend_start_for_holt():
    sv = init.start_values(np.array([10 + 3 * t for t in range(1, 15)], dtype=float), trend=True)
    assert sv.level == pytest.approx(10) and sv.trend == pytest.approx(3)


def test_seasonal_start_values():
    y = seasonal_trend().to_numpy()
    sv = init.start_values(y, s=12, trend=True, seasonal=True)
    assert len(sv.season) == 12 and sv.season.mean() == pytest.approx(1.0)
    assert sv.season[3] > 1.2 and sv.season[9] < 0.8  # sin peak at t=3, trough at t=9
    assert sv.trend == pytest.approx(2, abs=0.6)


def test_centred_moving_average_even():
    ma = init.centred_moving_average(np.arange(1, 9, dtype=float), 4)
    assert np.isnan(ma[1]) and ma[2] == pytest.approx(3.0)


def test_seasonal_needs_two_seasons():
    with pytest.raises(ValueError):
        init.start_values(np.ones(20), s=12, seasonal=True)


def test_start_values_ignore_test_data():
    s = seasonal_trend()
    sp_a = split.split(s, 12)
    changed = s.copy()
    changed.iloc[-12:] *= 3
    sp_b = split.split(changed, 12)
    a = init.start_values(sp_a.train.to_numpy(), 12, True, True)
    b = init.start_values(sp_b.train.to_numpy(), 12, True, True)
    assert a.level == b.level and a.trend == b.trend and np.allclose(a.season, b.season)


def test_overrides():
    sv = init.start_values(np.arange(1, 21, dtype=float), trend=True)
    over = sv.with_overrides(level=120)
    assert over.level == 120 and over.trend == sv.trend and sv.level != 120


# ---------------------------------------------------------------- 5.2 naive

def test_seasonal_naive_same_month_last_year():
    train = series(np.arange(48), start="2020-01")  # ends Dec 2023
    res = naive.seasonal_naive(train, 12)
    fc = res.forecast(12)
    assert fc[pd.Period("2024-03", "M")] == train[pd.Period("2023-03", "M")]


def test_naive_flat():
    res = naive.naive(series([1, 2, 7]))
    assert list(res.forecast(3).values) == [7, 7, 7]
    assert np.isnan(res.fitted.iloc[0]) and res.fitted.iloc[2] == 2


def test_seasonal_naive_fallback():
    res = naive.seasonal_naive(series([1, 2, 7]), None)
    assert res.extras["fallback"] and list(res.forecast(2).values) == [7, 7]


# ---------------------------------------------------------------- 5.3 smoothing

def test_ses_hand_example():
    res = smoothing.ses(series([54]), 0.3, init.StartValues(level=50))
    assert res.components["level"].iloc[0] == pytest.approx(51.2)
    assert res.forecast(1).iloc[0] == pytest.approx(51.2)


def test_holt_hand_example():
    res = smoothing.holt(series([104]), 0.5, 0.3, 1.0, init.StartValues(100, 2))
    assert res.fitted.iloc[0] == pytest.approx(102)
    assert res.components["level"].iloc[0] == pytest.approx(103)
    assert res.components["trend"].iloc[0] == pytest.approx(2.3)
    assert list(res.forecast(3).round(4)) == [105.3, 107.6, 109.9]


def test_holt_damped_hand_example():
    res = smoothing.holt(series([104]), 0.5, 0.3, 0.9, init.StartValues(100, 2))
    assert res.fitted.iloc[0] == pytest.approx(101.8)
    assert res.forecast(2).iloc[1] == pytest.approx(102.9 + (0.9 + 0.81) * 2.13)


def test_winters_hand_example():
    res = smoothing.winters(series([90]), 2, 0.5, 0.2, 0.3, 1.0, init.StartValues(100, 0, np.array([0.8, 1.2])))
    assert res.fitted.iloc[0] == pytest.approx(80)
    assert res.components["level"].iloc[0] == pytest.approx(106.25)
    assert res.components["trend"].iloc[0] == pytest.approx(1.25)
    assert res.components["season"].iloc[0] == pytest.approx(0.3 * 90 / 106.25 + 0.56)
    fc = res.forecast(2)
    assert fc.iloc[0] == pytest.approx(129) and fc.iloc[1] == pytest.approx(108.75 * (0.3 * 90 / 106.25 + 0.56))


def test_phi_one_equals_undamped():
    s = seasonal_trend()
    a = smoothing.holt(s, 0.4, 0.2, 1.0)
    b = smoothing.holt(s, 0.4, 0.2)
    assert np.allclose(a.forecast(5), b.forecast(5))


def test_winters_rejects_zero():
    with pytest.raises(ValueError):
        smoothing.winters(series([0.0] + [1.0] * 30), 12, 0.3, 0.1, 0.1)


# ---------------------------------------------------------------- 5.5 optimiser

def test_optimiser_within_bounds_and_improves():
    s = seasonal_trend()
    sp = split.split(s, 12)
    start = init.start_values(sp.train.to_numpy(), 12, True, True)
    params = optimise.optimise("winters", sp.train, start, s=12)
    for name in ("alpha", "beta", "gamma"):
        assert 0.01 <= params[name] <= 0.99
    tuned = smoothing.winters(sp.train, 12, params["alpha"], params["beta"], params["gamma"], 1.0, start).sse
    grid_start = smoothing.winters(sp.train, 12, 0.5, 0.5, 0.5, 1.0, start).sse
    assert tuned <= grid_start


def test_optimiser_ignores_test_data():
    s = seasonal_trend()
    changed = s.copy()
    changed.iloc[-12:] *= 5
    results = []
    for data in (s, changed):
        sp = split.split(data, 12)
        start = init.start_values(sp.train.to_numpy(), trend=True)
        results.append(optimise.optimise("holt", sp.train, start, damped=True))
    assert results[0] == results[1]


# ---------------------------------------------------------------- 5.6 availability

def test_availability_rules():
    av = availability(series(np.ones(20)), 12, has_driver=False)
    assert not av["winters"].available and "two seasons" in av["winters"].reason
    assert not av["regression"].available and av["regression"].reason == "No driver column mapped"
    av = availability(series(np.r_[0.0, np.ones(30)]), 12, has_driver=True)
    assert av["winters"].reason == "Multiplicative seasonality needs demand above zero"
    assert av["regression"].available


# ---------------------------------------------------------------- 5.7 regression

def test_regression_exact_line():
    x = series(np.arange(10, 30))
    y = 5 + 2 * x
    res = regression.regression(y, x)
    st = res.extras["stats"]
    assert st.coef == pytest.approx([5, 2]) and st.r2 == pytest.approx(1)
    fc = res.forecast(2, x_future=[100, 101])
    assert list(fc.values) == pytest.approx([205, 207])


def test_regression_seasonal_dummies():
    y = seasonal_trend(48)
    x = series(np.random.default_rng(3).normal(0, 1, 48))
    res = regression.regression(y, x, trend=True, seasonal=True, s=12)
    st = res.extras["stats"]
    assert len(st.coef) == 2 + 1 + 11
    assert st.names[3] == "Season: Jan" and "Season: Dec" not in st.names


def test_regression_needs_future_driver():
    x = series(np.arange(10.0))
    res = regression.regression(x * 2, x)
    with pytest.raises(ValueError):
        res.forecast(2)


def test_regression_few_observations_flag():
    x = series(np.arange(12.0))
    y = x * 2 + np.random.default_rng(0).normal(0, 1, 12)
    st = regression.regression(y, x, seasonal=True, s=4).extras["stats"]
    assert st.obs_per_coef < 10
