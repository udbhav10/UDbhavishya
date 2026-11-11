import numpy as np
import pandas as pd
import pytest

from engine import outliers, seasonality, split


def series(values, start="2020-01", freq="M"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.period_range(start, periods=len(values), freq=freq))


# ---------------------------------------------------------------- 4.1 split

@pytest.mark.parametrize("n,season,code,expected", [
    (72, 12, "M", 12), (200, 52, "W", 52), (20, None, "Y", 4), (24, None, "Y", 5), (60, 7, "D", 7), (3, 12, "M", 1),
])
def test_default_test_length(n, season, code, expected):
    assert split.default_test_length(n, season, code) == expected


def test_split_is_time_ordered():
    s = series(range(72))
    sp = split.split(s, 12)
    assert len(sp.train) == 60 and len(sp.test) == 12
    assert sp.train.index.max() < sp.test.index.min()
    assert list(sp.test.values) == list(range(60, 72))
    assert str(sp.boundary) == "2024-12"


def test_split_rejects_bad_length():
    with pytest.raises(ValueError):
        split.split(series(range(10)), 10)


# ---------------------------------------------------------------- 4.2 seasonality

def seasonal(period, n, trend=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return series(100 + trend * t + 20 * np.sin(2 * np.pi * t / period) + rng.normal(0, 2, n))


def test_acf_confirms_twelve():
    check = seasonality.check(seasonal(12, 72), "M")
    assert check.default == 12 and check.peak_lag == 12 and check.agrees and check.suggestion is None


def test_acf_suggests_six():
    check = seasonality.check(seasonal(6, 72), "M")
    assert check.peak_lag == 6 and not check.agrees and check.suggestion == 6


def test_acf_formula():
    r = seasonality.acf(np.array([1.0, 2, 3, 4, 5]), 2)
    assert r[0] == pytest.approx(1.0) and r[1] == pytest.approx(1.0)


def test_enough_seasons():
    assert seasonality.enough_seasons(24, 12) and not seasonality.enough_seasons(23, 12)
    assert not seasonality.enough_seasons(100, None)


# ---------------------------------------------------------------- 4.3 outliers

def slide52_series():
    # 12 months with mean 100.9 and MAD ~11.8, May high and Nov low (slide 52).
    normal = [103.25, 98.85] * 5
    values = normal[:4] + [159.7] + normal[4:9] + [40.6] + normal[9:]
    return series(values, start="2023-01")


def test_slide52_units_limits():
    res = outliers.detect(slide52_series(), season=None, z_limit=3)
    assert res.mode == "units"
    assert res.reference.iloc[0] == pytest.approx(100.9, abs=0.01)
    assert res.sigma == pytest.approx(14.7, abs=0.1)
    assert res.lower.iloc[0] == pytest.approx(57, abs=0.5)
    assert res.upper.iloc[0] == pytest.approx(145, abs=0.5)
    flagged = [str(p) for p in res.flags[res.flags].index]
    assert flagged == ["2023-05", "2023-11"]
    assert res.score[pd.Period("2023-05", "M")] == pytest.approx(4.0, abs=0.05)
    assert res.score[pd.Period("2023-11", "M")] == pytest.approx(-4.1, abs=0.05)


def test_tighter_limit_flags_more():
    s = slide52_series()
    wide = outliers.detect(s, None, 3)
    tight = outliers.detect(s, None, 0.15)
    assert tight.count > wide.count
    assert (tight.upper - tight.lower).iloc[0] < (wide.upper - wide.lower).iloc[0]


def test_seasonal_percent_mode():
    t = np.arange(48)
    idx = np.array([0.5, 0.6, 0.8, 1, 1.2, 1.6, 2.5, 2.4, 1.2, 0.8, 0.7, 0.7])
    values = 200 * idx[t % 12] * (1 + 0.01 * np.sin(t))
    values[19] *= 2.1  # bulk order
    res = outliers.detect(series(values), season=12, z_limit=3)
    assert res.mode == "percent"
    assert res.sigma == pytest.approx(1.25 * res.spread)
    assert list(np.flatnonzero(res.flags)) == [19]
    # band scales with the season: wider in peak months
    assert (res.upper - res.lower).iloc[6] > (res.upper - res.lower).iloc[0]


@pytest.mark.parametrize("how,expected", [
    ("none", 159.7), ("reference", 100.9), ("neighbours", (98.85 + 159.7 + 103.25) / 3), ("cap", None),
])
def test_replacements(how, expected):
    res = outliers.detect(slide52_series(), None, 3, how)
    may = pd.Period("2023-05", "M")
    if how == "cap":
        expected = res.upper[may]
    assert res.corrected[may] == pytest.approx(expected, abs=0.01)
    if how == "none":
        assert res.corrections.empty and res.count == 2
    else:
        assert len(res.corrections) == 2
        assert set(res.corrections.columns) == {"period", "original", "new", "z"}


def test_only_training_is_touched():
    full = series(list(slide52_series().values) + [500.0, 1.0])
    sp = split.split(full, 2)
    res = outliers.detect(sp.train, None, 3, "reference")
    assert res.corrected.index.equals(sp.train.index)
    assert list(sp.test.values) == [500.0, 1.0]
