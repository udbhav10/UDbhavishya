import numpy as np
import pandas as pd
import pytest

from engine import demo
from engine import validate as v
from engine.messages import has_errors

KEYS = list(demo.PRODUCTS)
MIN_LENGTH = {"daily": 365, "weekly": 156, "monthly": 60, "yearly": 20}
CODES = {"daily": "D", "weekly": "W", "monthly": "M", "yearly": "Y"}


def build(key):
    frame = demo.load(key)
    parsed = v.parse(frame, demo.mapping(key))
    raw = parsed.skus[v.SINGLE_SKU]
    diag = v.diagnose(raw)
    return parsed, raw, diag


def test_five_products():
    assert len(demo.PRODUCTS) == 5
    assert {p.frequency for p in demo.PRODUCTS.values()} == {"daily", "weekly", "monthly", "yearly"}


@pytest.mark.parametrize("key", KEYS)
def test_reproducible(key):
    pd.testing.assert_frame_equal(demo.load(key), demo.load(key))


@pytest.mark.parametrize("key", KEYS)
def test_realistic_shape(key):
    p = demo.PRODUCTS[key]
    frame = demo.load(key)
    hist = frame[frame["demand"].notna()]
    assert (hist["demand"] >= 0).all()
    assert len(hist) >= MIN_LENGTH[p.frequency]
    assert frame.shape[1] == 3 and frame.iloc[:, 2].notna().all()  # driver present everywhere
    assert frame["demand"].isna().sum() > 0  # future driver rows


def test_ac_demand_correlates_with_temperature():
    frame = demo.load("ac").dropna()
    assert np.corrcoef(frame["demand"], frame["avg_max_temp_c"])[0, 1] > 0.3


def test_paint_has_promotion_driver():
    frame = demo.load("paint")
    assert set(frame["promotion_month"].unique()) == {0, 1}


@pytest.mark.parametrize("key", KEYS)
def test_demo_validates_cleanly(key):
    parsed, raw, diag = build(key)
    assert not has_errors(parsed.messages + diag.messages)
    assert raw.freq.code == CODES[demo.PRODUCTS[key].frequency]
    assert not diag.gaps and diag.n_duplicate_periods == 0
    series = v.build_series(raw, diag, raw.freq)
    assert series.future_driver is not None and len(series.future_driver) > 0


EXPECTED_WINTERS = {"bread": True, "shampoo": True, "paint": True, "ac": True, "two_wheelers": False}


@pytest.mark.parametrize("key", KEYS)
def test_demo_supports_suitable_methods(key):
    from engine import split
    from engine.models import availability

    parsed, raw, diag = build(key)
    series = v.build_series(raw, diag, raw.freq)
    s = raw.freq.default_season
    test_len = split.default_test_length(len(series.demand), s, raw.freq.code)
    sp = split.split(series.demand, test_len)
    av = availability(sp.train, s, has_driver=series.driver is not None)
    for method in ("naive", "snaive", "ses", "holt", "regression"):
        assert av[method].available, (method, av[method].reason)
    assert av["winters"].available == EXPECTED_WINTERS[key], av["winters"].reason
    if key == "paint":
        assert test_len == 12
