import numpy as np
import pandas as pd
import pytest

from engine import metrics, pipeline, recommend, split


def series(values, start="2024-01", freq="M"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.period_range(start, periods=len(values), freq=freq))


# ---------------------------------------------------------------- 6.1 metrics

def test_slide47_reference():
    sc = metrics.score([95, 91, 102, 110, 105, 110], [100, 95, 115, 120, 101, 105])
    assert sc.mad == pytest.approx(6.83, abs=0.005)
    assert sc.mape == pytest.approx(0.066, abs=0.0005)
    assert sc.wmape == pytest.approx(0.067, abs=0.0005)
    assert sc.rmse == pytest.approx(7.65, abs=0.005)
    assert sc.bias == pytest.approx(-3.83, abs=0.005)
    assert sc.nbias == pytest.approx(-0.56, abs=0.005)
    assert sc.ts == pytest.approx(-3.37, abs=0.005)
    assert metrics.bias_direction(sc.bias, sc.mad) == ("▼", "over-forecasting")


def test_zero_actual_disables_mape():
    sc = metrics.score([0, 10, 20], [1, 11, 19])
    assert sc.mape is None and sc.wmape == pytest.approx(3 / 30)


def test_mase_scale():
    train = series([10, 12, 14, 16])
    assert metrics.naive_scale(train, None) == pytest.approx(2)
    sc = metrics.score([20, 22], [19, 21], scale=2)
    assert sc.mase == pytest.approx(0.5)


def test_under_forecasting_arrow():
    sc = metrics.score([110, 120], [100, 100])
    assert metrics.bias_direction(sc.bias, sc.mad) == ("▲", "under-forecasting")


# ---------------------------------------------------------------- 6.2 recommendation

TEST = series([100, 120, 140, 160])
TRAIN = series(np.arange(24) * 5 + 100, start="2022-01")


def fc(values):
    return series(values)


def test_clear_winner():
    cmp = recommend.compare(TEST, {
        "snaive": fc([80, 100, 120, 140]),   # wMAPE 80/520
        "ses": fc([130, 130, 130, 130]),      # 30+10+10+30=80 -> same
        "winters": fc([101, 121, 139, 161]),  # 4/520
        "holt": fc([95, 115, 135, 155]),      # 20/520
    }, TRAIN, 12)
    assert cmp.benchmark == "snaive" and cmp.recommended == "winters"
    assert cmp.sentence.startswith("Recommended: Winters · wMAPE 0.8%")
    assert cmp.fva["winters"] == pytest.approx((80 - 4) / 520 * 100)
    assert not cmp.beats["ses"]


def test_nothing_beats_naive():
    cmp = recommend.compare(TEST, {
        "snaive": fc([101, 121, 141, 161]),
        "ses": fc([130, 130, 130, 130]),
        "holt": fc([90, 110, 130, 150]),
    }, TRAIN, 12)
    assert cmp.recommended == "snaive" and cmp.no_winner
    assert "no method improved" in cmp.sentence


def test_combination_can_win():
    cmp = recommend.compare(TEST, {
        "snaive": fc([80, 100, 120, 140]),
        "holt": fc([90, 110, 130, 150]),       # -10 each
        "regression": fc([110, 130, 150, 170]),  # +10 each
    }, TRAIN, 12)
    assert recommend.COMBO in cmp.scores
    assert cmp.names[recommend.COMBO] in ("Average of Holt + Regression", "Average of Regression + Holt")
    assert cmp.recommended == recommend.COMBO and cmp.scores[recommend.COMBO].mad == 0


def test_non_seasonal_benchmark_is_naive():
    cmp = recommend.compare(TEST, {"naive": fc([90] * 4), "ses": fc([125] * 4)}, TRAIN, None)
    assert cmp.benchmark == "naive"
    assert set(cmp.table()["key"]) == {"naive", "ses"}


# ---------------------------------------------------------------- 6.3 correction effect

def test_correction_effect_reports_both():
    rng = np.random.default_rng(0)
    y = series(100 + rng.normal(0, 3, 36), start="2021-01")
    y.iloc[20] = 300
    sp = split.split(y, 6)
    corrected = sp.train.copy()
    corrected.iloc[20] = 100

    def fit_and_forecast(method, train):
        fit = pipeline.fit_method(method, train, None)
        return pipeline.forecast_ahead(fit, len(sp.test))

    eff = recommend.correction_effect("ses", sp.train, corrected, sp.test, fit_and_forecast)
    assert eff.wmape_without is not None and eff.wmape_with is not None
    assert recommend.correction_effect("ses", sp.train, sp.train, sp.test, fit_and_forecast) is None
