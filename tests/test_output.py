import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import demo, ranges, results, tracking
from engine import validate as v
from engine.pipeline import MethodSettings

WORKBOOK = Path(__file__).resolve().parents[2] / "Forecasting_Classroom_Workbook.xlsx"


def demo_series(key):
    frame = demo.load(key)
    parsed = v.parse(frame, demo.mapping(key))
    raw = parsed.skus[v.SINGLE_SKU]
    return v.build_series(raw, v.diagnose(raw), raw.freq, driver_name=frame.columns[2])


# ---------------------------------------------------------------- 7.1 refit and horizon

def test_refit_uses_all_periods_and_caps_horizon():
    s = demo_series("paint")
    res = results.run(s, results.RunSettings(chosen="holt", horizon=99))
    assert res.max_horizon == 24
    assert len(res.final.table) == 24
    assert res.final.table.index[0] == s.demand.index[-1] + 1
    # final forecast differs from the training-only fit: it has seen the test year
    train_only = res.fits["holt"].forecast(12)
    assert not np.allclose(train_only.values, res.final.table["forecast"].values[:12])


def test_yearly_horizon_limit():
    res = results.run(demo_series("two_wheelers"), results.RunSettings(horizon=50))
    assert res.max_horizon == 10 and len(res.final.table) == 10


# ---------------------------------------------------------------- 7.2 ranges

def test_z_values():
    assert ranges.z_value(20) == pytest.approx(0.253, abs=0.001)
    assert ranges.z_value(60) == pytest.approx(0.842, abs=0.001)
    assert ranges.z_value(80) == pytest.approx(1.282, abs=0.001)


def test_range_example_362():
    b = ranges.basis(0.074, 0.07)
    assert b.r == pytest.approx(0.0925) and b.source == "MAPE"
    t = ranges.build(pd.Series([362.0]), b.r, 80)
    assert t["lower"].iloc[0] == pytest.approx(319, abs=0.5)
    assert t["upper"].iloc[0] == pytest.approx(405, abs=0.5)


def test_range_floor_and_wmape_fallback():
    b = ranges.basis(None, 0.9)
    assert b.source == "wMAPE"
    t = ranges.build(pd.Series([10.0]), b.r, 80)
    assert t["lower"].iloc[0] == 0


# ---------------------------------------------------------------- 7.3 tracking signal

def test_tracking_matches_workbook_error_metrics():
    actual = pd.Series([95, 91, 102, 110, 105, 110], dtype=float)
    forecast = pd.Series([100, 95, 115, 120, 101, 105], dtype=float)
    tr = tracking.running(actual, forecast)
    assert list(tr.signal.round(4)) == [-1, -2, -3, -4, -3.8889, -3.3659]
    assert tr.ok  # reaches -4 but never goes beyond it


@pytest.mark.skipif(not WORKBOOK.exists(), reason="course workbook not available")
def test_tracking_against_workbook_cells():
    import openpyxl
    ws = openpyxl.load_workbook(WORKBOOK, data_only=True)["Error metrics"]
    actual = pd.Series([ws.cell(5, c).value for c in range(2, 8)], dtype=float)
    forecast = pd.Series([ws.cell(6, c).value for c in range(2, 8)], dtype=float)
    expected = [ws.cell(r, 7).value for r in range(24, 30)]
    assert np.allclose(tracking.running(actual, forecast).signal.values, expected)


def test_tracking_alarm():
    idx = pd.period_range("2024-01", periods=6, freq="M")
    tr = tracking.running(pd.Series([10.0] * 6, index=idx), pd.Series([12, 12, 12, 12, 12, 11.0], index=idx))
    assert not tr.ok and str(tr.breach) == "2024-05"
    assert "over-forecasting" in tr.verdict


# ---------------------------------------------------------------- 7.4 driver values and CSV

def test_future_driver_from_file():
    s = demo_series("paint")
    values, source = results.future_driver(s, 12)
    assert source == "From the future rows in your file"
    assert list(values.values) == list(s.future_driver.values[:12])


def test_future_driver_last_known():
    s = demo_series("paint")
    s.future_driver = None
    values, source = results.future_driver(s, 3)
    assert (values == s.driver.iloc[-1]).all() and "last known value" in source


def test_future_driver_edited():
    s = demo_series("paint")
    values, source = results.future_driver(s, 2, [1, 0])
    assert list(values) == [1, 0] and source == "Edited by you"


def test_forecast_csv_columns():
    res = results.run(demo_series("ac"), results.RunSettings(chosen="regression", horizon=6))
    frame = pd.read_csv(io.BytesIO(results.forecast_csv(res)))
    assert list(frame.columns) == ["period", "forecast", "lower", "upper", "method", "range_level"]
    assert len(frame) == 6 and frame["method"].iloc[0] == "Regression"
    assert res.final.driver_source == "From the future rows in your file"


# ---------------------------------------------------------------- 7.5 run result end to end

@pytest.mark.parametrize("key", list(demo.PRODUCTS))
def test_run_end_to_end(key):
    res = results.run(demo_series(key))
    assert res.comparison.recommended in res.comparison.scores
    assert res.final.level == 80 and (res.final.table["upper"] >= res.final.table["forecast"]).all()
    assert len(res.tracking.signal) == len(res.split.test)
    assert res.split.train.index.max() < res.split.test.index.min()


def test_run_with_outlier_correction_and_settings():
    s = demo_series("paint")
    settings = results.RunSettings(outlier_replacement="reference",
                                   methods={"winters": MethodSettings({"alpha": 0.2, "gamma": 0.3})})
    res = results.run(s, settings)
    assert res.outliers.count >= 2
    assert res.fits["winters"].params["alpha"] == 0.2
    assert res.correction is None or res.correction.wmape_with is not None
    # test period untouched
    assert res.split.test.equals(s.demand.iloc[-12:])
