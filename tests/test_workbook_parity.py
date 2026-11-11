"""Check our equations reproduce the course workbook when given the workbook's own
start values and parameters. Skipped when the workbook is not next to the app
(it is course material and is not shipped with the app)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.init import StartValues
from engine.models import regression, smoothing

WORKBOOK = Path(__file__).resolve().parents[2] / "Forecasting_Classroom_Workbook.xlsx"
pytestmark = pytest.mark.skipif(not WORKBOOK.exists(), reason="course workbook not available")


@pytest.fixture(scope="module")
def wb():
    import openpyxl
    return openpyxl.load_workbook(WORKBOOK, data_only=True)


def row(ws, r, c0, c1):
    return np.array([ws.cell(r, c).value for c in range(c0, c1 + 1)], dtype=float)


def as_series(values, start):
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M"))


def test_ses_tshirts(wb):
    ws = wb["Smoothing (T-shirts)"]
    actual = row(ws, 4, 2, 24)  # B4:X4
    expected = row(ws, 17, 2, 25)  # B17:Y17 forecasts, incl. July 2016
    alpha = ws["X13"].value
    res = smoothing.ses(as_series(actual, "2014-08"), alpha, StartValues(level=actual[0]))
    assert np.allclose(res.fitted.values, expected[:-1], atol=1e-4)
    assert res.forecast(1).iloc[0] == pytest.approx(expected[-1], abs=1e-4)


def test_holt_cfl(wb):
    ws = wb["Holt trend (CFL)"]
    actual = row(ws, 4, 2, 24)  # B4:X4
    expected = row(ws, 11, 2, 29)  # B11:AC11 (fitted + 5 future)
    alpha, beta, phi = ws["P23"].value, ws["Q23"].value, ws["R23"].value
    res = smoothing.holt(as_series(actual, "2014-08"), alpha, beta, phi, StartValues(actual[0], 0.0))
    assert np.allclose(res.fitted.values, expected[:23], atol=1e-4)
    assert np.allclose(res.forecast(5).values, expected[23:], atol=1e-4)


def test_winters_paint(wb):
    ws = wb["Winters (paint)"]
    actual = row(ws, 3, 15, 98)  # O3 onwards: Jan 2009 .. Dec 2015
    actual = actual[~np.isnan(actual)]
    expected = row(ws, 25, 15, 15 + len(actual) - 1)
    start = StartValues(ws["N18"].value, ws["N20"].value, row(ws, 22, 3, 14))
    a, b, g, phi = ws["CT33"].value, ws["CU33"].value, ws["CV33"].value, ws["CS33"].value
    res = smoothing.winters(as_series(actual, "2009-01"), 12, a, b, g, phi, start)
    assert np.allclose(res.fitted.values, expected, atol=1e-4)


def test_simple_regression_juice(wb):
    ws = wb["Simple regression (juice)"]
    y = as_series(np.array([ws.cell(r, 2).value for r in range(5, 31)], dtype=float), "2020-01")
    x = as_series(np.array([ws.cell(r, 3).value for r in range(5, 31)], dtype=float), "2020-01")
    st = regression.regression(y, x).extras["stats"]
    assert st.coef[0] == pytest.approx(ws["G45"].value, rel=1e-9)
    assert st.coef[1] == pytest.approx(ws["G46"].value, rel=1e-9)
    assert st.std_err[1] == pytest.approx(ws["H46"].value, rel=1e-9)
    assert st.t_stat[1] == pytest.approx(ws["I46"].value, rel=1e-9)
    assert st.r2 == pytest.approx(ws["G33"].value, rel=1e-9)
    assert st.adj_r2 == pytest.approx(ws["G34"].value, rel=1e-9)
    assert st.se == pytest.approx(ws["G35"].value, rel=1e-9)
