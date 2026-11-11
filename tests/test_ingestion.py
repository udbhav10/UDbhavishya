import io

import numpy as np
import pandas as pd
import pytest

from engine import frequency as fq
from engine import io as eio
from engine import validate as v
from engine.messages import ChangeNote, Message, has_errors


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def monthly(n=24, start="2022-01", **extra):
    dates = pd.period_range(start, periods=n, freq="M").to_timestamp()
    data = {"Date": dates.strftime("%Y-%m-%d"), "Sales": np.arange(100, 100 + n, dtype=float)}
    data.update(extra)
    return pd.DataFrame(data)


# ---------------------------------------------------------------- 2.1 io

def test_read_csv_and_preview():
    res = eio.read_upload("data.csv", csv_bytes(monthly()))
    assert res.messages == [] and len(res.frame) == 24


def test_read_semicolon_csv():
    data = monthly().to_csv(index=False, sep=";").encode()
    assert list(eio.read_upload("d.csv", data).frame.columns) == ["Date", "Sales"]


def test_read_xlsx():
    buf = io.BytesIO()
    monthly().to_excel(buf, index=False)
    res = eio.read_upload("data.xlsx", buf.getvalue())
    assert res.frame is not None and len(res.frame) == 24


def test_oversized_file_rejected():
    res = eio.read_upload("big.csv", b"x" * (eio.MAX_BYTES + 1))
    assert res.frame is None and res.messages[0].code == "file_too_large"
    assert "50 MB" in res.messages[0].summary


def test_wrong_extension_rejected():
    res = eio.read_upload("data.txt", b"a,b\n1,2\n")
    assert res.frame is None and res.messages[0].code == "file_type"


def test_mapping_suggestion():
    frame = pd.DataFrame({"Date": ["2024-01-01"], "Sales": [1], "Price": [9.5]})
    assert eio.suggest_mapping(frame) == {"date": "Date", "demand": "Sales", "driver": "Price", "sku": None}


def test_template_has_future_rows():
    t = pd.read_csv(io.BytesIO(eio.template_csv()))
    assert list(t.columns) == ["date", "demand", "driver"]
    assert t["demand"].isna().sum() == 3 and t["driver"].notna().all()


# ---------------------------------------------------------------- 2.2 messages

@pytest.mark.parametrize("level", ["error", "warning", "info"])
def test_message_levels(level):
    m = Message(level, "x", "one line")
    assert m.blocking == (level == "error")
    assert has_errors([m]) == (level == "error")


def test_message_rejects_unknown_level():
    with pytest.raises(ValueError):
        Message("fatal", "x", "y")


def test_change_note():
    assert ChangeNote("gaps", "2 gaps filled").kind == "gaps"


# ---------------------------------------------------------------- 2.3 frequency

@pytest.mark.parametrize("freq,code", [("D", "D"), ("W-SUN", "W"), ("MS", "M"), ("QS", "Q"), ("YS", "Y")])
def test_detect_each_frequency(freq, code):
    dates = pd.Series(pd.date_range("2015-01-05", periods=30, freq=freq))
    assert fq.detect(dates).code == code


def test_weekly_anchor_follows_data():
    dates = pd.Series(pd.date_range("2024-01-03", periods=10, freq="7D"))  # Wednesdays
    f = fq.detect(dates)
    assert f.week_start == 2
    periods = fq.to_periods(dates, f)
    assert periods[0].start_time == pd.Timestamp("2024-01-03")


def test_irregular_dates():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-20", "2024-03-15", "2024-06-01"]))
    assert fq.detect(dates) is None


def test_rollup_daily_to_monthly():
    idx = pd.period_range("2024-01-01", "2024-03-31", freq="D")
    demand = pd.Series(1.0, index=idx)
    driver = pd.Series(2.0, index=idx)
    d, drv = fq.rollup(demand, driver, "M", "mean")
    assert list(d.values) == [31, 29, 31] and list(drv.values) == [2, 2, 2]


# ---------------------------------------------------------------- 2.4 SKU handling

def test_sku_column_detected_case_insensitive():
    a = monthly(12).assign(**{" sku ": "A"})
    b = monthly(18).assign(**{" sku ": "B"})
    frame = pd.concat([a, b], ignore_index=True)
    mapping = eio.suggest_mapping(frame)
    assert mapping["sku"] == " sku "
    parsed = v.parse(frame, mapping)
    assert set(parsed.skus) == {"A", "B"}
    assert len(parsed.skus["B"].rows) == 18


def test_no_sku_column_single_product():
    parsed = v.parse(monthly(), eio.suggest_mapping(monthly()))
    assert list(parsed.skus) == [v.SINGLE_SKU]
    assert any(m.code == "single_sku" and "no SKU column" in m.summary for m in parsed.messages)


def test_mixed_frequencies_error():
    a = monthly(12).assign(SKU="A")
    b = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
                      "Sales": 1.0, "SKU": "B"})
    parsed = v.parse(pd.concat([a, b]), {"date": "Date", "demand": "Sales", "driver": None, "sku": "SKU"})
    err = [m for m in parsed.messages if m.code == "mixed_freq"]
    assert err and err[0].blocking and set(err[0].rows["SKU"]) == {"A", "B"}


# ---------------------------------------------------------------- 2.5 checks

MAP = {"date": "Date", "demand": "Sales", "driver": None, "sku": None}


def diag(frame, mapping=MAP):
    parsed = v.parse(frame, mapping)
    raw = parsed.skus[v.SINGLE_SKU]
    return parsed, raw, v.diagnose(raw)


def test_unreadable_dates_error():
    f = monthly(12)
    f.loc[3, "Date"] = "not a date"
    parsed = v.parse(f, MAP)
    m = next(m for m in parsed.messages if m.code == "bad_dates")
    assert m.blocking and list(m.rows["row"]) == [5]


def test_non_numeric_demand_error():
    f = monthly(12)
    f["Sales"] = f["Sales"].astype(object)
    f.loc[2, "Sales"] = "abc"
    parsed = v.parse(f, MAP)
    m = next(m for m in parsed.messages if m.code == "bad_demand")
    assert m.blocking and list(m.rows["row"]) == [4]


def test_thousands_separator_is_numeric():
    f = monthly(12)
    f["Sales"] = ["1,234"] * 12
    parsed = v.parse(f, MAP)
    assert not has_errors(parsed.messages)
    assert parsed.skus[v.SINGLE_SKU].rows["demand"].iloc[0] == 1234


def test_duplicates_conflicting_and_exact():
    f = monthly(12)
    extra = pd.DataFrame({"Date": ["2022-03-01", "2022-05-01"], "Sales": [95.0, f.loc[4, "Sales"]]})
    _, _, d = diag(pd.concat([f, extra], ignore_index=True))
    m = next(m for m in d.messages if m.code == "duplicates")
    assert m.blocking and d.n_duplicate_periods == 2
    types = d.duplicates.groupby("period")["type"].first()
    assert types[pd.Period("2022-03", "M")] == "conflicting"
    assert types[pd.Period("2022-05", "M")] == "exact copy"
    assert len(m.actions) == 4


def test_duplicates_after_week_alignment():
    dates = list(pd.date_range("2024-01-01", periods=10, freq="7D").strftime("%Y-%m-%d"))
    dates.append("2024-03-06")  # Wednesday in the week starting Mon 4 Mar (also in list)
    f = pd.DataFrame({"Date": dates, "Sales": [10.0] * 10 + [12.0]})
    _, _, d = diag(f)
    assert d.n_duplicate_periods == 1
    assert set(d.duplicates["date"].dt.strftime("%Y-%m-%d")) == {"2024-03-04", "2024-03-06"}


def test_gaps_warning():
    f = monthly(12).drop(index=[4, 5])
    _, _, d = diag(f)
    m = next(m for m in d.messages if m.code == "gaps")
    assert m.level == "warning" and [str(p) for p in d.gaps] == ["2022-05", "2022-06"]


def test_negative_and_zero_warnings():
    f = monthly(12)
    f.loc[2, "Sales"] = -5
    f.loc[3, "Sales"] = 0
    _, _, d = diag(f)
    codes = {m.code: m for m in d.messages}
    assert codes["negative"].level == "warning" and codes["zeros"].level == "warning"
    assert "MAPE will be excluded" in codes["zeros"].summary


def test_future_driver_rows():
    f = monthly(18, Price=np.linspace(10, 12, 18))
    f.loc[12:, "Sales"] = np.nan
    mapping = {"date": "Date", "demand": "Sales", "driver": "Price", "sku": None}
    parsed, raw, d = diag(f, mapping)
    assert len(d.future) == 6 and len(d.history) == 12
    assert not d.gaps


# ---------------------------------------------------------------- 2.6 resolution

def _dup_frame():
    f = monthly(12)
    extra = pd.DataFrame({"Date": ["2022-03-01"], "Sales": [50.0]})
    return pd.concat([f, extra], ignore_index=True)


@pytest.mark.parametrize("choice,expected", [("sum", 152.0), ("first", 102.0), ("last", 50.0), ("mean", 76.0)])
def test_duplicate_resolution(choice, expected):
    parsed, raw, d = diag(_dup_frame())
    s = v.build_series(raw, d, raw.freq, duplicate_choice=choice)
    assert s.demand.index.is_unique and len(s.demand) == 12
    assert s.demand[pd.Period("2022-03", "M")] == expected
    assert s.notes[0].kind == "duplicates" and "1 duplicate periods" in s.notes[0].summary


def test_duplicate_choice_required():
    parsed, raw, d = diag(_dup_frame())
    with pytest.raises(ValueError):
        v.build_series(raw, d, raw.freq)


@pytest.mark.parametrize("choice,expected", [("zero", 0.0), ("ffill", 103.0)])
def test_gap_filling(choice, expected):
    parsed, raw, d = diag(monthly(12).drop(index=[4]))
    s = v.build_series(raw, d, raw.freq, gap_choice=choice)
    assert len(s.demand) == 12 and s.demand[pd.Period("2022-05", "M")] == expected
    assert s.notes[0].kind == "gaps"


def test_rollup_in_build_drops_partial_month():
    days = pd.date_range("2024-01-01", "2024-03-15", freq="D")
    f = pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Sales": 1.0})
    parsed, raw, d = diag(f)
    s = v.build_series(raw, d, raw.freq, rollup_to="M")
    assert list(s.demand.values) == [31, 29]
    assert s.freq.code == "M" and any(n.kind == "rollup" for n in s.notes)
    assert any(m.code == "partial_periods" for m in s.messages)


@pytest.mark.parametrize("values,expected", [
    (["2024-03-01", "2024-04-01"], "2024-04-01"),          # ISO
    (["01/03/2024", "01/04/2024"], "2024-04-01"),          # constant first -> day-first
    (["03/01/2024", "04/01/2024"], "2024-04-01"),          # constant second -> month-first
    (["25/03/2024", "26/03/2024"], "2024-03-26"),          # first > 12
    (["Mar-2024", "Apr-2024"], "2024-04-01"),              # month names
    ([2015, 2016], "2016-01-01"),                          # year numbers
])
def test_parse_dates(values, expected):
    parsed = v.parse_dates(pd.Series(values))
    assert parsed.iloc[1].strftime("%Y-%m-%d") == expected
