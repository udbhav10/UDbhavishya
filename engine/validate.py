"""Turning a raw table + column mapping into one clean series per SKU.

Pipeline:
    parse(frame, mapping)          -> Parsed: global messages + one SkuRaw per SKU
    diagnose(sku_raw)              -> Diagnosis: duplicates, gaps, messages for one SKU
    build_series(sku_raw, ...)     -> SeriesData: regular period-indexed series + change notes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import frequency as fq
from .messages import ChangeNote, Message

SINGLE_SKU = "All data"

DUPLICATE_CHOICES = {
    "sum": "Add them up",
    "first": "Keep the first row",
    "last": "Keep the last row",
    "mean": "Average them",
}
GAP_CHOICES = {
    "zero": "Fill with 0",
    "ffill": "Carry the last value forward",
}


@dataclass
class SkuRaw:
    """Rows of one SKU after parsing. `rows` has columns: row, date, demand, driver."""

    sku: str
    rows: pd.DataFrame
    freq: fq.Frequency | None


@dataclass
class Parsed:
    skus: dict[str, SkuRaw]
    messages: list[Message]
    has_sku_column: bool
    has_driver: bool

    @property
    def freq(self) -> fq.Frequency | None:
        found = {s.freq for s in self.skus.values() if s.freq is not None}
        return next(iter(found)) if len(found) == 1 else None


@dataclass
class Diagnosis:
    periods: pd.PeriodIndex  # period of each history row (aligned with history)
    history: pd.DataFrame  # rows with demand (or blank demand inside the history range)
    future: pd.DataFrame  # trailing rows with driver but no demand
    duplicates: pd.DataFrame  # flagged duplicate rows with a 'type' column
    gaps: list[pd.Period]
    messages: list[Message] = field(default_factory=list)

    @property
    def n_duplicate_periods(self) -> int:
        return 0 if self.duplicates.empty else self.duplicates["period"].nunique()


@dataclass
class SeriesData:
    sku: str
    freq: fq.Frequency
    demand: pd.Series  # PeriodIndex, regular, no gaps
    driver: pd.Series | None  # aligned with demand
    future_driver: pd.Series | None  # periods after demand's end
    notes: list[ChangeNote]
    messages: list[Message]
    driver_name: str | None = None


# ---------------------------------------------------------------- parsing

def _row_numbers(index: pd.Index) -> pd.Series:
    # Spreadsheet row numbers: header is row 1, first data row is row 2.
    return pd.Series(index + 2, index=index)


def parse_dates(col: pd.Series) -> pd.Series:
    """Parse a date column into Timestamps (NaT where unreadable).

    Handles real datetimes, year numbers (2015), ISO and month-name formats, and
    chooses day-first vs month-first by whichever reads more values and gives
    more regular spacing.
    """
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.to_datetime(col)
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().sum() == col.notna().sum() and numeric.notna().any():
        if numeric.dropna().between(1800, 2200).all() and (numeric.dropna() % 1 == 0).all():
            return pd.to_datetime(numeric.astype("Int64").astype(str), format="%Y", errors="coerce")
    text = col.astype("string").str.strip()
    return pd.to_datetime(text, errors="coerce", dayfirst=day_first(text), format="mixed")


def day_first(text: pd.Series) -> bool:
    """Decide whether numeric dates like 03/04/2024 are day-first.

    ISO dates (2024-03-04) are never day-first. Otherwise: a first number above 12
    means day-first; a second number above 12 means month-first; if only one of the
    two positions changes between rows, the constant one is the day (monthly data is
    usually dated the 1st). If still undecided, day-first is assumed (common outside
    the US) and reported by `describe_date_order`.
    """
    sample = text.dropna()
    if sample.empty or sample.str.match(r"^\d{4}[-/.]").all():
        return False
    parts = sample.str.extract(r"^(\d{1,2})[-/. ](\d{1,2})[-/. ]\d{2,4}")
    parts = parts.dropna().astype(int)
    if parts.empty:
        return False  # month names etc.: order is unambiguous
    first, second = parts[0], parts[1]
    if (first > 12).any():
        return True
    if (second > 12).any():
        return False
    if first.nunique() == 1 and second.nunique() > 1:
        return True
    if second.nunique() == 1 and first.nunique() > 1:
        return False
    return True


def ambiguous_date_order(col: pd.Series) -> bool:
    """True when day-first was assumed rather than proven by the data."""
    if pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_numeric_dtype(col):
        return False
    sample = col.astype("string").str.strip().dropna()
    if sample.str.match(r"^\d{4}[-/.]").all():
        return False
    parts = sample.str.extract(r"^(\d{1,2})[-/. ](\d{1,2})[-/. ]\d{2,4}").dropna().astype(int)
    if parts.empty:
        return False
    first, second = parts[0], parts[1]
    decided = (first > 12).any() or (second > 12).any() or \
        (first.nunique() == 1) != (second.nunique() == 1)
    return not decided


def parse_numbers(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float)
    cleaned = col.astype("string").str.replace(r"[,\s]", "", regex=True).replace("", pd.NA)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse(frame: pd.DataFrame, mapping: dict[str, str | None]) -> Parsed:
    messages: list[Message] = []
    date_col, demand_col = mapping.get("date"), mapping.get("demand")
    driver_col, sku_col = mapping.get("driver"), mapping.get("sku")
    if not date_col or not demand_col:
        return Parsed({}, [Message("error", "mapping", "Choose a date column and a demand column.")], False, False)
    if date_col == demand_col or (driver_col and driver_col in (date_col, demand_col)):
        return Parsed({}, [Message("error", "mapping", "Date, demand and driver must be different columns.")],
                      False, False)

    frame = frame.reset_index(drop=True)
    rows = pd.DataFrame({"row": _row_numbers(frame.index)})
    rows["date"] = parse_dates(frame[date_col])
    if ambiguous_date_order(frame[date_col]):
        messages.append(Message("info", "date_order",
                                "Dates like 03/04 were read as day/month (3 April). "
                                "Use YYYY-MM-DD dates to avoid any doubt."))
    rows["demand"] = parse_numbers(frame[demand_col])
    rows["driver"] = parse_numbers(frame[driver_col]) if driver_col else np.nan

    bad_dates = rows["date"].isna() & frame[date_col].notna()
    if bad_dates.any():
        table = pd.DataFrame({"row": rows.loc[bad_dates, "row"], "date value": frame.loc[bad_dates, date_col]})
        messages.append(Message("error", "bad_dates",
                                f"{bad_dates.sum()} date value(s) could not be read.", table,
                                ["Fix these rows in your file and upload again."]))
    bad_demand = rows["demand"].isna() & frame[demand_col].notna() & \
        (frame[demand_col].astype("string").str.strip() != "")
    if bad_demand.any():
        table = pd.DataFrame({"row": rows.loc[bad_demand, "row"], "demand value": frame.loc[bad_demand, demand_col]})
        messages.append(Message("error", "bad_demand",
                                f"{bad_demand.sum()} demand value(s) are not numbers.", table,
                                ["Fix these rows in your file and upload again."]))
    if driver_col:
        bad_driver = rows["driver"].isna() & frame[driver_col].notna() & \
            (frame[driver_col].astype("string").str.strip() != "")
        if bad_driver.any():
            table = pd.DataFrame({"row": rows.loc[bad_driver, "row"], "driver value": frame.loc[bad_driver, driver_col]})
            messages.append(Message("error", "bad_driver",
                                    f"{bad_driver.sum()} driver value(s) are not numbers.", table,
                                    ["Fix these rows, or choose no driver column."]))
    rows = rows[rows["date"].notna()]

    if sku_col:
        sku_values = frame.loc[rows.index, sku_col].astype("string").str.strip().fillna("(blank)")
        groups = {str(k): g.drop(columns=[]) for k, g in rows.groupby(sku_values, sort=True)}
    else:
        groups = {SINGLE_SKU: rows}
        messages.append(Message("info", "single_sku", "Single product: no SKU column found."))

    skus: dict[str, SkuRaw] = {}
    for name, g in groups.items():
        g = g.sort_values(["date", "row"]).reset_index(drop=True)
        skus[name] = SkuRaw(name, g, fq.detect(g.loc[g["demand"].notna(), "date"]))

    undetected = [n for n, s in skus.items() if s.freq is None]
    if undetected:
        messages.append(Message("error", "irregular",
                                "Could not tell whether the data is daily, weekly, monthly, quarterly or yearly"
                                + ("" if not sku_col else f" for {_list(undetected)}") + ".",
                                actions=["Check the dates are evenly spaced, or set the frequency manually."]))
    found = {s.freq.code for s in skus.values() if s.freq is not None}
    if len(found) > 1:
        table = pd.DataFrame({"SKU": list(skus), "frequency": [s.freq.label if s.freq else "unknown"
                                                              for s in skus.values()]})
        messages.append(Message("error", "mixed_freq",
                                "SKUs have different frequencies; all SKUs must share one.", table,
                                ["Split the file by frequency, or roll the finer SKUs up before uploading."]))
    elif found:
        f = next(s.freq for s in skus.values() if s.freq is not None)
        messages.append(Message("info", "frequency", f"Detected: {f.label}."))
    return Parsed(skus, messages, bool(sku_col), bool(driver_col))


def _list(items: list[str], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


# ---------------------------------------------------------------- diagnosis

def split_future(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing rows after the last demand value that carry a driver are future driver inputs."""
    has_demand = rows["demand"].notna()
    if not has_demand.any():
        return rows.iloc[0:0], rows.iloc[0:0]
    last = has_demand[has_demand].index.max()
    tail = rows.loc[last + 1:]
    future = tail[tail["demand"].isna() & tail["driver"].notna()]
    history = rows.loc[:last]
    return history, future


def diagnose(raw: SkuRaw, freq: fq.Frequency | None = None) -> Diagnosis:
    freq = freq or raw.freq
    history, future = split_future(raw.rows)
    messages: list[Message] = []
    if freq is None or history.empty:
        return Diagnosis(pd.PeriodIndex([], freq="M"), history, future, _empty_dups(), [], messages)

    periods = fq.to_periods(history["date"], freq)
    history = history.assign(period=periods)
    blank = history["demand"].isna()

    # duplicates: two or more rows with demand in the same period
    filled = history[~blank]
    counts = filled.groupby("period")["row"].transform("count")
    dups = filled[counts > 1].copy()
    if not dups.empty:
        kinds = dups.groupby("period").apply(
            lambda g: "exact copy" if g["demand"].nunique() == 1 and g["driver"].nunique(dropna=False) <= 1
            else "conflicting", include_groups=False)
        dups["type"] = dups["period"].map(kinds)
        dups = dups[["row", "period", "date", "demand", "driver", "type"]]
        n_periods = dups["period"].nunique()
        n_exact = int((kinds == "exact copy").sum())
        n_conf = n_periods - n_exact
        where = "" if raw.sku == SINGLE_SKU else f" in {raw.sku}"
        messages.append(Message(
            "error", "duplicates",
            f"{n_periods} duplicate period(s) found{where}: {n_exact} exact copies, {n_conf} conflicting. "
            "Choose how to combine them before forecasting.",
            dups, list(DUPLICATE_CHOICES.values())))
    else:
        dups = _empty_dups()

    # gaps: periods with no demand between first and last period
    present = set(filled["period"])
    full = pd.period_range(filled["period"].min(), filled["period"].max(), freq=periods.freq)
    gaps = [p for p in full if p not in present]
    if gaps:
        table = pd.DataFrame({"missing period": fq.labels(gaps)})
        messages.append(Message("warning", "gaps",
                                f"{len(gaps)} {freq.unit}(s) have no demand value. Choose how to fill them.",
                                table, list(GAP_CHOICES.values())))

    neg = filled[filled["demand"] < 0]
    if not neg.empty:
        messages.append(Message("warning", "negative",
                                f"{len(neg)} period(s) have negative demand (returns?). Check they are correct.",
                                neg[["row", "period", "demand"]]))
    zero = filled[filled["demand"] == 0]
    if not zero.empty:
        messages.append(Message("warning", "zeros",
                                f"{len(zero)} period(s) have zero demand: MAPE will be excluded; "
                                "wMAPE and MASE are used instead.",
                                zero[["row", "period", "demand"]]))
    if not future.empty:
        messages.append(Message("info", "future_driver",
                                f"{len(future)} future row(s) with driver values found; "
                                "they will be used for regression forecasts."))
    return Diagnosis(periods, history, future, dups, gaps, messages)


def _empty_dups() -> pd.DataFrame:
    return pd.DataFrame(columns=["row", "period", "date", "demand", "driver", "type"])


# ---------------------------------------------------------------- building

def build_series(raw: SkuRaw, diagnosis: Diagnosis, freq: fq.Frequency,
                 duplicate_choice: str | None = None, gap_choice: str | None = None,
                 rollup_to: str | None = None, driver_how: str = "mean",
                 driver_name: str | None = None) -> SeriesData:
    """Combine duplicates, fill gaps and optionally roll up. Raises ValueError if a
    needed choice is missing (the UI blocks before calling this)."""
    notes: list[ChangeNote] = []
    hist = diagnosis.history.assign(period=diagnosis.periods)
    hist = hist[hist["demand"].notna()]
    has_driver = hist["driver"].notna().any() or not diagnosis.future.empty

    if diagnosis.n_duplicate_periods:
        if duplicate_choice not in DUPLICATE_CHOICES:
            raise ValueError("A duplicate resolution choice is required.")
        grouped = hist.groupby("period", sort=True)
        demand = getattr(grouped["demand"], duplicate_choice)()
        drv_how = duplicate_choice if duplicate_choice in ("first", "last") else "mean"
        driver = getattr(grouped["driver"], drv_how)()
        notes.append(ChangeNote("duplicates",
                                f"{diagnosis.n_duplicate_periods} duplicate periods: "
                                f"{DUPLICATE_CHOICES[duplicate_choice].lower()}",
                                diagnosis.duplicates))
    else:
        demand = hist.set_index("period")["demand"].sort_index()
        driver = hist.set_index("period")["driver"].sort_index()

    full = pd.period_range(demand.index.min(), demand.index.max(), freq=demand.index.freq)
    demand = demand.reindex(full)
    driver = driver.reindex(full)
    if diagnosis.gaps:
        if gap_choice not in GAP_CHOICES:
            raise ValueError("A gap filling choice is required.")
        demand = demand.fillna(0.0) if gap_choice == "zero" else demand.ffill()
        notes.append(ChangeNote("gaps", f"{len(diagnosis.gaps)} gaps: {GAP_CHOICES[gap_choice].lower()}",
                                pd.DataFrame({"period": fq.labels(diagnosis.gaps)})))
    driver = driver.ffill().bfill() if has_driver else None

    future = None
    if not diagnosis.future.empty:
        fut = diagnosis.future.assign(period=fq.to_periods(diagnosis.future["date"], freq))
        future = fut.groupby("period")["driver"].mean().sort_index()
        future = future[future.index > demand.index.max()]

    messages: list[Message] = []
    out_freq = freq
    if rollup_to and rollup_to != freq.code:
        demand, driver, future, dropped = _rollup_all(demand, driver, future, rollup_to, driver_how)
        out_freq = fq.Frequency(rollup_to)
        notes.append(ChangeNote("rollup", f"rolled up from {freq.label} to {out_freq.label}"))
        if dropped:
            messages.append(Message("info", "partial_periods",
                                    f"Incomplete {out_freq.unit}(s) at the edges were left out: "
                                    f"{', '.join(dropped)}."))
    demand.name = "demand"
    if driver is not None:
        driver.name = driver_name or "driver"
    return SeriesData(raw.sku, out_freq, demand.astype(float), driver, future, notes, messages, driver_name)


def _rollup_all(demand, driver, future, to_code, driver_how):
    new_demand, new_driver = fq.rollup(demand, driver, to_code, driver_how)
    # Drop coarse periods that the source data covers only partly (first/last).
    src_start = demand.index.min().start_time
    src_end = demand.index.max().end_time
    dropped = []
    keep = []
    for p in new_demand.index:
        partial = p.start_time < src_start or p.end_time > src_end
        if partial:
            dropped.append(fq.label(p))
        else:
            keep.append(p)
    new_demand = new_demand.loc[keep]
    if new_driver is not None:
        new_driver = new_driver.reindex(new_demand.index)
    new_future = None
    if future is not None and len(future):
        _, new_future = fq.rollup(future * 0, future, to_code, driver_how)
        new_future = new_future[new_future.index > new_demand.index.max()]
    new_demand.index = pd.PeriodIndex(new_demand.index, freq=to_code)
    return new_demand, new_driver, new_future, dropped
