"""Reading uploaded files, suggesting column mappings and building the template."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd

from .messages import Message

MAX_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = (".csv", ".xlsx")

DATE_ALIASES = ("date", "period", "month", "week", "day", "year", "quarter", "time", "ds", "timestamp")
DEMAND_ALIASES = ("demand", "sales", "sale", "qty", "quantity", "units", "volume", "y", "value", "orders", "shipments")


def normalise_name(name: object) -> str:
    """Lower-case and strip spaces, underscores and dashes: ' S K U ' -> 'sku'."""
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


@dataclass
class ReadResult:
    frame: pd.DataFrame | None
    messages: list[Message]


def read_upload(filename: str, data: bytes) -> ReadResult:
    """Read a CSV or XLSX upload. Returns the raw table or an error message."""
    name = filename.lower()
    if len(data) > MAX_BYTES:
        size_mb = len(data) / 1024 / 1024
        return ReadResult(None, [Message("error", "file_too_large",
                                         f"File is {size_mb:.0f} MB; the limit is 50 MB.")])
    if not name.endswith(ALLOWED_EXTENSIONS):
        return ReadResult(None, [Message("error", "file_type",
                                         "Unsupported file type. Upload a .csv or .xlsx file.")])
    try:
        if name.endswith(".csv"):
            frame = _read_csv(data)
        else:
            frame = pd.read_excel(io.BytesIO(data), sheet_name=0)
    except Exception as exc:  # noqa: BLE001 - any parse failure is reported, not raised
        return ReadResult(None, [Message("error", "unreadable",
                                         f"Could not read the file ({type(exc).__name__}). "
                                         "Check it is a valid CSV or Excel file.")])
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty or len(frame.columns) < 2:
        return ReadResult(None, [Message("error", "empty",
                                         "The file needs at least a date column and a demand column with data.")])
    return ReadResult(frame.reset_index(drop=True), [])


def _read_csv(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    # sep=None sniffs commas, semicolons and tabs.
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def find_sku_column(frame: pd.DataFrame) -> str | None:
    for col in frame.columns:
        if normalise_name(col) == "sku":
            return col
    return None


def suggest_mapping(frame: pd.DataFrame) -> dict[str, str | None]:
    """Guess which columns hold date, demand, driver and SKU."""
    cols = list(frame.columns)
    norm = {c: normalise_name(c) for c in cols}
    sku = find_sku_column(frame)

    def first_match(aliases: tuple[str, ...], exclude: set) -> str | None:
        for alias in aliases:
            for c in cols:
                if c not in exclude and norm[c] == alias:
                    return c
        for alias in aliases:
            for c in cols:
                if c not in exclude and alias in norm[c] and len(alias) > 2:
                    return c
        return None

    taken: set = {sku} if sku else set()
    date = first_match(DATE_ALIASES, taken)
    if date is None:
        date = next((c for c in cols if c not in taken and pd.api.types.is_datetime64_any_dtype(frame[c])), None)
    if date:
        taken.add(date)
    demand = first_match(DEMAND_ALIASES, taken)
    if demand is None:
        demand = next((c for c in cols if c not in taken and pd.api.types.is_numeric_dtype(frame[c])), None)
    if demand:
        taken.add(demand)
    driver = next((c for c in cols if c not in taken and _mostly_numeric(frame[c])), None)
    return {"date": date, "demand": demand, "driver": driver, "sku": sku}


def _mostly_numeric(col: pd.Series) -> bool:
    values = pd.to_numeric(col, errors="coerce")
    present = col.notna().sum()
    return present > 0 and values.notna().sum() >= 0.8 * present


def template_csv() -> bytes:
    """Example file: monthly history plus three future rows with driver only."""
    months = pd.period_range("2023-01", periods=15, freq="M")
    demand = [412, 398, 455, 470, 520, 610, 380, 365, 540, 690, 720, 505, None, None, None]
    driver = [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0]
    frame = pd.DataFrame({
        "date": [m.start_time.strftime("%Y-%m-%d") for m in months],
        "demand": demand,
        "driver": driver,
    })
    return frame.to_csv(index=False).encode("utf-8")
