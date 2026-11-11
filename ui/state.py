"""Session state, cached data loading, and the per-rerun pipeline.

Session state holds the user's choices only. Data and results are recomputed
(cheaply, with caching for file reading and parsing) on every rerun.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from engine import demo, frequency as fq, io as eio, results, split
from engine import validate as v
from engine.messages import ChangeNote, Message, has_errors
from engine.pipeline import MethodSettings

STEPS = ["Data", "Split & clean", "Models", "Compare", "Forecast"]


def init() -> None:
    defaults = {
        "step": STEPS[0],
        "source": "demo",
        "demo_key": "paint",
        "mapping": None,
        "mapping_for": None,
        "sku": None,
        "dup_choice": {},
        "gap_choice": {},
        "freq_override": None,
        "rollup_to": None,
        "driver_how": "mean",
        "test_length": None,
        "season": None,
        "outlier_z": 3.0,
        "outlier_replacement": "none",
        "params": {},
        "overrides": {},
        "damped": {},
        "tuned": {},
        "chosen": None,
        "horizon": None,
        "range_level": 80,
        "future_driver": None,
        "seen": {},
    }
    for k, val in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = val


def reset_downstream(level: str) -> None:
    """Clear choices that no longer apply after the data changes."""
    from .widgets import forget

    ss = st.session_state
    forget(("w_",))
    if level in ("data",):
        ss.sku = None
        ss.dup_choice, ss.gap_choice = {}, {}
        ss.freq_override, ss.rollup_to = None, None
    if level in ("data", "series"):
        ss.test_length, ss.season = None, None
        ss.overrides, ss.tuned = {}, {}
        ss.chosen, ss.horizon, ss.future_driver = None, None, None


# ---------------------------------------------------------------- cached loaders

@st.cache_data(show_spinner=False, max_entries=8)
def read_file(name: str, data: bytes) -> eio.ReadResult:
    return eio.read_upload(name, data)


@st.cache_data(show_spinner=False, max_entries=8)
def load_demo(key: str) -> pd.DataFrame:
    return demo.load(key)


@st.cache_data(show_spinner=False, max_entries=16)
def parse(frame: pd.DataFrame, mapping_json: str) -> v.Parsed:
    return v.parse(frame, json.loads(mapping_json))


# ---------------------------------------------------------------- data context

@dataclass
class DataContext:
    source_label: str
    frame: pd.DataFrame | None = None
    mapping: dict | None = None
    parsed: v.Parsed | None = None
    raw: v.SkuRaw | None = None
    freq: fq.Frequency | None = None
    diagnosis: v.Diagnosis | None = None
    series: v.SeriesData | None = None
    messages: list[Message] = field(default_factory=list)
    unit: str = ""
    is_demo: bool = False

    @property
    def blocked(self) -> bool:
        return self.series is None

    @property
    def block_reason(self) -> str:
        errors = [m for m in self.messages if m.level == "error"]
        if errors:
            return errors[0].summary
        if self.diagnosis is not None and self.diagnosis.n_duplicate_periods:
            return "Resolve the duplicate periods first."
        if self.diagnosis is not None and self.diagnosis.gaps:
            return "Choose how to fill the missing periods first."
        return "Load data first."


def fingerprint(*parts) -> str:
    return hashlib.md5(json.dumps(parts, default=str, sort_keys=True).encode()).hexdigest()[:12]


def data_context() -> DataContext:
    ss = st.session_state
    if ss.source == "demo":
        product = demo.PRODUCTS[ss.demo_key]
        frame = load_demo(ss.demo_key)
        ctx = DataContext(f"Demo: {product.name}", frame, unit=product.unit, is_demo=True)
        default_map = demo.mapping(ss.demo_key)
        map_key = f"demo:{ss.demo_key}"
    else:
        name, data = ss.get("upload_name"), ss.get("upload_bytes")
        if data is None:
            return DataContext("No file yet")
        res = read_file(name, data)
        if res.frame is None:
            return DataContext(name, messages=res.messages)
        frame = res.frame
        ctx = DataContext(name, frame)
        default_map = eio.suggest_mapping(frame)
        map_key = f"file:{name}:{hashlib.md5(data).hexdigest()}"

    if ss.mapping_for != map_key:
        ss.mapping, ss.mapping_for = default_map, map_key
        reset_downstream("data")
    ctx.mapping = dict(ss.mapping)
    ctx.mapping["sku"] = eio.find_sku_column(frame)
    ctx.parsed = parse(frame, json.dumps(ctx.mapping))
    ctx.messages = list(ctx.parsed.messages)
    if has_errors(ctx.messages) or not ctx.parsed.skus:
        return ctx

    skus = list(ctx.parsed.skus)
    if ss.sku not in skus:
        ss.sku = skus[0]
    ctx.raw = ctx.parsed.skus[ss.sku]
    detected = ctx.raw.freq
    ctx.freq = fq.Frequency(ss.freq_override, detected.week_start if detected else 0) \
        if ss.freq_override else detected
    if ctx.freq is None:
        return ctx
    ctx.diagnosis = v.diagnose(ctx.raw, ctx.freq)
    ctx.messages += ctx.diagnosis.messages
    dup = ss.dup_choice.get(ss.sku)
    gap = ss.gap_choice.get(ss.sku)
    if ctx.diagnosis.n_duplicate_periods and dup is None:
        return ctx
    if ctx.diagnosis.gaps and gap is None:
        return ctx
    if ctx.diagnosis.history["demand"].notna().sum() < 4:
        ctx.messages.append(Message("error", "too_short", "At least 4 periods of demand are needed."))
        return ctx
    driver_name = ctx.mapping.get("driver")
    rollup = ss.rollup_to if ss.rollup_to and ss.rollup_to != ctx.freq.code else None
    ctx.series = v.build_series(ctx.raw, ctx.diagnosis, ctx.freq, dup, gap, rollup, ss.driver_how, driver_name)
    ctx.messages += ctx.series.messages
    if len(ctx.series.demand) < 4:
        ctx.messages.append(Message("error", "too_short", "At least 4 periods are needed after roll-up."))
        ctx.series = None
    return ctx


# ---------------------------------------------------------------- run

def settings_from_state(series: v.SeriesData) -> results.RunSettings:
    ss = st.session_state
    methods = {}
    for key in ("ses", "holt", "winters", "regression"):
        params = dict(ss.params.get(key, {}))
        if key in ("holt", "winters") and not ss.damped.get(key):
            params["phi"] = 1.0
        methods[key] = MethodSettings(params, dict(ss.overrides.get(key, {})))
    return results.RunSettings(
        test_length=ss.test_length,
        season=ss.season,
        use_default_season=ss.season is None,
        outlier_z=ss.outlier_z,
        outlier_replacement=ss.outlier_replacement,
        methods=methods,
        chosen=ss.chosen,
        horizon=ss.horizon,
        range_level=ss.range_level,
        future_driver=ss.future_driver,
    )


def run(series: v.SeriesData) -> results.RunResult:
    return results.run(series, settings_from_state(series))


def all_notes(series: v.SeriesData, result: results.RunResult | None) -> list[ChangeNote]:
    notes = list(series.notes)
    if result is not None and len(result.outliers.corrections):
        notes.append(ChangeNote("outliers", f"{len(result.outliers.corrections)} outliers replaced",
                                result.outliers.corrections))
    return notes


def stage_fingerprints(ctx: DataContext, result: results.RunResult | None) -> dict[str, str]:
    ss = st.session_state
    return {
        "data": fingerprint(ctx.source_label, ss.sku, ss.dup_choice, ss.gap_choice, ss.freq_override,
                            ss.rollup_to, ss.mapping),
        "test period": fingerprint(ss.test_length, ss.season),
        "outlier settings": fingerprint(ss.outlier_z, ss.outlier_replacement),
        "model settings": fingerprint(ss.params, ss.overrides, ss.damped),
    }


UPSTREAM = {
    "Data": [],
    "Split & clean": ["data"],
    "Models": ["data", "test period", "outlier settings"],
    "Compare": ["data", "test period", "outlier settings", "model settings"],
    "Forecast": ["data", "test period", "outlier settings", "model settings"],
}


def changed_since_seen(step: str, fps: dict[str, str]) -> list[str]:
    seen = st.session_state.seen.get(step)
    st.session_state.seen[step] = fps
    if seen is None:
        return []
    return [k for k in UPSTREAM[step] if seen.get(k) != fps.get(k)]


def default_test_length(series: v.SeriesData) -> int:
    return split.default_test_length(len(series.demand), series.freq.default_season, series.freq.code)
