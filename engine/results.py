"""One run of the whole method, from a clean series to the final forecast.

`run(series, settings)` returns a `RunResult` holding every intermediate result,
so the UI only reads from it and a later report or audit panel can serialise it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import outliers as outl
from . import pipeline, ranges, recommend, seasonality, split, tracking
from .models import ORDER, Availability, FitResult, availability
from .pipeline import MethodSettings
from .validate import SeriesData

SMOOTHING = ("ses", "holt", "winters")


@dataclass
class RunSettings:
    test_length: int | None = None  # None: default (one season / 20%)
    season: int | None = None  # None: default for the frequency
    use_default_season: bool = True
    outlier_z: float = 3.0
    outlier_replacement: str = "none"
    methods: dict[str, MethodSettings] = field(default_factory=dict)
    chosen: str | None = None  # None: the recommended method
    horizon: int | None = None  # None: default
    range_level: int = 80
    future_driver: list[float] | None = None  # user-edited future driver values

    def method(self, key: str) -> MethodSettings:
        return self.methods.get(key) or MethodSettings()


@dataclass
class FinalForecast:
    method: str
    name: str
    table: pd.DataFrame  # index period; columns forecast, lower, upper
    basis: ranges.RangeBasis
    level: int
    driver_values: pd.Series | None
    driver_source: str | None


@dataclass
class RunResult:
    series: SeriesData
    settings: RunSettings
    s: int | None
    season_check: seasonality.SeasonCheck
    split: split.Split
    outliers: outl.OutlierResult
    train: pd.Series  # training data used for fitting (after any correction)
    availability: dict[str, Availability]
    fits: dict[str, FitResult]
    test_forecasts: dict[str, pd.Series]
    comparison: recommend.Comparison
    correction: recommend.CorrectionEffect | None
    tracking: tracking.Tracking
    final: FinalForecast
    max_horizon: int
    start_values: dict[str, object]  # computed (not overridden) start values per smoothing method

    @property
    def chosen(self) -> str:
        return self.final.method


def max_horizon(freq_code: str, s: int | None) -> int:
    return 2 * s if s and freq_code != "Y" else 10


def default_horizon(freq_code: str, s: int | None) -> int:
    return s if s and freq_code != "Y" else 5


def future_driver(series: SeriesData, h: int, edited: list[float] | None = None) -> tuple[pd.Series, str]:
    """Driver values for the next h periods: user edits, else the file's future rows,
    else the last known value repeated. Returns the values and a one-line source note."""
    last = series.demand.index[-1]
    idx = pd.PeriodIndex([last + i for i in range(1, h + 1)], freq=series.demand.index.freq)
    if edited is not None and len(edited) >= h:
        return pd.Series(np.asarray(edited[:h], dtype=float), index=idx), "Edited by you"
    last_known = float(series.driver.dropna().iloc[-1])
    values = pd.Series(last_known, index=idx, dtype=float)
    from_file = 0
    if series.future_driver is not None and len(series.future_driver):
        for p, v in series.future_driver.items():
            if p in values.index and pd.notna(v):
                values[p] = float(v)
                from_file += 1
    if from_file == h:
        return values, "From the future rows in your file"
    if from_file:
        return values, f"{from_file} from your file; the rest repeat the last known value ({last_known:g})"
    return values, f"No future values in your file: the last known value ({last_known:g}) is repeated"


def _fit_all(train, s, settings, avail, driver, driver_name):
    fits = {}
    for key in ORDER:
        if not avail[key].available:
            continue
        try:
            fits[key] = pipeline.fit_method(key, train, s, settings.method(key), driver, driver_name)
        except ValueError as exc:
            avail[key] = Availability(False, str(exc))
    if "snaive" in fits and fits["snaive"].extras.get("fallback"):
        # Without a season, seasonal naive is the naive forecast: show it once.
        del fits["snaive"]
        avail["snaive"] = Availability(False, "No seasonal pattern: same as naive")
    return fits


def _test_forecasts(fits, test, driver):
    out = {}
    for key, fit in fits.items():
        drv = driver.reindex(test.index) if (key == "regression" and driver is not None) else None
        out[key] = pipeline.forecast_ahead(fit, len(test), drv).set_axis(test.index)
    return out


def run(series: SeriesData, settings: RunSettings | None = None) -> RunResult:
    settings = settings or RunSettings()
    freq = series.freq
    demand = series.demand
    default_s = freq.default_season
    s = default_s if settings.use_default_season else settings.season
    test_len = settings.test_length or split.default_test_length(len(demand), s, freq.code)
    sp = split.split(demand, test_len)
    check = seasonality.check(sp.train, freq.code)
    if s and len(sp.train) < s:
        s_eff = None
    else:
        s_eff = s

    out = outl.detect(sp.train, s_eff, settings.outlier_z, settings.outlier_replacement)
    train = out.corrected
    driver = series.driver
    driver_name = series.driver_name or "driver"
    avail = availability(train, s_eff, driver is not None)
    if avail["winters"].available and (sp.test <= 0).any():
        # The final forecast refits on all data, which must also be above zero.
        avail["winters"] = Availability(False, "Multiplicative seasonality needs demand above zero")
    fits = _fit_all(train, s_eff, settings, avail, driver, driver_name)
    test_fc = _test_forecasts(fits, sp.test, driver)
    comparison = recommend.compare(sp.test, test_fc, train, s_eff)

    def fit_and_forecast(method, tr):
        fit = pipeline.fit_method(method, tr, s_eff, settings.method(method), driver, driver_name)
        drv = driver.reindex(sp.test.index) if method == "regression" else None
        return pipeline.forecast_ahead(fit, len(sp.test), drv)

    correction = recommend.correction_effect(comparison.recommended, sp.train, train, sp.test, fit_and_forecast)

    chosen = settings.chosen if settings.chosen in comparison.scores else comparison.recommended
    track = tracking.running(sp.test, comparison.forecasts[chosen])

    # Final forecast: refit on training (corrected) + test, keep parameters.
    full = pd.concat([train, sp.test])
    h_max = max_horizon(freq.code, s_eff)
    h = min(settings.horizon or default_horizon(freq.code, s_eff), h_max)
    parts = _parts(chosen, comparison)
    needs_driver = "regression" in parts
    drv_future, drv_source = (future_driver(series, h, settings.future_driver) if needs_driver else (None, None))
    point = None
    for key in parts:
        fit = pipeline.fit_method(key, full, s_eff, settings.method(key), driver, driver_name)
        fc = pipeline.forecast_ahead(fit, h, drv_future)
        point = fc if point is None else point + fc
    point = point / len(parts)
    sc = comparison.scores[chosen]
    basis = ranges.basis(sc.mape, sc.wmape)
    table = ranges.build(point, basis.r, settings.range_level)
    final = FinalForecast(chosen, comparison.names[chosen], table, basis, settings.range_level,
                          drv_future, drv_source)

    starts = {k: pipeline.computed_start(k, train, s_eff) for k in SMOOTHING if k in fits}
    return RunResult(series, settings, s_eff, check, sp, out, train, avail, fits, test_fc, comparison,
                     correction, track, final, h_max, starts)


def _parts(chosen: str, comparison: recommend.Comparison) -> list[str]:
    if chosen != recommend.COMBO:
        return [chosen]
    candidates = [k for k in comparison.scores if k not in ("naive", "snaive", recommend.COMBO)]
    return sorted(candidates, key=lambda k: comparison.scores[k].rank_value)[:2]


def forecast_csv(result: RunResult) -> bytes:
    t = result.final.table
    frame = pd.DataFrame({
        "period": [str(p) for p in t.index],
        "forecast": t["forecast"].round(3).to_numpy(),
        "lower": t["lower"].round(3).to_numpy(),
        "upper": t["upper"].round(3).to_numpy(),
        "method": result.final.name,
        "range_level": f"{result.final.level}%",
    })
    return frame.to_csv(index=False).encode("utf-8")
