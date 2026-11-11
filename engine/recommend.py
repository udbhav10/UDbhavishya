"""Benchmark comparison, top-two combination and the recommendation rule."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .models import METHOD_NAMES

COMBO = "combo"


@dataclass
class Comparison:
    benchmark: str
    scores: dict[str, metrics.Scores]
    forecasts: dict[str, pd.Series]
    names: dict[str, str]
    recommended: str
    beats: dict[str, bool]
    fva: dict[str, float | None]  # percentage points vs benchmark (positive = better)
    sentence: str
    no_winner: bool

    def table(self) -> pd.DataFrame:
        rows = []
        for key, sc in self.scores.items():
            arrow, label = metrics.bias_direction(sc.bias, sc.mad)
            rows.append({"key": key, "method": self.names[key], "wMAPE": sc.wmape, "MAD": sc.mad,
                         "MAPE": sc.mape, "RMSE": sc.rmse, "bias": sc.bias, "bias direction": f"{arrow} {label}",
                         "tracking signal": sc.ts, "MASE": sc.mase, "beats benchmark": self.beats.get(key),
                         "value added (pts)": self.fva.get(key)})
        return pd.DataFrame(rows).sort_values("wMAPE", na_position="last").reset_index(drop=True)


def benchmark_for(s: int | None, train_len: int) -> str:
    return "snaive" if s and train_len >= s else "naive"


def compare(test: pd.Series, forecasts: dict[str, pd.Series], train: pd.Series, s: int | None) -> Comparison:
    """Score every method's test forecasts, add the top-two average, and recommend."""
    bench = benchmark_for(s, len(train))
    if bench not in forecasts:
        raise ValueError("The benchmark forecast is required.")
    scale = metrics.naive_scale(train, s)
    forecasts = dict(forecasts)
    names = {k: METHOD_NAMES.get(k, k) for k in forecasts}
    scores = {k: metrics.score(test.values, f.values, scale) for k, f in forecasts.items()}

    candidates = [k for k in scores if k not in ("naive", "snaive")]
    ranked = sorted(candidates, key=lambda k: scores[k].rank_value)
    if len(ranked) >= 2:
        a, b = ranked[:2]
        forecasts[COMBO] = (forecasts[a] + forecasts[b]) / 2
        names[COMBO] = f"Average of {names[a]} + {names[b]}"
        scores[COMBO] = metrics.score(test.values, forecasts[COMBO].values, scale)

    bench_value = scores[bench].rank_value
    beats = {k: scores[k].rank_value < bench_value for k in scores if k != bench}
    pct = scores[bench].wmape is not None
    fva = {k: ((bench_value - sc.rank_value) * 100 if pct else None) for k, sc in scores.items() if k != bench}

    winners = [k for k, ok in beats.items() if ok]
    if winners:
        best = min(winners, key=lambda k: scores[k].rank_value)
        margin = f"beats {names[bench].lower()} by {fva[best]:.1f} pts" if pct else \
            f"beats {names[bench].lower()} (MAD {scores[best].mad:.1f} vs {scores[bench].mad:.1f})"
        sentence = f"Recommended: {names[best]} · {_metric(scores[best])} · {margin}"
        no_winner = False
    else:
        best = bench
        sentence = (f"Recommended: {names[bench]} · {_metric(scores[bench])} · "
                    f"no method improved on this free benchmark")
        no_winner = True
    return Comparison(bench, scores, forecasts, names, best, beats, fva, sentence, no_winner)


def _metric(sc: metrics.Scores) -> str:
    return f"wMAPE {sc.wmape * 100:.1f}%" if sc.wmape is not None else f"MAD {sc.mad:.1f}"


@dataclass
class CorrectionEffect:
    method: str
    wmape_without: float | None
    wmape_with: float | None


def correction_effect(method: str, raw_train: pd.Series, corrected_train: pd.Series, test: pd.Series,
                      fit_and_forecast) -> CorrectionEffect | None:
    """Test wMAPE of `method` fitted on raw vs corrected training data.

    fit_and_forecast(method, train) -> test-period forecast Series.
    Returns None when no value was corrected.
    """
    if np.allclose(raw_train.values, corrected_train.values):
        return None
    if method == COMBO:
        return None
    without = metrics.score(test.values, fit_and_forecast(method, raw_train).values).wmape
    with_ = metrics.score(test.values, fit_and_forecast(method, corrected_train).values).wmape
    return CorrectionEffect(method, without, with_)
