"""Outlier detection and correction on the training period (slides 52-54).

Only the training series is ever passed in, so the test period cannot be checked
or altered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REPLACEMENTS = {
    "none": "Leave unchanged",
    "reference": "Replace with the reference forecast",
    "neighbours": "Average with its two neighbours",
    "cap": "Cap at the limit",
}


@dataclass
class OutlierResult:
    mode: str  # "units" or "percent"
    reference: pd.Series
    lower: pd.Series
    upper: pd.Series
    score: pd.Series  # z = gap / sigma
    flags: pd.Series  # bool
    sigma: float  # in units (mode units) or as a fraction (mode percent)
    spread: float  # MAD (units) or MAPE (fraction)
    corrected: pd.Series
    corrections: pd.DataFrame  # period, original, new, z

    @property
    def count(self) -> int:
        return int(self.flags.sum())


def reference_forecast(train: pd.Series, season: int | None) -> pd.Series:
    """Non-seasonal: the training average.

    Seasonal: the same season in the other training years. Each value is first
    expressed relative to a straight trend line, so growth does not distort the
    comparison; the median (not the mean) of the other years' ratios is used, so
    one outlier cannot shift its neighbours' reference (slide 53)."""
    y = train.to_numpy(dtype=float)
    if not season:
        return pd.Series(np.full(len(y), y.mean()), index=train.index)
    t = np.arange(len(y))
    slope, intercept = np.polyfit(t, y, 1)
    line = intercept + slope * t
    line = np.where(line > 0, line, y.mean())
    ratio = y / line
    ref = np.empty(len(y))
    for i in range(len(y)):
        same = [ratio[j] for j in range(i % season, len(y), season) if j != i]
        ref[i] = line[i] * (np.median(same) if same else ratio[i])
    return pd.Series(ref, index=train.index)


def detect(train: pd.Series, season: int | None, z_limit: float = 3.0,
           replacement: str = "none") -> OutlierResult:
    if replacement not in REPLACEMENTS:
        raise ValueError(f"Unknown replacement: {replacement}")
    seasonal = bool(season) and len(train) >= 2 * season
    ref = reference_forecast(train, season if seasonal else None)
    y = train.astype(float)
    if seasonal and (ref > 0).all():
        mode = "percent"
        pct = (y - ref) / ref
        spread = float(pct.abs().mean())
        sigma = 1.25 * spread
        score = pct / sigma if sigma > 0 else pct * 0
        lower = ref * (1 - z_limit * sigma)
        upper = ref * (1 + z_limit * sigma)
    else:
        mode = "units"
        gap = y - ref
        spread = float(gap.abs().mean())
        sigma = 1.25 * spread
        score = gap / sigma if sigma > 0 else gap * 0
        lower = ref - z_limit * sigma
        upper = ref + z_limit * sigma
    flags = score.abs() > z_limit
    corrected = _correct(y, ref, lower, upper, flags, replacement)
    changed = flags & (corrected != y)
    corrections = pd.DataFrame({
        "period": [str(p) for p in y.index[changed]],
        "original": y[changed].to_numpy(),
        "new": corrected[changed].to_numpy(),
        "z": score[changed].round(2).to_numpy(),
    })
    return OutlierResult(mode, ref, lower, upper, score, flags, sigma, spread, corrected, corrections)


def _correct(y, ref, lower, upper, flags, replacement) -> pd.Series:
    out = y.copy()
    if replacement == "none":
        return out
    for i in np.flatnonzero(flags.to_numpy()):
        if replacement == "reference":
            out.iloc[i] = ref.iloc[i]
        elif replacement == "cap":
            out.iloc[i] = min(max(y.iloc[i], lower.iloc[i]), upper.iloc[i])
        elif replacement == "neighbours":
            window = y.iloc[max(0, i - 1): i + 2]
            out.iloc[i] = float(window.mean())
    return out
