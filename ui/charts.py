"""Chart builders. Each takes engine results and returns a Plotly figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import tracking

from .style import layout, palette


def x(index: pd.PeriodIndex):
    """Plot periods at their start date."""
    return index.to_timestamp(how="start")


def _actual(fig, s: pd.Series, name="Actual", **kw):
    p = palette()
    fig.add_trace(go.Scatter(x=x(s.index), y=s.values, name=name, mode="lines",
                             line=dict(color=p.ink, width=2), hovertemplate="%{y:,.1f}", **kw))


def _forecast(fig, s: pd.Series, name="Forecast", dash="dash", color=None, width=2, opacity=1.0, **kw):
    p = palette()
    fig.add_trace(go.Scatter(x=x(s.index), y=s.values, name=name, mode="lines", opacity=opacity,
                             line=dict(color=color or p.primary, width=width, dash=dash),
                             hovertemplate="%{y:,.1f}", **kw))


def _test_shade(fig, test_index: pd.PeriodIndex, row=None, col=None):
    if len(test_index) == 0:
        return
    p = palette()
    start = test_index[0].start_time
    end = test_index[-1].end_time
    kw = dict(row=row, col=col) if row else {}
    fig.add_vrect(x0=start, x1=end, fillcolor=p.test_wash, line_width=0, layer="below",
                  annotation_text="Test period", annotation_position="top left",
                  annotation_font=dict(color=p.muted, size=11), **kw)


def series_chart(demand: pd.Series, unit: str = "") -> go.Figure:
    fig = go.Figure()
    _actual(fig, demand, name="Demand")
    return layout(fig, 300, unit, legend=False)


def acf_chart(acf: pd.Series, threshold: float, s: int | None, peak: int | None) -> go.Figure:
    p = palette()
    colors = [p.primary if (s and lag == s) else p.bar for lag in acf.index]
    fig = go.Figure(go.Bar(x=acf.index, y=acf.values, marker=dict(color=colors, cornerradius=4),
                           hovertemplate="Lag %{x}: %{y:.2f}<extra></extra>", name="Autocorrelation"))
    for sign in (1, -1):
        fig.add_hline(y=sign * threshold, line=dict(color=p.muted, width=1, dash="dot"))
    if s and s in acf.index:
        fig.add_annotation(x=s, y=float(acf[s]), text=f"s = {s}", showarrow=False, yshift=12,
                           font=dict(color=p.ink2, size=12))
    fig.update_xaxes(title=dict(text="Lag (periods back)", font=dict(color=p.muted, size=12)))
    return layout(fig, 240, "Correlation", legend=False)


def split_chart(train: pd.Series, test: pd.Series, out) -> go.Figure:
    """Training and test with the outlier limit band and flagged points."""
    p = palette()
    fig = go.Figure()
    idx = x(train.index)
    fig.add_trace(go.Scatter(x=idx, y=out.upper.values, mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=idx, y=out.lower.values, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=p.primary_band, name="Normal range", hoverinfo="skip"))
    _actual(fig, pd.concat([train, test]), name="Actual")
    flagged = out.flags[out.flags].index
    if len(flagged):
        vals = train[flagged]
        fig.add_trace(go.Scatter(x=x(flagged), y=vals.values, mode="markers", name="Outlier",
                                 marker=dict(symbol="x-thin", size=12, line=dict(width=3, color=p.outlier)),
                                 customdata=out.score[flagged].round(1).values,
                                 hovertemplate="Outlier: %{y:,.1f} (z = %{customdata})<extra></extra>"))
        changed = out.corrected[flagged] != vals
        if changed.any():
            fixed = out.corrected[flagged][changed]
            fig.add_trace(go.Scatter(x=x(fixed.index), y=fixed.values, mode="markers", name="Corrected value",
                                     marker=dict(symbol="circle", size=9, color=p.second,
                                                 line=dict(width=2, color=p.second)),
                                     hovertemplate="Corrected to %{y:,.1f}<extra></extra>"))
    _test_shade(fig, test.index)
    return layout(fig, 340)


def fit_chart(train: pd.Series, test: pd.Series, fitted: pd.Series, test_fc: pd.Series,
              benchmark: pd.Series | None, name: str) -> go.Figure:
    p = palette()
    fig = go.Figure()
    _actual(fig, pd.concat([train, test]))
    if benchmark is not None:
        _forecast(fig, benchmark, name="Seasonal naive (benchmark)", dash="dot", color=p.muted, width=1.5,
                  opacity=0.8)
    _forecast(fig, pd.concat([fitted.dropna(), test_fc]), name=name)
    _test_shade(fig, test.index)
    return layout(fig, 340)


def components_chart(components: dict[str, pd.Series]) -> go.Figure:
    p = palette()
    names = [k for k in ("level", "trend", "season") if k in components]
    titles = {"level": "Level", "trend": "Trend (per period)", "season": "Seasonal index"}
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=[titles[k] for k in names],
                        horizontal_spacing=0.08)
    for i, k in enumerate(names, start=1):
        s = components[k]
        fig.add_trace(go.Scatter(x=x(s.index), y=s.values, mode="lines", line=dict(color=p.primary, width=2),
                                 name=titles[k], hovertemplate="%{y:,.3f}<extra></extra>"), row=1, col=i)
    for ann in fig.layout.annotations:
        ann.font = dict(color=p.ink2, size=12)
    fig = layout(fig, 200, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def compare_chart(table: pd.DataFrame, benchmark_key: str, recommended: str) -> go.Figure:
    """Horizontal bars of test wMAPE (or MAD), best at the top, benchmark as a reference line."""
    p = palette()
    use_pct = table["wMAPE"].notna().all()
    col = "wMAPE" if use_pct else "MAD"
    t = table.sort_values(col, ascending=False)
    values = t[col] * (100 if use_pct else 1)
    labels = [("★ " if k == recommended else "") + m for k, m in zip(t["key"], t["method"])]
    colors = [p.primary if k == recommended else p.bar for k in t["key"]]
    fmt = "%{x:.1f}%" if use_pct else "%{x:,.1f}"
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:.1f}%" if use_pct else f"{v:,.1f}" for v in values],
                           textposition="outside", textfont=dict(color=p.ink2),
                           hovertemplate=f"%{{y}}: {fmt}<extra></extra>", cliponaxis=False))
    bench = t.loc[t["key"] == benchmark_key, col]
    if len(bench):
        bv = float(bench.iloc[0]) * (100 if use_pct else 1)
        fig.add_vline(x=bv, line=dict(color=p.muted, width=1.5, dash="dot"),
                      annotation_text="benchmark", annotation_position="top",
                      annotation_font=dict(color=p.muted, size=11))
    fig = layout(fig, 60 + 40 * len(t), f"test {col}" + (" (%)" if use_pct else ""), legend=False)
    fig.update_yaxes(title=None, tickfont=dict(color=p.ink2, size=13), gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor=p.grid, title=dict(text=f"Test {col}" + (" (%)" if use_pct else ""),
                                                                 font=dict(color=p.muted, size=12)))
    fig.update_layout(hovermode="closest", bargap=0.35)
    return fig


def forecast_chart(history: pd.Series, table: pd.DataFrame, level: int, name: str) -> go.Figure:
    p = palette()
    fig = go.Figure()
    idx = x(table.index)
    fig.add_trace(go.Scatter(x=idx, y=table["upper"].values, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=idx, y=table["lower"].values, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=p.primary_band, name=f"{level}% range",
                             customdata=np.c_[table["lower"].values, table["upper"].values],
                             hovertemplate=f"{level}% range: %{{customdata[0]:,.1f}} – %{{customdata[1]:,.1f}}"
                                           "<extra></extra>"))
    _actual(fig, history, name="History")
    joined = pd.concat([history.iloc[-1:], table["forecast"]])
    _forecast(fig, joined, name=f"Forecast ({name})")
    return layout(fig, 360)


def tracking_chart(tr: tracking.Tracking) -> go.Figure:
    p = palette()
    s = tr.signal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x(s.index), y=s.values, mode="lines+markers", name="Tracking signal",
                             line=dict(color=p.ink, width=2), marker=dict(size=8, color=p.ink),
                             hovertemplate="%{y:+.2f}<extra></extra>"))
    for lim in (tracking.LIMIT, -tracking.LIMIT):
        fig.add_hline(y=lim, line=dict(color=p.alarm, width=1.5, dash="dot"),
                      annotation_text=f"{lim:+g} alarm", annotation_position="top left" if lim > 0 else "bottom left",
                      annotation_font=dict(color=p.alarm, size=11))
    if not tr.ok:
        fig.add_trace(go.Scatter(x=x(pd.PeriodIndex([tr.breach])), y=[tr.breach_value], mode="markers",
                                 name="Alarm", marker=dict(symbol="x-thin", size=14,
                                                           line=dict(width=3, color=p.alarm)),
                                 hovertemplate="Alarm: %{y:+.2f}<extra></extra>"))
    top = max(5.0, float(np.nanmax(np.abs(s.values))) + 0.5) if len(s) else 5.0
    fig.update_yaxes(range=[-top, top])
    return layout(fig, 240, "RSFE ÷ MAD", legend=False)
