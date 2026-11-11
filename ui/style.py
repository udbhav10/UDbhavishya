"""Visual encoding shared by every chart, for light and dark mode.

Colours follow a validated palette (categorical slots 1-2 pass colour-blind and
contrast checks on both surfaces). Meaning never depends on colour alone:
actual = solid line, forecast = dashed line, outlier = cross marker, alarm =
dotted line, test period = shaded background with a text label.
"""
from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Palette:
    surface: str
    ink: str  # primary text / actual line
    ink2: str  # secondary text
    muted: str  # axis labels, benchmark
    grid: str
    axis: str
    primary: str  # forecast, recommended
    primary_band: str  # range band fill
    second: str  # corrected values
    test_wash: str  # test-period shading
    outlier: str  # status: serious
    alarm: str  # status: critical
    good: str  # status: good
    bar: str  # non-highlighted bars


LIGHT = Palette(
    surface="rgba(0,0,0,0)", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
    primary="#2a78d6", primary_band="rgba(42,120,214,0.16)", second="#eb6834", test_wash="rgba(137,135,129,0.12)",
    outlier="#ec835a", alarm="#d03b3b", good="#0ca30c", bar="#c3c2b7",
)
DARK = Palette(
    surface="rgba(0,0,0,0)", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
    primary="#3987e5", primary_band="rgba(57,135,229,0.22)", second="#d95926", test_wash="rgba(195,194,183,0.10)",
    outlier="#ec835a", alarm="#d03b3b", good="#0ca30c", bar="#5a5955",
)


def mode() -> str:
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - older runtimes or bare mode
        kind = None
    return "dark" if kind == "dark" else "light"


def palette() -> Palette:
    return DARK if mode() == "dark" else LIGHT


def layout(fig: go.Figure, height: int = 340, y_title: str | None = None, legend: bool = True) -> go.Figure:
    p = palette()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor=p.surface,
        plot_bgcolor=p.surface,
        font=dict(family=FONT, color=p.ink2, size=13),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                    font=dict(color=p.ink2)),
    )
    fig.update_xaxes(showgrid=False, linecolor=p.axis, tickcolor=p.axis, tickfont=dict(color=p.muted),
                     zeroline=False)
    fig.update_yaxes(gridcolor=p.grid, gridwidth=1, zeroline=False, tickfont=dict(color=p.muted),
                     title=dict(text=y_title or "", font=dict(color=p.muted, size=12)))
    return fig


def show(fig: go.Figure, key: str | None = None, toolbar: bool = True) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key,
                    config={"displaylogo": False, "displayModeBar": "hover" if toolbar else False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
