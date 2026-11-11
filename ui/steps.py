"""The five step screens. Each draws controls, calls the engine via state, and charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from engine import demo, frequency as fq, io as eio, metrics, optimise, recommend
from engine import validate as v
from engine.models import METHOD_NAMES
from engine.outliers import REPLACEMENTS
from engine.pipeline import DEFAULT_PARAMS
from engine.ranges import LEVELS

from . import charts, messages, state
from .style import show
from .widgets import bind, bind_state, forget

ss = st.session_state
NONE = "__none__"  # option token: Streamlit shows a None value as "nothing selected"


def subtitle(text: str) -> None:
    st.caption(text)


def how(text: str) -> None:
    with st.expander("How is this calculated?"):
        st.markdown(text)


def fmt(value: float | None, pct: bool = False, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%" if pct else f"{value:,.{digits}f}"


# ================================================================ 1. Data

def _on_source_change():
    ss.source = ss.src_source
    state.reset_downstream("data")
    forget(("w_",))


def _on_demo_change():
    ss.demo_key = ss.src_demo
    forget(("w_",))


def _on_upload():
    f = ss.get("src_upload")
    if f is None:
        ss.upload_name, ss.upload_bytes = None, None
    else:
        ss.upload_name, ss.upload_bytes = f.name, f.getvalue()
    forget(("w_",))


def data_step(ctx: state.DataContext) -> None:
    subtitle("Load your sales history, or try a demo product.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        if "src_source" not in ss:
            ss.src_source = ss.source
        st.segmented_control("Data source", ["demo", "upload"], key="src_source", on_change=_on_source_change,
                             format_func={"demo": "Demo product", "upload": "Upload a file"}.get,
                             help="Use a realistic sample product, or upload your own sales file.")
        if ss.source == "demo":
            if "src_demo" not in ss:
                ss.src_demo = ss.demo_key
            st.selectbox("Demo product", list(demo.PRODUCTS), key="src_demo", on_change=_on_demo_change,
                         format_func=lambda k: f"{demo.PRODUCTS[k].name} · {demo.PRODUCTS[k].frequency}",
                         help="Each demo shows a different demand pattern.")
            p = demo.PRODUCTS[ss.demo_key]
            messages.note(f"Simulated data for demonstration: {p.pattern}; driver: {p.driver}.")
        else:
            st.file_uploader("Sales file (.csv or .xlsx, up to 50 MB)", type=["csv", "xlsx"], key="src_upload",
                             on_change=_on_upload, help="One row per period: a date and a demand value.")
            with st.popover("File format"):
                st.markdown(
                    "- **date**: any common format (2024-01, Jan-2024, 01/01/2024)\n"
                    "- **demand**: numbers\n"
                    "- **driver** *(optional)*: one numeric cause, e.g. price or promotion (1/0)\n"
                    "- **SKU** *(optional)*: product code, if the file has several products\n"
                    "- Future rows with a driver but no demand are used for regression forecasts.")
                st.download_button("Download template", eio.template_csv(), "forecast_template.csv", "text/csv")
        messages.note("Your data stays in this browser session only; nothing is stored.")
    with right:
        if ctx.frame is not None and not ctx.is_demo:
            _mapping_controls(ctx)
        if ctx.raw is not None:
            _frequency_controls(ctx)

    if ctx.frame is None and not ctx.messages:
        return
    messages.render_all(ctx.messages, "data", skip=("duplicates", "gaps", "single_sku", "frequency"))
    if ctx.diagnosis is not None:
        _duplicates_panel(ctx)
        _gaps_panel(ctx)
    if ctx.series is None:
        return
    messages.change_bar(ctx.series.notes, "data_notes")
    st.markdown(f"**{ctx.series.sku if ctx.parsed.has_sku_column else 'Demand'}** · "
                f"{len(ctx.series.demand)} {ctx.series.freq.unit}s · "
                f"{ctx.series.demand.index[0]} to {ctx.series.demand.index[-1]}")
    show(charts.series_chart(ctx.series.demand, ctx.unit), "data_series")


def _mapping_controls(ctx: state.DataContext) -> None:
    cols = list(ctx.frame.columns)
    st.markdown("**Columns**")
    c1, c2, c3 = st.columns(3)

    def bind_map(col, label, field, optional=False, help=None):
        options = ([NONE] if optional else []) + cols

        def get():
            current = ss.mapping.get(field)
            return current if current in options else options[0]

        def set_(val):
            ss.mapping = {**ss.mapping, field: None if val == NONE else val}
            state.reset_downstream("series")

        with col:
            bind(st.selectbox, label, f"w_map_{field}", get, set_, options=options,
                 format_func=lambda c: "— none —" if c == NONE else str(c), help=help)

    bind_map(c1, "Date", "date", help="The column holding each period's date.")
    bind_map(c2, "Demand", "demand", help="The column holding sales or demand numbers.")
    bind_map(c3, "Driver", "driver", optional=True, help="Optional numeric cause of demand, for regression.")
    if ctx.mapping.get("sku"):
        messages.note(f"SKU column found: “{ctx.mapping['sku']}”. Pick a SKU in the sidebar.")


def _frequency_controls(ctx: state.DataContext) -> None:
    detected = ctx.raw.freq
    c1, c2 = st.columns(2)
    with c1:
        bind(st.selectbox, "Frequency", "w_freq_override", lambda: ss.freq_override or NONE,
             _setter("freq_override", reset=True), options=[NONE] + list(fq.ORDER),
             format_func=lambda c: (f"Detected: {detected.label}" if detected else "Not detected")
             if c == NONE else fq.LABELS[c],
             help="How often demand is recorded. Change it only if detection is wrong.")
    base = ctx.freq
    if base is not None and fq.coarser_options(base):
        with c2:
            options = [NONE] + fq.coarser_options(base)
            bind(st.selectbox, "Roll up to", "w_rollup_to",
                 lambda: ss.rollup_to if ss.rollup_to in options else NONE, _setter("rollup_to", reset=True),
                 options=options, format_func=lambda c: "Keep as is" if c == NONE else fq.LABELS[c],
                 help="Add periods together into weeks, months, quarters or years.")
        if ss.rollup_to and ctx.mapping.get("driver"):
            bind_state(st.radio, "Combine the driver by", "driver_how", options=["mean", "sum"], horizontal=True,
                       format_func={"mean": "Average", "sum": "Sum"}.get,
                       help="Average suits prices or temperature; sum suits counts like promotion days.")


def _setter(name: str, reset: bool = False):
    def set_(val):
        ss[name] = None if val == NONE else val
        if reset:
            ss.dup_choice, ss.gap_choice = {}, {}
            state.reset_downstream("series")
    return set_


def _duplicates_panel(ctx: state.DataContext) -> None:
    d = ctx.diagnosis
    if not d.n_duplicate_periods:
        return
    msg = next(m for m in d.messages if m.code == "duplicates")
    chosen = ss.dup_choice.get(ss.sku)
    if chosen:
        return
    messages.render(msg, "dups", show_rows=False)
    rows = d.duplicates.copy()
    rows["period"] = rows["period"].astype(str)
    rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")

    def highlight(row):
        colour = "rgba(208,59,59,0.18)" if row["type"] == "conflicting" else "rgba(250,178,25,0.18)"
        return [f"background-color: {colour}"] * len(row)

    styled = rows.style.apply(highlight, axis=1).format({"demand": "{:,.2f}", "driver": "{:,.2f}"}, na_rep="")
    st.dataframe(styled, hide_index=True, width="stretch")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        choice = st.radio("What should we do?", list(v.DUPLICATE_CHOICES), horizontal=True, key="dup_radio",
                          format_func=v.DUPLICATE_CHOICES.get, index=None,
                          help="Add up suits separate orders in one period; keep or average suits copies.")
    with c2:
        if st.button("Apply", type="primary", disabled=choice is None, key="dup_apply"):
            ss.dup_choice = {**ss.dup_choice, ss.sku: choice}
            st.rerun()
    with c3:
        st.download_button("Download flagged rows", rows.to_csv(index=False).encode(), "duplicates.csv",
                           "text/csv", key="dup_dl")


def _gaps_panel(ctx: state.DataContext) -> None:
    d = ctx.diagnosis
    if not d.gaps or ss.gap_choice.get(ss.sku):
        return
    if d.n_duplicate_periods and not ss.dup_choice.get(ss.sku):
        return
    msg = next(m for m in d.messages if m.code == "gaps")
    messages.render(msg, "gaps")
    c1, c2 = st.columns([3, 1])
    with c1:
        choice = st.radio("Fill the missing periods with", list(v.GAP_CHOICES), horizontal=True, index=None,
                          key="gap_radio", format_func=v.GAP_CHOICES.get,
                          help="Zero if the business was closed; carry forward if data is just missing.")
    with c2:
        if st.button("Apply", type="primary", disabled=choice is None, key="gap_apply"):
            ss.gap_choice = {**ss.gap_choice, ss.sku: choice}
            st.rerun()
    messages.note("Or fix the file and upload it again.")


def seasonality_panel(result) -> None:
    check = result.season_check
    s = result.s
    st.markdown("**Seasonal pattern**")
    if check.default is None:
        messages.note("Yearly data: no seasonal pattern is modelled.")
    elif check.agrees and (s == check.default):
        st.caption(f"✔ s = {s}: the autocorrelation confirms a {s}-period cycle.")
    elif check.suggestion and s == check.default:
        st.caption(f"ℹ The autocorrelation peaks at lag {check.suggestion}; consider s = {check.suggestion}.")
    else:
        st.caption(f"s = {s}.")
    show(charts.acf_chart(check.acf, check.threshold, s, check.peak_lag), "acf")
    with st.expander("Advanced: seasonal period"):
        default = check.default
        bind_state(st.number_input, "Seasonal period s (periods per cycle)", "season",
                   default=default or 1, min_value=1, max_value=max(2, len(result.split.train) // 2), step=1,
                   help="Periods in one full cycle, e.g. 12 months or 7 days.")
        if ss.season is not None and st.button("Reset to default", key="season_reset"):
            ss.season = None
            forget(("w_season",))
            st.rerun()
    how("Each bar is the correlation between the series and itself shifted back *k* periods "
        "(Excel `=CORREL`), after removing a straight-line trend. A peak at lag *k* means the pattern repeats "
        "every *k* periods (slide 24). Dotted lines: roughly ±2/√n, the level that random noise rarely exceeds.")


# ================================================================ 2. Split & clean

def split_step(ctx, result) -> None:
    subtitle("Hide the latest periods so we can test how well each method forecasts them.")
    series = ctx.series
    n = len(series.demand)
    unit = series.freq.unit
    c1, c2 = st.columns(2)
    with c1:
        bind_state(st.slider, f"Test period ({unit}s)", "test_length", default=state.default_test_length(series),
                   min_value=1, max_value=max(1, n - 4),
                   help="The latest periods held back to test forecasts honestly.")
        sp = result.split
        st.caption(f"Training {sp.train.index[0]} – {sp.train.index[-1]} ({len(sp.train)}) · "
                   f"Test {sp.test.index[0]} – {sp.test.index[-1]} ({len(sp.test)})")
    with c2:
        bind_state(st.slider, "Outlier limit (± standard deviations)", "outlier_z", min_value=1.5,
                   max_value=5.0, step=0.5, help="Points further than this from normal are flagged as outliers.")
        out = result.outliers
        st.caption(f"{out.count} outlier(s) flagged in training · measured in "
                   f"{'percent (seasonal)' if out.mode == 'percent' else 'units'}")
    show(charts.split_chart(result.split.train, result.split.test, result.outliers), "split_chart")
    with st.expander("Advanced: what to do with outliers"):
        bind_state(st.radio, "Replacement", "outlier_replacement", options=list(REPLACEMENTS),
                   format_func=REPLACEMENTS.get, horizontal=True,
                   help="Corrections apply to training data only; test data is never changed.")
        if result.outliers.mode == "units":
            st.caption(f"Reference: training average · σ = 1.25 × MAD = {result.outliers.sigma:,.1f}")
        else:
            st.caption(f"Reference: same season in other years · σ = 1.25 × MAPE = "
                       f"{result.outliers.sigma * 100:.1f}%")
    if len(result.outliers.corrections):
        st.markdown("**Corrections**")
        t = result.outliers.corrections.copy()
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"original": st.column_config.NumberColumn(format="%.1f"),
                                    "new": st.column_config.NumberColumn(format="%.1f")})
    elif result.outliers.count:
        messages.note("Outliers are flagged but left unchanged. Choose a replacement under Advanced to correct them.")
    if len(result.split.train) < 2 * (result.s or 0):
        st.warning("Fewer than two full seasons of training data: Winters and seasonal outlier checks are off.",
                   icon="⚠")
    how("**Split** — training data fits each method; the test period is forecast from the end of training and "
        "compared with what really happened (slide 46). Rows are never shuffled.\n\n"
        "**Outliers** (slides 52–53) — gap = actual − reference. Non-seasonal: reference = training average, "
        "σ = 1.25 × MAD. Seasonal: reference = typical value of the same season in the other training years "
        "(trend-adjusted median), gaps in percent, σ = 1.25 × MAPE. Flag when |gap ÷ σ| exceeds the limit.")


# ================================================================ 3. Models

SMOOTH_PARAMS = {"ses": ["alpha"], "holt": ["alpha", "beta"], "winters": ["alpha", "beta", "gamma"]}
PARAM_LABEL = {"alpha": "α level", "beta": "β trend", "gamma": "γ season", "phi": "φ damping"}
PARAM_HELP = {
    "alpha": "How strongly the newest actual moves the level.",
    "beta": "How quickly the trend follows recent changes in level.",
    "gamma": "How quickly seasonal indices follow the latest season.",
    "phi": "Below 1, the trend fades out over the forecast horizon.",
}


def _param_get(method, name):
    return lambda: float(ss.params.get(method, {}).get(name, DEFAULT_PARAMS[method].get(name, 0.98)))


def _param_set(method, name):
    def set_(val):
        ss.params = {**ss.params, method: {**ss.params.get(method, {}), name: float(val)}}
        ss.tuned = {**ss.tuned, method: False}
    return set_


def _optimise(method, train, start, s, damped):
    params = optimise.optimise(method, train, start, s=s, damped=damped)
    ss.params = {**ss.params, method: {**ss.params.get(method, {}), **params}}
    ss.tuned = {**ss.tuned, method: True}
    forget((f"w_p_{method}_",))


def models_step(ctx, result) -> None:
    subtitle("Fit one method at a time on the training data; the dotted line is the benchmark to beat.")
    av = result.availability
    options = [m for m in ("ses", "holt", "winters", "regression") if av[m].available and m in result.fits]
    unavailable = [m for m in ("ses", "holt", "winters", "regression") if m not in options]
    if not options:
        st.warning("No smoothing or regression method can run on this data.", icon="⚠")
        return
    if ss.get("model_view") not in options:
        ss.model_view = options[0]
    bind_state(st.segmented_control, "Method", "model_view", options=options, format_func=METHOD_NAMES.get,
               help="Pick a method to adjust. Benchmarks need no settings.")
    m = ss.model_view or options[0]
    fit = result.fits[m]

    if m in SMOOTH_PARAMS:
        _smoothing_controls(m, result)
    else:
        _regression_controls(result)

    bench = None
    if "snaive" in result.fits:
        bench = pd.concat([result.fits["snaive"].fitted.dropna(), result.test_forecasts["snaive"]])
    show(charts.fit_chart(result.train, result.split.test, fit.fitted, result.test_forecasts[m], bench, fit.name),
         f"fit_{m}")
    resid = (result.train - fit.fitted).dropna()
    if len(resid):
        st.caption(f"Training fit: MAD {resid.abs().mean():,.1f} · bias {resid.mean():+,.1f} "
                   f"({metrics.bias_direction(resid.mean(), resid.abs().mean())[1]})")
    if m in SMOOTH_PARAMS and fit.components:
        show(charts.components_chart(fit.components), f"comp_{m}", toolbar=False)
    if unavailable:
        st.caption(" · ".join(f"{METHOD_NAMES[k]}: {av[k].reason}" for k in unavailable if av[k].reason))
    _models_how(m)


def _smoothing_controls(m, result) -> None:
    names = SMOOTH_PARAMS[m]
    cols = st.columns(len(names) + 1)
    for col, name in zip(cols, names):
        with col:
            bind(st.slider, PARAM_LABEL[name], f"w_p_{m}_{name}", _param_get(m, name), _param_set(m, name),
                 min_value=0.01, max_value=0.99, step=0.01, help=PARAM_HELP[name])
    start = result.fits[m].start
    with cols[-1]:
        st.write("")
        st.button("Optimise", key=f"opt_{m}", type="primary", on_click=_optimise,
                  args=(m, result.train, start, result.s, bool(ss.damped.get(m))),
                  help="Find the settings with the smallest training errors.")
    if ss.tuned.get(m):
        st.caption("✔ Tuned on training data only")
    with st.expander("Advanced: damping and start values"):
        if m in ("holt", "winters"):
            def set_damped(val, m=m):
                ss.damped = {**ss.damped, m: bool(val)}
                ss.tuned = {**ss.tuned, m: False}
            bind(st.toggle, "Damp the trend", f"w_damp_{m}", lambda: bool(ss.damped.get(m)), set_damped,
                 help="Lets the trend fade instead of growing forever.")
            if ss.damped.get(m):
                bind(st.slider, PARAM_LABEL["phi"], f"w_p_{m}_phi", _param_get(m, "phi"), _param_set(m, "phi"),
                     min_value=0.80, max_value=1.00, step=0.01, help=PARAM_HELP["phi"])
        _start_value_editor(m, result)


def _start_value_editor(m, result) -> None:
    computed = result.start_values.get(m)
    if computed is None:
        return
    over = dict(ss.overrides.get(m, {}))

    def set_override(field, value):
        current = dict(ss.overrides.get(m, {}))
        current[field] = value
        ss.overrides = {**ss.overrides, m: current}

    st.markdown("**Start values** · from training data only")
    fields = ["level"] + (["trend"] if m in ("holt", "winters") else [])
    cols = st.columns(len(fields))
    for col, field in zip(cols, fields):
        base = getattr(computed, field)
        with col:
            label = {"level": "Starting level L₀", "trend": "Starting trend T₀"}[field]
            if field in over:
                label += " · overridden"
            bind(st.number_input, label, f"w_sv_{m}_{field}_{base:.6g}",
                 lambda f=field, b=base: float(over.get(f, b)), lambda val, f=field: set_override(f, float(val)),
                 format="%.3f", help="Value before the first training period; edit to override.")
            st.caption(computed.notes.get(field, ""))
    if m == "winters" and computed.season is not None:
        idx = result.train.index
        labels = [str(idx[i]) if i < len(idx) else str(i) for i in range(len(computed.season))]
        values = np.asarray(over.get("season", computed.season), dtype=float)
        frame = pd.DataFrame({"first period": labels, "seasonal index": values})
        edited = st.data_editor(frame, hide_index=True, disabled=["first period"], width="stretch",
                                key=f"w_sv_winters_season_{hash(tuple(np.round(computed.season, 6)))}")
        new = edited["seasonal index"].to_numpy(dtype=float)
        if not np.allclose(new, values):
            set_override("season", new.tolist())
            st.rerun()
        st.caption(("Overridden. " if "season" in over else "") + computed.notes.get("season", ""))
    if over and st.button("Reset to calculated", key=f"reset_{m}"):
        ss.overrides = {k: val for k, val in ss.overrides.items() if k != m}
        forget((f"w_sv_{m}_",))
        st.rerun()


def _regression_controls(result) -> None:
    fit = result.fits["regression"]
    stats = fit.extras["stats"]
    with st.expander("Advanced: add trend or seasonal terms"):
        def setter(field):
            def set_(val):
                ss.params = {**ss.params, "regression": {**ss.params.get("regression", {}), field: bool(val)}}
            return set_
        get = lambda f: (lambda: bool(ss.params.get("regression", {}).get(f, False)))  # noqa: E731
        bind(st.checkbox, "Time trend", "w_reg_trend", get("trend"), setter("trend"),
             help="Adds a steady change per period to the driver's effect.")
        if result.s:
            bind(st.checkbox, "Seasonal dummies", "w_reg_seasonal", get("seasonal"), setter("seasonal"),
                 help="Adds one term per season, relative to the last season.")
    st.caption(f"R² {stats.r2:.3f} · adjusted R² {stats.adj_r2:.3f} · standard error {stats.se:,.1f} · "
               f"{stats.n} observations")
    table = stats.table()
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("coefficient", "std. error", "t-stat")})
    if stats.obs_per_coef < 10:
        st.warning(f"Only {stats.obs_per_coef:.0f} observations per coefficient (about 10 are needed): "
                   "estimates may be unreliable.", icon="⚠")


def _models_how(m) -> None:
    text = {
        "ses": "**SES** (slide 33): L_t = αA_t + (1−α)L_{t−1}; forecast = L_t for every future period.",
        "holt": "**Holt** (slides 35, 38): L_t = αA_t + (1−α)(L_{t−1} + φT_{t−1}); "
                "T_t = β(L_t − L_{t−1}) + (1−β)φT_{t−1}; forecast h ahead = L_t + (φ + … + φ^h)T_t.",
        "winters": "**Winters** (slide 43, multiplicative): as Holt with A_t ÷ S in the level; "
                   "S_t = γ(A_t ÷ L_t) + (1−γ)S_{t−s}; forecast multiplied by the seasonal index.",
        "regression": "**Regression** (slides 60–70): demand = a + b × driver (+ trend, + seasonal dummies), "
                      "least squares on training data. |t| > 2 suggests a real effect.",
    }[m]
    how(text + "\n\n**Start values** (Hyndman et al., 2008): seasonal indices from a centred moving average over "
        "the first two seasons; a straight line through the first ten (deseasonalised) values gives L₀ (intercept) "
        "and T₀ (slope). Only training data is used.\n\n**Optimise** minimises the sum of squared one-step "
        "errors over training data only.")


# ================================================================ 4. Compare

def compare_step(ctx, result) -> None:
    subtitle("Which method forecast the hidden test period best?")
    cmp = result.comparison
    st.markdown(f"#### ★ {cmp.sentence.replace('Recommended: ', 'Recommended: **', 1).replace(' · ', '** · ', 1)}")
    table = cmp.table()
    show(charts.compare_chart(table, cmp.benchmark, cmp.recommended), "compare_chart")
    summary = pd.DataFrame({
        "method": [("★ " if k == cmp.recommended else "") + m for k, m in zip(table["key"], table["method"])],
        "test wMAPE": [fmt(w, pct=True) for w in table["wMAPE"]],
        "bias": table["bias direction"],
        "beats benchmark": ["benchmark" if k == cmp.benchmark else ("✔ yes" if b else "✖ no")
                            for k, b in zip(table["key"], table["beats benchmark"])],
    })
    st.dataframe(summary, hide_index=True, width="stretch")
    if any(sc.mape is None for sc in cmp.scores.values()):
        st.warning("MAPE hidden: the test period has zero-demand periods, so wMAPE is used.", icon="⚠")
    if result.correction is not None:
        c = result.correction
        st.caption(f"Outlier correction: {cmp.names[c.method]} test wMAPE {fmt(c.wmape_with, True)} with it, "
                   f"{fmt(c.wmape_without, True)} without it.")
    with st.expander("All metrics"):
        full = table.drop(columns=["key"]).copy()
        for col in ("wMAPE", "MAPE"):
            full[col] = full[col].map(lambda val: fmt(val, True))
        st.dataframe(full, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ("MAD", "RMSE", "bias", "tracking signal", "MASE",
                                              "value added (pts)")})
    how("Error e = actual − forecast over the test period (slide 47). **wMAPE** = Σ|e| ÷ ΣA. **MAD** = average |e|. "
        "**Bias** = average e (positive: under-forecasting). **Tracking signal** = Σe ÷ MAD. "
        "**MASE** = test MAD ÷ the training MAD of the seasonal-naive forecast (below 1 beats it).\n\n"
        "The **benchmark** is seasonal naive (naive for non-seasonal data). The recommended method has the "
        "lowest wMAPE among those that beat the benchmark; *value added* is the benchmark's wMAPE minus the "
        "method's, in points (slide 78). The average of the two best methods is also tested (slide 79).")


# ================================================================ 5. Forecast

def forecast_step(ctx, result) -> None:
    subtitle("The chosen method, refitted on all data, forecasts beyond the end of your history.")
    cmp = result.comparison
    keys = sorted(cmp.scores, key=lambda k: (k != cmp.recommended, cmp.scores[k].rank_value))
    c1, c2, c3 = st.columns([2, 2, 1.4])
    with c1:
        bind_state(st.selectbox, "Method", "chosen", default=cmp.recommended, options=keys,
                   format_func=lambda k: ("★ " if k == cmp.recommended else "") + cmp.names[k],
                   help="The recommended method is preselected; any tested method can be used.")
    with c2:
        bind_state(st.slider, f"Horizon ({ctx.series.freq.unit}s)", "horizon", default=len(result.final.table),
                   min_value=1, max_value=result.max_horizon, help="How many periods ahead to forecast.")
    with c3:
        bind_state(st.segmented_control, "Range", "range_level", options=list(LEVELS),
                   format_func=lambda val: f"{val}%",
                   help="How likely demand is to fall inside the shaded band.")
    messages.note("Refitted on all data (training + test) before forecasting.")
    final = result.final
    s = result.s or 12
    history = pd.concat([result.train, result.split.test]).iloc[-(len(result.split.test) + 3 * s):]
    show(charts.forecast_chart(history, final.table, final.level, final.name), "forecast_chart")
    st.caption(f"{final.level}% range = forecast × (1 ± z × r) · r = 1.25 × test {final.basis.source} "
               f"({final.basis.error * 100:.1f}%) = {final.basis.r * 100:.1f}%")
    if final.driver_values is not None:
        _driver_editor(final, result)

    st.markdown("**Bias check on the test period**")
    tr = result.tracking
    if tr.ok:
        st.success(tr.verdict, icon="✔")
    else:
        st.error(tr.verdict, icon="✖")
    show(charts.tracking_chart(tr), "tracking_chart")

    table = final.table.copy()
    table.index = table.index.astype(str)
    with st.expander("Forecast table"):
        st.dataframe(table.round(2), width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in table.columns})
    from engine.results import forecast_csv
    st.download_button("Download CSV", forecast_csv(result), "forecast.csv", "text/csv", type="primary")
    how("**Refit** (slide 46): the chosen method is re-estimated on training + test data with your settings, then "
        "forecasts forward.\n\n**Range** (slide 74): forecast × (1 ± z × r), with r ≈ 1.25 × the method's test "
        "MAPE and z = 0.253 / 0.842 / 1.282 for 20% / 60% / 80%. The band widens in peak periods because it "
        "scales with the forecast.\n\n**Tracking signal** (slide 47): running sum of errors ÷ MAD to date over "
        "the test period; beyond ±4 means persistent bias.")


def _driver_editor(final, result) -> None:
    name = result.series.driver_name or "driver"
    st.markdown(f"**Future {name} values** · {final.driver_source}")
    frame = pd.DataFrame({"period": final.driver_values.index.astype(str), name: final.driver_values.values})
    edited = st.data_editor(frame, hide_index=True, disabled=["period"], width="stretch",
                            key=f"w_drv_{len(frame)}_{hash(tuple(frame[name]))}")
    new = edited[name].to_numpy(dtype=float)
    if not np.allclose(new, frame[name].to_numpy(dtype=float), equal_nan=True):
        ss.future_driver = new.tolist()
        st.rerun()
    if ss.future_driver is not None and st.button("Reset driver values", key="drv_reset"):
        ss.future_driver = None
        st.rerun()
