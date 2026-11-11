"""Forecast Studio: a guided demand-forecasting app (Streamlit entry point)."""
from __future__ import annotations

import streamlit as st

from engine import frequency as fq
from engine import seasonality, split
from engine import validate as v
from engine.messages import has_errors
from ui import messages, state, steps
from ui.widgets import bind_state

st.set_page_config(page_title="Forecast Studio", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
state.init()
ss = st.session_state


# ---------------------------------------------------------------- data + run

ctx = state.data_context()
result = None
run_error = None
if ctx.series is not None:
    try:
        result = state.run(ctx.series)
    except Exception as exc:  # noqa: BLE001 - shown to the user, never a stack trace
        run_error = f"The forecast could not be computed ({exc})."


# ---------------------------------------------------------------- sidebar

def sku_status(name: str) -> str:
    raw = ctx.parsed.skus[name]
    if raw.freq is None:
        return "dates unclear"
    d = v.diagnose(raw, ctx.freq or raw.freq)
    issues = [m for m in d.messages if m.level in ("error", "warning")]
    n = int(raw.rows["demand"].notna().sum())
    s = raw.freq.default_season
    if issues:
        return f"{len(issues)} issue{'s' if len(issues) > 1 else ''}"
    if s and n < 2 * s + split.default_test_length(n, s, raw.freq.code):
        return "OK · too short for Winters"
    return "OK"


with st.sidebar:
    st.markdown("### 📈 Forecast Studio")
    st.caption(ctx.source_label)
    if ctx.parsed is not None and ctx.parsed.has_sku_column and ctx.parsed.skus:
        names = list(ctx.parsed.skus)

        def _sku_changed():
            ss.sku = ss.w_sku_pick
            state.reset_downstream("series")

        if ss.get("w_sku_pick") not in names:
            ss.w_sku_pick = ss.sku if ss.sku in names else names[0]
        st.selectbox("SKU", names, key="w_sku_pick", on_change=_sku_changed,
                     format_func=lambda n: f"{n} · {len(ctx.parsed.skus[n].rows)} rows · {sku_status(n)}",
                     help="Pick the product to forecast. Each SKU is checked separately.")
    if ctx.series is not None:
        f = ctx.series.freq
        st.markdown(f"**Frequency** · {f.label}")
        if result is not None:
            st.markdown(f"**Seasonal period** · {result.s if result.s else 'none'}")
            sp = result.split
            st.markdown(f"**Training** · {sp.train.index[0]} – {sp.train.index[-1]}  \n"
                        f"**Test** · {sp.test.index[0]} – {sp.test.index[-1]}")
    st.divider()
    st.caption("ℹ Your data stays in this session only. Nothing is saved or shared.")
    st.caption("Theme: ⋮ menu → Settings → light or dark.")


# ---------------------------------------------------------------- stepper

blocked = ctx.series is None or result is None


def _go(target: str):
    ss.step = target
    ss.pop("w_step", None)


labels = {name: f"{i + 1} {name}" for i, name in enumerate(state.STEPS)}
locked_attempt = blocked and (ss.step or state.STEPS[0]) != state.STEPS[0]
if locked_attempt:
    ss.step = state.STEPS[0]
    ss.pop("w_step", None)
bind_state(st.segmented_control, "Step", "step", options=state.STEPS, label_visibility="collapsed",
           format_func=lambda s: labels[s] if not (blocked and s != state.STEPS[0]) else f"🔒 {labels[s]}")
current = ss.step or state.STEPS[0]
if locked_attempt:
    st.warning(f"Finish the Data step first: {ctx.block_reason}", icon="⚠")

if result is not None:
    changed = state.changed_since_seen(current, state.stage_fingerprints(ctx, result))
    if changed:
        st.caption(f"↻ Results updated: {', '.join(changed)} changed")
    messages.change_bar(state.all_notes(ctx.series, result), f"notes_{current}")


# ---------------------------------------------------------------- step body

if current == "Data":
    steps.data_step(ctx)
    if run_error:
        st.error(run_error, icon="✖")
    if result is not None:
        steps.seasonality_panel(result)
elif current == "Split & clean":
    steps.split_step(ctx, result)
elif current == "Models":
    steps.models_step(ctx, result)
elif current == "Compare":
    steps.compare_step(ctx, result)
else:
    steps.forecast_step(ctx, result)


# ---------------------------------------------------------------- navigation

st.divider()
i = state.STEPS.index(current)
left, _, right = st.columns([1, 4, 1])
with left:
    if i > 0:
        st.button("‹ Back", on_click=_go, args=(state.STEPS[i - 1],), width="stretch")
with right:
    if i < len(state.STEPS) - 1:
        st.button(f"Next: {state.STEPS[i + 1]} ›", type="primary", on_click=_go, args=(state.STEPS[i + 1],),
                  disabled=blocked, width="stretch",
                  help=ctx.block_reason if blocked else None)
