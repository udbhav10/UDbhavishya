"""UDbhavishya: a guided demand-forecasting app (Streamlit entry point).

Udbhav (origin) + bhavishya (the future), by Udbhav Dalal.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
APP_NAME = "UDbhavishya"
TAGLINE = "See what demand comes next · by Udbhav Dalal"

st.set_page_config(page_title=APP_NAME, page_icon=str(ASSETS / "mark_64.png"), layout="wide",
                   initial_sidebar_state="expanded")
_theme = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
_veil, _card, _track, _ink = (
    ("rgba(0,0,0,0.55)", "#1f2229", "rgba(255,255,255,0.14)", "#f0efec") if _theme == "dark"
    else ("rgba(15,23,42,0.28)", "#ffffff", "rgba(42,120,214,0.16)", "#0b0b0b"))

# Styles go in before the heavier imports, so the loading overlay already works on the first load.
# The overlay shows while Streamlit is running the script (its status widget is present). It
# blocks clicks at once and fades in after 0.35 s, so quick slider updates don't flash.
st.html(f"""<style>
[data-testid="stSidebarHeader"] {{ margin-bottom: 0; height: auto; padding-top: 0.25rem; }}
[data-testid="stSidebarHeader"] img {{ height: 3rem; max-width: 100%; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0; }}
[data-testid="stSidebarUserContent"] [data-testid="stCaptionContainer"]:first-of-type {{ margin-top: 0.6rem; }}

.st-key-topbar {{ position: fixed; top: 0.6rem; right: 3.75rem; z-index: 999991; width: auto; }}
.st-key-topbar button {{ padding: 0.15rem 0.75rem; min-height: 2rem; }}

@keyframes ud-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes ud-spin {{ to {{ transform: translate(-50%, -50%) rotate(360deg); }} }}
.stApp:has([data-testid="stStatusWidget"])::before {{
  content: "Working on it…"; position: fixed; inset: 0; z-index: 1000000;
  display: flex; align-items: center; justify-content: center; padding-top: 6.4rem;
  background: radial-gradient(circle at 50% calc(50% + 1.1rem), {_card} 0 6.2rem, transparent 6.25rem), {_veil};
  backdrop-filter: blur(2px); color: {_ink};
  font: 600 1rem system-ui, -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.01em;
  animation: ud-fade 0.25s ease 0.35s both; cursor: progress;
}}
.stApp:has([data-testid="stStatusWidget"])::after {{
  content: ""; position: fixed; left: 50%; top: 50%; z-index: 1000001;
  width: 3.25rem; height: 3.25rem; border-radius: 50%;
  border: 0.32rem solid {_track}; border-top-color: #2a78d6; border-right-color: #2a78d6;
  transform: translate(-50%, -50%);
  animation: ud-fade 0.25s ease 0.35s both, ud-spin 0.8s linear infinite;
}}
</style>""")

from engine import frequency as fq  # noqa: E402
from engine import seasonality, split  # noqa: E402
from engine import validate as v  # noqa: E402
from engine.messages import has_errors  # noqa: E402,F401
from ui import messages, state, steps  # noqa: E402
from ui.widgets import bind_state  # noqa: E402

state.init()
ss = st.session_state
st.logo(str(ASSETS / f"wordmark_{_theme}.svg"), size="large", icon_image=str(ASSETS / "mark.svg"))


# ---------------------------------------------------------------- reset

def _ask_reset() -> None:
    ss.confirm_reset = True


def _cancel_reset() -> None:
    ss.confirm_reset = False


def _do_reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


@st.dialog("Start over?", dismissible=False)
def confirm_reset() -> None:
    st.markdown("This clears your uploaded file, every choice you made and all results, "
                "and takes you back to the first screen.")
    st.warning("Your progress will be lost. This cannot be undone.", icon="⚠")
    left, right = st.columns(2)
    left.button("Cancel", width="stretch", on_click=_cancel_reset, key="reset_cancel")
    right.button("Reset everything", type="primary", width="stretch", on_click=_do_reset, key="reset_confirm")


with st.container(key="topbar"):
    st.button("Reset", icon=":material/restart_alt:", key="reset_all", on_click=_ask_reset,
              help="Start over: clear the data, choices and results.")
if ss.get("confirm_reset"):
    confirm_reset()


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
    st.caption(TAGLINE)
    st.markdown(f"**{ctx.source_label}**")
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
            st.markdown(f"**Training** · {fq.label(sp.train.index[0])} – {fq.label(sp.train.index[-1])}  \n"
                        f"**Test** · {fq.label(sp.test.index[0])} – {fq.label(sp.test.index[-1])}")
    st.divider()
    st.caption("ℹ Your data stays in this session only. Nothing is saved or shared.")
    st.caption("Theme: ⋮ menu (top right) → Light or Dark.")


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
