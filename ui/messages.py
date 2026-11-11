"""Rendering engine messages and change notes: one line each, details on demand."""
from __future__ import annotations

import streamlit as st

from engine.messages import ChangeNote, Message

ICONS = {"error": "✖", "warning": "⚠", "info": "ℹ"}


def render(msg: Message, key: str, show_rows: bool = True) -> None:
    text = f"{msg.summary}"
    box = {"error": st.error, "warning": st.warning, "info": st.info}[msg.level]
    box(text, icon=ICONS[msg.level])
    if show_rows and msg.rows is not None and len(msg.rows):
        with st.expander("Details", expanded=False):
            st.dataframe(_display(msg.rows), hide_index=True, width="stretch", key=f"{key}_rows")


def render_all(messages: list[Message], key: str, skip: tuple[str, ...] = ()) -> None:
    order = {"error": 0, "warning": 1, "info": 2}
    for i, m in enumerate(sorted(messages, key=lambda m: order[m.level])):
        if m.code not in skip:
            render(m, f"{key}_{i}")


def note(text: str) -> None:
    """A grey one-line fact (info level without a box)."""
    st.caption(f"ℹ {text}")


def change_bar(notes: list[ChangeNote], key: str) -> None:
    """Compact persistent summary of every change made to the data."""
    if not notes:
        return
    summary = " · ".join(n.summary for n in notes)
    with st.popover(f"⚙ {summary}", width="content"):
        for i, n in enumerate(notes):
            st.markdown(f"**{n.summary[:1].upper() + n.summary[1:]}**")
            if n.details is not None and len(n.details):
                st.dataframe(_display(n.details), hide_index=True, width="stretch", key=f"{key}_{i}")


def _display(frame):
    out = frame.copy()
    for col in out.columns:
        if str(out[col].dtype).startswith("period"):
            out[col] = out[col].astype(str)
        elif str(out[col].dtype).startswith("datetime"):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out
