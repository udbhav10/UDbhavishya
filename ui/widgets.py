"""Widgets bound to persistent session values.

Streamlit forgets a widget's value when the widget is not drawn (e.g. on another
step). Each bound widget therefore keeps its own key ("w_...") seeded from a
persistent value, and writes back to that value when the user changes it.
"""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st


def bind(fn: Callable, label: str, wkey: str, get: Callable[[], Any], set_: Callable[[Any], None], **kw):
    if wkey not in st.session_state:
        st.session_state[wkey] = get()

    def changed():
        set_(st.session_state[wkey])

    return fn(label, key=wkey, on_change=changed, **kw)


def bind_state(fn: Callable, label: str, name: str, default: Any = None, **kw):
    """Bind to st.session_state[name]; `default` is used when the stored value is None."""
    def get():
        val = st.session_state.get(name)
        return default if val is None else val

    def set_(val):
        st.session_state[name] = val

    return bind(fn, label, f"w_{name}", get, set_, **kw)


def forget(prefixes: tuple[str, ...]) -> None:
    """Drop widget keys so they re-seed from persistent values on the next draw."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(prefixes):
            del st.session_state[k]
