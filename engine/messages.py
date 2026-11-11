"""Message and change-note types shared by every engine module.

The engine never prints or renders anything; it returns these objects and the UI
decides how to show them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Level = Literal["error", "warning", "info"]
LEVELS: tuple[Level, ...] = ("error", "warning", "info")


@dataclass
class Message:
    """One validation result.

    level   error blocks analysis; warning lets it continue with something disabled;
            info is a plain fact.
    code    stable identifier, e.g. "duplicates", used by the UI to attach actions.
    summary one line saying what happened.
    rows    optional table showing where it happened.
    actions optional labels of actions the user can take.
    """

    level: Level
    code: str
    summary: str
    rows: pd.DataFrame | None = None
    actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"Unknown message level: {self.level}")

    @property
    def blocking(self) -> bool:
        return self.level == "error"


@dataclass
class ChangeNote:
    """A record of a change made to the user's data (automatic or chosen)."""

    kind: str  # e.g. "duplicates", "gaps", "rollup", "outliers"
    summary: str  # short phrase for the change bar, e.g. "6 duplicates added up"
    details: pd.DataFrame | None = None


def has_errors(messages: list[Message]) -> bool:
    return any(m.blocking for m in messages)
