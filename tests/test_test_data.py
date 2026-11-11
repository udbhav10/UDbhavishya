"""Every bundled test-data file loads, runs end to end, and triggers what it was built to test."""
from pathlib import Path

import pandas as pd
import pytest

from engine import frequency as fq
from engine import io as eio
from engine import results
from engine import validate as v

ROOT = Path(__file__).resolve().parents[1] / "test-data"
FILES = sorted([*ROOT.glob("real/*"), *ROOT.glob("synthetic/*")])

EXPECT = {  # file prefix -> (frequency code, message codes that must appear)
    "01": ("D", set()), "02": ("W", set()), "03": ("M", set()), "04": ("M", set()), "05": ("M", set()),
    "06": ("M", set()), "07": ("M", set()), "08": ("M", set()), "09": ("Q", set()), "10": ("Y", set()),
    "11": ("M", set()), "12": ("W", set()), "13": ("D", {"duplicates", "gaps"}), "14": ("M", {"zeros"}),
    "15": ("M", {"negative"}), "16": ("M", {"future_driver"}), "17": ("Y", set()),
}


def test_all_seventeen_files_present():
    assert len(FILES) == 17


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_file_loads_and_runs(path):
    res = eio.read_upload(path.name, path.read_bytes())
    assert res.frame is not None, res.messages
    mapping = eio.suggest_mapping(res.frame)
    parsed = v.parse(res.frame, mapping)
    assert not [m for m in parsed.messages if m.level == "error"]
    code, must = EXPECT[path.name[:2]]
    seen = set()
    for raw in parsed.skus.values():
        assert raw.freq.code == code
        d = v.diagnose(raw)
        seen |= {m.code for m in d.messages}
        series = v.build_series(raw, d, raw.freq, "sum" if d.n_duplicate_periods else None,
                                "zero" if d.gaps else None, driver_name=mapping.get("driver"))
        out = results.run(series)
        assert len(out.final.table) > 0
    assert must <= seen, (must, seen)


def test_week_label():
    week = pd.Period("2024-03-06", freq="W-SUN")
    assert fq.label(week) == "wk 2024-03-04"
    assert fq.label(pd.Period("2024-03", freq="M")) == "2024-03"
