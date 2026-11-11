"""Headless UI tests: drive the Streamlit app with AppTest."""
import pytest
from streamlit.testing.v1 import AppTest

from engine import demo
from ui.state import STEPS

APP = "../app.py"


def new_app(**session):
    at = AppTest.from_file(APP, default_timeout=60)
    session = {"source": "demo", **session}
    for k, val in session.items():
        at.session_state[k] = val
    return at


def go(at, step):
    at.session_state["step"] = step
    at.session_state["w_step"] = step
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("key", list(demo.PRODUCTS))
def test_default_path_through_all_steps(key):
    at = new_app(demo_key=key)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    for step in STEPS[1:]:
        go(at, step)
        problems = [e.value for e in at.error if "tracking signal" not in e.value]  # alarms are results
        assert not problems, problems
    # Forecast step: forecast chart and 80% range present
    assert any("80% range" in c.value for c in at.caption)
    assert any(b.label == "Download CSV" for b in at.get("download_button"))


FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "messy_two_skus.csv"


def test_duplicates_block_later_steps_until_resolved():
    data = FIXTURE.read_bytes()
    at = new_app(source="upload", upload_name="messy.csv", upload_bytes=data)
    at.run()
    at.selectbox(key="w_sku_pick").set_value("SKU-207").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("duplicate period" in e.value for e in at.error)
    next_button = next(b for b in at.button if b.label.startswith("Next"))
    assert next_button.disabled
    go(at, "Models")  # trying to open a later step falls back to Data
    assert any("Finish the Data step first" in w.value for w in at.warning)
    # resolve duplicates, then gaps
    at.session_state["dup_choice"] = {"SKU-207": "sum"}
    at.session_state["gap_choice"] = {"SKU-207": "ffill"}
    go(at, "Models")
    assert not any("Finish the Data step first" in w.value for w in at.warning)
    assert at.session_state["w_model_view"] in ("ses", "holt", "winters", "regression")  # Models screen drawn


def test_sku_selector_lists_both_skus():
    at = new_app(source="upload", upload_name="messy.csv", upload_bytes=FIXTURE.read_bytes())
    at.run()
    picker = at.selectbox(key="w_sku_pick")
    assert [o.split(" · ")[0] for o in picker.options] == ["SKU-114", "SKU-207"]
    assert "issue" in picker.options[1]


def at_models(method="holt"):
    at = new_app(demo_key="paint")
    at.run()
    go(at, "Models")
    at.session_state["model_view"] = method
    at.session_state["w_model_view"] = method
    at.run()
    return at


def test_slider_updates_fit():
    at = at_models("holt")
    before = at.caption  # training fit caption changes with alpha
    fit_before = next(c.value for c in at.caption if c.value.startswith("Training fit"))
    at.slider(key="w_p_holt_alpha").set_value(0.9).run()
    assert not at.exception
    assert at.session_state["params"]["holt"]["alpha"] == 0.9
    fit_after = next(c.value for c in at.caption if c.value.startswith("Training fit"))
    assert fit_before != fit_after


def test_optimise_marks_tuned():
    at = at_models("ses")
    at.button(key="opt_ses").click().run()
    assert not at.exception
    assert any("Tuned on training data only" in c.value for c in at.caption)


def test_start_value_override_and_reset():
    at = at_models("holt")
    box = next(n for n in at.number_input if n.key.startswith("w_sv_holt_level_"))
    box.set_value(250.0).run()
    assert at.session_state["overrides"]["holt"]["level"] == 250.0
    assert any("overridden" in n.label for n in at.number_input)
    at.button(key="reset_holt").click().run()
    assert "holt" not in at.session_state["overrides"]
    assert not any("overridden" in n.label for n in at.number_input)


def test_compare_sentence_matches_engine():
    from engine import results
    from engine import validate as v

    at = new_app(demo_key="paint")
    at.run()
    go(at, "Compare")
    frame = demo.load("paint")
    raw = v.parse(frame, demo.mapping("paint")).skus[v.SINGLE_SKU]
    series = v.build_series(raw, v.diagnose(raw), raw.freq, driver_name=frame.columns[2])
    expected = results.run(series).comparison.recommended
    assert at.session_state["chosen"] in (None, expected)
    assert any("Recommended: **Winters" in m.value for m in at.markdown) == (expected == "winters")


def test_tooltips_are_short():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for path in [root / "app.py", *(root / "ui").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "help" and isinstance(node.value, ast.Constant):
                words = len(str(node.value.value).split())
                assert words <= 15, (path.name, node.value.value)


def test_results_updated_note_after_split_change():
    at = new_app(demo_key="paint")
    at.run()
    go(at, "Models")
    go(at, "Split & clean")
    at.slider(key="w_test_length").set_value(6).run()
    go(at, "Models")
    assert any("Results updated: test period changed" in c.value for c in at.caption)


def test_upload_is_the_default_start():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.session_state["source"] == "upload"
    assert any("Upload a sales file to begin" in i.value for i in at.info)
    next_button = next(b for b in at.button if b.label.startswith("Next"))
    assert next_button.disabled


def test_reset_clears_everything_after_confirmation():
    at = new_app(demo_key="ac")
    at.run()
    go(at, "Compare")
    at.button(key="reset_all").click().run()
    assert not at.exception
    # the confirmation dialog is open; nothing has been cleared yet
    assert at.session_state["step"] == "Compare"
    assert any("Your progress will be lost" in w.value for w in at.warning)
    at.button(key="reset_cancel").click().run()  # cancel keeps everything
    assert at.session_state["step"] == "Compare" and not at.session_state["confirm_reset"]
    at.button(key="reset_all").click().run()
    at.button(key="reset_confirm").click().run()
    assert not at.exception
    assert at.session_state["step"] == "Data"
    assert at.session_state["source"] == "upload"
    assert at.session_state["demo_key"] == "paint"


def test_pick_a_test_file():
    from ui.steps import TEST_DIR, TEST_FILES

    assert all((TEST_DIR / k).exists() for k in TEST_FILES)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.selectbox(key="src_testfile").set_value("synthetic/13_daily_pharmacy_pos_export_messy.csv").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["upload_origin"] == "test"
    assert any("Test file loaded" in c.value for c in at.caption)
    assert any("duplicate period" in e.value for e in at.error)  # the messy file's duplicates are caught
