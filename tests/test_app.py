"""End-to-end Smoke-Test via Streamlits offizielles AppTest-Framework: laedt app.py mit
den Standardeinstellungen sowie allen Presets und prueft, dass kein Python-Fehler
auftritt - ein Fehler wie `StreamlitDuplicateElementId` (zwei st.plotly_chart-Aufrufe
ohne eindeutiges key=) liegt in app.py's Widget-Verdrahtung selbst und kann nur durch
einen echten End-to-End-Lauf gefunden werden, nicht durch Unit-Tests der
Algorithmus-/Visualisierungs-Module (etabliertes Muster aus hdbscan-demo/leiden-demo)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import cb_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("preset_name", list(C.PRESETS.keys()))
def test_all_presets_load_without_exception(preset_name):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    buttons = {b.label: b for b in at.button}
    buttons[preset_name].click().run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


def test_nested_step_sliders_are_navigable():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    outer_slider = at.slider(key="cb_outer_step")
    outer_slider.set_value(0).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    inner_slider = at.slider(key="cb_inner_step")
    inner_slider.set_value(0).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("preset_name", [None, *C.PRESETS.keys()])
def test_every_outer_step_position_is_navigable(preset_name):
    """Regression test for a real crash: st.slider raises StreamlitInvalidMinMaxError
    when min_value == max_value, which happens whenever a CluB has exactly ONE
    refinement step (n_inner == 1, so the inner slider's bounds would be (0, 0)) - hit
    at outer_idx=5 with default settings. Sweeps EVERY outer-slider position, not just
    0, since the bug only shows up for specific CluB-Funde, not the first one."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    if preset_name is not None:
        buttons = {b.label: b for b in at.button}
        buttons[preset_name].click().run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    n_outer = int(at.slider(key="cb_outer_step").max) + 1
    for outer_idx in range(n_outer):
        at_i = AppTest.from_file(APP_PATH)
        at_i.run(timeout=120)
        if preset_name is not None:
            buttons_i = {b.label: b for b in at_i.button}
            buttons_i[preset_name].click().run(timeout=120)
        at_i.slider(key="cb_outer_step").set_value(outer_idx).run(timeout=120)
        assert not at_i.exception, (outer_idx, [str(e) for e in at_i.exception])
