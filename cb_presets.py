"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus
dem OR-Demo-Portfolio, siehe ld_presets.py in leiden-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cb_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _degree_caster(v):
    v = int(float(v))
    return v if v in (1, 2, 3) else C.DEFAULT_DEGREE_A


SETTING_SPECS = {
    "n_points_slider": SettingSpec("n", int, C.DEFAULT_N_POINTS, C.N_POINTS_MIN, C.N_POINTS_MAX),
    "degree_a_select": SettingSpec("da", _degree_caster, C.DEFAULT_DEGREE_A),
    "degree_b_select": SettingSpec("db", _degree_caster, C.DEFAULT_DEGREE_B),
    "offset_slider": SettingSpec("off", float, C.DEFAULT_OFFSET, C.OFFSET_MIN, C.OFFSET_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "n_neighbors_slider": SettingSpec(
        "nn", int, C.DEFAULT_N_NEIGHBORS, C.N_NEIGHBORS_MIN, C.N_NEIGHBORS_MAX
    ),
    "tau_eps_slider": SettingSpec("tau", float, C.DEFAULT_TAU_EPS, C.TAU_EPS_MIN, C.TAU_EPS_MAX),
    "r_merge_slider": SettingSpec("rm", float, C.DEFAULT_R_MERGE, C.R_MERGE_MIN, C.R_MERGE_MAX),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_points, degree_a, degree_b, offset, noise, seed, n_neighbors, tau_eps, r_merge):
    try:
        st.query_params["n"] = str(int(n_points))
        st.query_params["da"] = str(int(degree_a))
        st.query_params["db"] = str(int(degree_b))
        st.query_params["off"] = str(offset)
        st.query_params["noise"] = str(noise)
        st.query_params["seed"] = str(int(seed))
        st.query_params["nn"] = str(int(n_neighbors))
        st.query_params["tau"] = str(tau_eps)
        st.query_params["rm"] = str(r_merge)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_points_slider"] = p["n_points"]
    st.session_state["degree_a_select"] = p["degree_a"]
    st.session_state["degree_b_select"] = p["degree_b"]
    st.session_state["offset_slider"] = p["offset"]
    st.session_state["noise_slider"] = p["noise"]
    st.session_state["n_neighbors_slider"] = p["n_neighbors"]
    st.session_state["tau_eps_slider"] = p["tau_eps"]
    st.session_state["r_merge_slider"] = p["r_merge"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
