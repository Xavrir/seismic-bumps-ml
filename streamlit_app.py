"""Public Streamlit demo for the frozen Seismic Bumps hazard policy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import (
    CATEGORICAL_OPTIONS,
    DEFAULT_SAMPLE_INPUT_PATH,
    NUMERIC_COLUMNS,
    PREDICTION_COLUMNS,
    ScoringInputError,
    load_model_bundle,
    load_required_columns,
    load_sample_input,
    prediction_summary,
    read_csv_input,
    score_features,
)

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "artifacts" / "final_policy" / "final_policy.json"
DIAGNOSTICS_PATH = ROOT / "reports" / "figures" / "final_policy_diagnostics.png"
TREND_PATH = ROOT / "reports" / "figures" / "danger_level_trend.png"
RELIABILITY_PATH = ROOT / "reports" / "figures" / "calibration_reliability.png"
METRIC_CIS_PATH = ROOT / "artifacts" / "final_policy" / "lockbox_metric_cis.csv"
CALIBRATION_SUMMARY_PATH = ROOT / "artifacts" / "final_policy" / "calibration_summary.csv"

CATEGORICAL_LABELS = {
    "seismic": "Seismic assessment",
    "seismoacoustic": "Seismoacoustic assessment",
    "ghazard": "General hazard assessment",
    "shift": "Shift",
}

CATEGORICAL_HELP = {
    "seismic": "UCI ordinal code for seismic hazard assessment. Valid values: a, b, c, d.",
    "seismoacoustic": "UCI ordinal code for seismoacoustic hazard assessment. Valid values: a, b, c, d.",
    "ghazard": "UCI ordinal code for general hazard assessment. Valid values: a, b, c, d.",
    "shift": "UCI shift code. N is night shift, W is afternoon/evening shift.",
}

NUMERIC_LABELS = {
    "genergy": "Total seismic energy",
    "gpuls": "Total seismic pulses",
    "gdenergy": "Energy change",
    "gdpuls": "Pulse change",
    "nbumps": "Total bumps",
    "nbumps2": "Bumps in energy class 2",
    "nbumps3": "Bumps in energy class 3",
    "nbumps4": "Bumps in energy class 4",
    "nbumps5": "Bumps in energy class 5",
    "nbumps6": "Bumps in energy class 6",
    "nbumps7": "Bumps in energy class 7",
    "nbumps89": "Bumps in energy class 8-9",
    "energy": "Current shift energy",
    "maxenergy": "Maximum event energy",
}

NUMERIC_HELP = {
    "genergy": "Aggregated seismic energy reported for the shift context.",
    "gpuls": "Aggregated seismic pulse count reported for the shift context.",
    "gdenergy": "Change in seismic energy compared with prior context.",
    "gdpuls": "Change in seismic pulse count compared with prior context.",
    "nbumps": "Total recorded bump events in the shift context.",
    "nbumps2": "Count of bump events in UCI energy class 2.",
    "nbumps3": "Count of bump events in UCI energy class 3.",
    "nbumps4": "Count of bump events in UCI energy class 4.",
    "nbumps5": "Count of bump events in UCI energy class 5.",
    "nbumps6": "Count of bump events in UCI energy class 6.",
    "nbumps7": "Count of bump events in UCI energy class 7.",
    "nbumps89": "Count of bump events in combined UCI energy classes 8 and 9.",
    "energy": "Total energy from bump events in the current shift row.",
    "maxenergy": "Largest single bump-event energy in the current shift row.",
}

NUMERIC_GROUPS = (
    ("Energy and pulse signals", ("genergy", "gpuls", "gdenergy", "gdpuls")),
    ("Bump counts", ("nbumps", "nbumps2", "nbumps3", "nbumps4")),
    ("High-energy bump counts", ("nbumps5", "nbumps6", "nbumps7", "nbumps89")),
    ("Current shift energy", ("energy", "maxenergy")),
)


st.set_page_config(
    page_title="Seismic Risk Console",
    page_icon=":material/earthquake:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_model_bundle()


@st.cache_data(show_spinner=False)
def get_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_metric_cis() -> pd.DataFrame | None:
    if not METRIC_CIS_PATH.exists():
        return None
    return pd.read_csv(METRIC_CIS_PATH)


@st.cache_data(show_spinner=False)
def get_calibration_summary() -> pd.DataFrame | None:
    if not CALIBRATION_SUMMARY_PATH.exists():
        return None
    return pd.read_csv(CALIBRATION_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def get_sample_input() -> pd.DataFrame:
    return load_sample_input()


@st.cache_data(show_spinner=False)
def get_required_columns() -> list[str]:
    return load_required_columns()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
        :root {
          --sr-bg: #201d1d;
          --sr-surface: #302c2c;
          --sr-surface-2: #262323;
          --sr-text: #fdfcfc;
          --sr-muted: #b8b3b3;
          --sr-faint: #8c8787;
          --sr-border: rgba(253, 252, 252, 0.14);
          --sr-border-strong: rgba(253, 252, 252, 0.28);
          --sr-blue: #2f93ff;
          --sr-green: #30d158;
          --sr-orange: #ff9f0a;
          --sr-red: #ff453a;
          --sr-step: 150ms;
          --sr-ease: cubic-bezier(0.22, 1, 0.36, 1);
          /* z-index scale */
          --z-grain: 40;
          --z-scanline: 50;
          --z-sticky: 100;
          --z-overlay: 200;
          --z-tooltip: 300;
          /* tactical telemetry */
          --sr-display: "Space Grotesk", "Archivo", ui-sans-serif, system-ui, sans-serif;
          --sr-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          --sr-tick: var(--sr-border-strong);
        }

        .stApp {
          background:
            linear-gradient(90deg, rgba(253,252,252,0.035) 1px, transparent 1px),
            linear-gradient(180deg, rgba(253,252,252,0.03) 1px, transparent 1px),
            var(--sr-bg);
          background-size: 56px 56px;
          color: var(--sr-text);
          font-family: "Berkeley Mono", "IBM Plex Mono", ui-monospace,
            SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }

        .block-container {
          max-width: 1160px;
          padding-top: 32px;
          padding-bottom: 64px;
        }

        h1, h2, h3, h4, h5, p, label, span, div, button, input, textarea {
          font-family: "Berkeley Mono", "IBM Plex Mono", ui-monospace,
            SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }

        /* Fixed rem scale (product register): no fluid clamp on a tool. */
        h1 {
          font-size: 2.35rem;
          line-height: 1.1;
          letter-spacing: -0.02em;
          text-wrap: balance;
          margin: 6px 0 10px;
        }

        h2 { font-size: 1.5rem; letter-spacing: -0.01em; }
        h3 { font-size: 1.2rem; letter-spacing: -0.01em; }

        .sr-frame p,
        .sr-step p,
        .sr-frame .sr-lede {
          color: var(--sr-muted);
          max-width: 70ch;
          line-height: 1.55;
        }

        [data-testid="stMetric"],
        div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div {
          min-width: 0;
        }

        [data-testid="stMetric"] {
          background: var(--sr-surface-2);
          border: 1px solid var(--sr-border);
          border-radius: 6px;
          padding: 14px 16px;
          transition: border-color var(--sr-step) var(--sr-ease);
        }

        [data-testid="stMetric"]:hover {
          border-color: var(--sr-border-strong);
        }

        [data-testid="stMetricLabel"] p {
          color: var(--sr-faint);
          text-transform: uppercase;
          letter-spacing: 0.04em;
          font-size: 0.72rem;
        }

        [data-testid="stMetricValue"] {
          color: var(--sr-text);
          font-variant-numeric: tabular-nums;
        }

        [data-testid="stMetricDelta"] { color: var(--sr-muted); }

        /* ---- Hero ---- */
        .sr-frame {
          border: 1px solid var(--sr-border);
          border-radius: 8px;
          background:
            radial-gradient(120% 140% at 0% 0%, rgba(47,147,255,0.06), transparent 55%),
            var(--sr-surface-2);
          padding: 24px 26px;
          margin: 8px 0 24px;
        }

        .sr-console-title {
          display: flex;
          justify-content: space-between;
          gap: 24px;
          align-items: flex-start;
        }

        .sr-eyebrow {
          color: var(--sr-faint);
          font-size: 0.72rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
        }

        .sr-meta {
          color: var(--sr-muted);
          font-size: 0.8rem;
          line-height: 1.7;
          white-space: nowrap;
          border-left: 1px solid var(--sr-border);
          padding-left: 18px;
        }

        .sr-meta b { color: var(--sr-text); font-weight: 600; }

        .sr-label { color: var(--sr-muted); font-size: 0.82rem; line-height: 1.5; }

        .sr-disclaimer {
          display: flex;
          gap: 10px;
          align-items: baseline;
          border: 1px solid rgba(255, 159, 10, 0.4);
          border-radius: 6px;
          background: rgba(255, 159, 10, 0.08);
          color: var(--sr-text);
          padding: 12px 15px;
          margin-top: 20px;
          font-size: 0.86rem;
        }

        .sr-disclaimer::before {
          content: "DEMO";
          color: var(--sr-orange);
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          border: 1px solid rgba(255, 159, 10, 0.5);
          border-radius: 4px;
          padding: 2px 6px;
          flex: none;
        }

        /* ---- Start-here: flat numbered steps, no nested cards ---- */
        .sr-steps {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0;
          border: 1px solid var(--sr-border);
          border-radius: 8px;
          overflow: hidden;
          margin: 4px 0 26px;
          background: var(--sr-surface-2);
        }

        .sr-step {
          padding: 18px 20px;
          border-left: 1px solid var(--sr-border);
        }

        .sr-step:first-child { border-left: 0; }

        .sr-step-no {
          display: inline-block;
          color: var(--sr-blue);
          font-size: 0.78rem;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .sr-step strong {
          display: block;
          margin-bottom: 6px;
          font-size: 0.98rem;
        }

        .sr-step p { margin: 0; font-size: 0.86rem; }

        /* ---- Risk badges ---- */
        .sr-low, .sr-watch, .sr-dangerous {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          border: 1px solid currentColor;
          border-radius: 999px;
          padding: 5px 13px;
          font-weight: 700;
          font-size: 0.82rem;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        .sr-low::before, .sr-watch::before, .sr-dangerous::before {
          content: "";
          width: 7px; height: 7px;
          border-radius: 999px;
          background: currentColor;
          box-shadow: 0 0 8px currentColor;
        }

        .sr-low { color: var(--sr-green); }
        .sr-watch { color: var(--sr-orange); }
        .sr-dangerous { color: var(--sr-red); }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
          gap: 6px;
          border-bottom: 1px solid var(--sr-border);
        }

        .stTabs [data-baseweb="tab"] {
          border-radius: 6px 6px 0 0;
          border: 1px solid transparent;
          border-bottom: 0;
          background: transparent;
          color: var(--sr-muted);
          padding: 10px 16px;
          transition: color var(--sr-step) var(--sr-ease),
                      background var(--sr-step) var(--sr-ease);
        }

        .stTabs [data-baseweb="tab"]:hover { color: var(--sr-text); }

        .stTabs [aria-selected="true"] {
          background: var(--sr-surface-2);
          border-color: var(--sr-border);
          color: var(--sr-text);
        }

        /* ---- Buttons: full state vocabulary ---- */
        .stButton > button,
        .stDownloadButton > button {
          border-radius: 6px;
          border: 1px solid var(--sr-text);
          background: var(--sr-text);
          color: var(--sr-bg);
          font-weight: 700;
          min-height: 42px;
          transition: background var(--sr-step) var(--sr-ease),
                      border-color var(--sr-step) var(--sr-ease),
                      transform var(--sr-step) var(--sr-ease);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
          border-color: var(--sr-blue);
          background: var(--sr-blue);
          color: #fff;
        }

        .stButton > button:active,
        .stDownloadButton > button:active { transform: translateY(1px); }

        .stButton > button:focus-visible,
        .stDownloadButton > button:focus-visible,
        .stTabs [data-baseweb="tab"]:focus-visible {
          outline: 2px solid var(--sr-blue);
          outline-offset: 2px;
        }

        /* ---- Inputs ---- */
        div[data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stFileUploader"] section {
          border-radius: 6px;
          border-color: var(--sr-border);
          background-color: var(--sr-surface-2);
          color: var(--sr-text);
          transition: border-color var(--sr-step) var(--sr-ease);
        }

        div[data-baseweb="select"] > div:focus-within,
        [data-testid="stNumberInput"]:focus-within input,
        [data-testid="stTextInput"]:focus-within input {
          border-color: var(--sr-blue);
        }

        [data-testid="stWidgetLabel"] p,
        .stNumberInput label, .stSelectbox label {
          color: var(--sr-muted);
          font-size: 0.82rem;
        }

        [data-testid="stDataFrame"] {
          border: 1px solid var(--sr-border);
          border-radius: 6px;
        }

        /* ---- Re-theme Streamlit alert/info boxes to the palette ---- */
        [data-testid="stAlert"],
        [data-testid="stAlertContainer"],
        .stAlert > div {
          border-radius: 6px;
          border: 1px solid rgba(47, 147, 255, 0.3);
          background: rgba(47, 147, 255, 0.08) !important;
          color: var(--sr-text) !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlertContainer"] p {
          color: var(--sr-muted) !important;
        }

        /* ---- Motion: respect reduced-motion ---- */
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation: none !important; }
        }

        @media (max-width: 820px) {
          .block-container { padding-left: 16px; padding-right: 16px; }
          .sr-console-title { flex-direction: column; }
          .sr-meta { border-left: 0; padding-left: 0;
            border-top: 1px solid var(--sr-border); padding-top: 14px; }
          .sr-steps { grid-template-columns: 1fr; }
          .sr-step { border-left: 0; border-top: 1px solid var(--sr-border); }
          .sr-step:first-child { border-top: 0; }
        }

        /* ---- Tactical overlays: scanlines + grain (pointer-safe) ---- */
        .stApp::before {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: var(--z-scanline);
          background: repeating-linear-gradient(
            to bottom,
            rgba(0, 0, 0, 0.13) 0px,
            rgba(0, 0, 0, 0.13) 1px,
            transparent 1px,
            transparent 3px
          );
          opacity: 0.16;
        }

        .stApp::after {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: var(--z-grain);
          opacity: 0.05;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        }

        @media (prefers-reduced-motion: no-preference) {
          .stApp::before { animation: sr-scan 9s linear infinite; }
        }
        @keyframes sr-scan {
          from { background-position-y: 0; }
          to { background-position-y: 3px; }
        }

        /* ---- Crosshair corner framing ---- */
        .sr-crosshair { position: relative; }
        .sr-crosshair::after {
          content: "";
          position: absolute;
          inset: 7px;
          pointer-events: none;
          background:
            linear-gradient(var(--sr-tick), var(--sr-tick)) left top / 12px 1px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) left top / 1px 12px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) right top / 12px 1px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) right top / 1px 12px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) left bottom / 12px 1px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) left bottom / 1px 12px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) right bottom / 12px 1px no-repeat,
            linear-gradient(var(--sr-tick), var(--sr-tick)) right bottom / 1px 12px no-repeat;
        }

        /* ---- Risk-result hero ---- */
        .sr-readout {
          display: grid;
          grid-template-columns: auto 1fr;
          gap: 26px;
          align-items: center;
          border: 1px solid var(--sr-border);
          border-radius: 8px;
          background:
            radial-gradient(130% 130% at 0% 0%, rgba(47, 147, 255, 0.05), transparent 55%),
            var(--sr-surface-2);
          padding: 24px 26px;
          margin: 6px 0 14px;
        }

        .sr-gauge {
          --val: 0;
          --ring: var(--sr-faint);
          position: relative;
          width: 152px;
          height: 152px;
          border-radius: 999px;
          background: conic-gradient(
            var(--ring) calc(var(--val) * 1%),
            rgba(253, 252, 252, 0.07) 0
          );
          display: grid;
          place-items: center;
          flex: none;
        }
        .sr-gauge::before {
          content: "";
          position: absolute;
          inset: 13px;
          border-radius: 999px;
          background: var(--sr-bg);
          border: 1px solid var(--sr-border);
        }
        .sr-gauge-inner { position: relative; text-align: center; line-height: 1; }
        .sr-gauge-val {
          font-family: var(--sr-display);
          font-size: 3.05rem;
          font-weight: 700;
          color: var(--sr-text);
          font-variant-numeric: tabular-nums;
          text-shadow: 0 0 18px var(--ring);
        }
        .sr-gauge-unit {
          display: block;
          margin-top: 5px;
          font-size: 0.64rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--sr-faint);
        }

        .sr-verdict-eyebrow {
          font-size: 0.72rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--sr-faint);
          margin-bottom: 10px;
        }
        .sr-verdict-text {
          color: var(--sr-text);
          font-size: 1.02rem;
          line-height: 1.5;
          max-width: 54ch;
          margin: 12px 0 0;
        }

        /* ---- Telemetry readouts ---- */
        .sr-telemetry { display: flex; flex-wrap: wrap; gap: 10px 30px; margin-top: 16px; }
        .sr-readout-field { line-height: 1.3; }
        .sr-readout-field .k {
          display: block;
          font-size: 0.64rem;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: var(--sr-faint);
        }
        .sr-readout-field .v {
          font-family: var(--sr-mono);
          font-size: 1.02rem;
          color: var(--sr-text);
          font-variant-numeric: tabular-nums;
        }

        /* ---- Risk-level tiles + allocation bar ---- */
        .sr-tiles {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          margin: 4px 0 10px;
        }
        .sr-tile {
          border: 1px solid var(--sr-border);
          border-radius: 6px;
          background: var(--sr-surface-2);
          padding: 14px 16px;
        }
        .sr-tile .k {
          font-size: 0.64rem;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: var(--sr-faint);
          display: flex;
          align-items: center;
          gap: 7px;
        }
        .sr-tile .k::before {
          content: "";
          width: 7px;
          height: 7px;
          border-radius: 999px;
          background: currentColor;
          box-shadow: 0 0 8px currentColor;
        }
        .sr-tile .v {
          font-family: var(--sr-display);
          font-size: 1.85rem;
          font-weight: 700;
          font-variant-numeric: tabular-nums;
          color: var(--sr-text);
          margin-top: 6px;
        }
        .sr-tile.low .k { color: var(--sr-green); }
        .sr-tile.watch .k { color: var(--sr-orange); }
        .sr-tile.dangerous .k { color: var(--sr-red); }
        .sr-tile.dangerous.alert {
          border-color: var(--sr-red);
          background: rgba(255, 69, 58, 0.10);
          box-shadow: 0 0 0 1px var(--sr-red) inset;
        }

        .sr-alert-banner {
          display: flex;
          gap: 10px;
          align-items: baseline;
          border: 1px solid rgba(255, 69, 58, 0.45);
          background: rgba(255, 69, 58, 0.10);
          color: var(--sr-text);
          border-radius: 6px;
          padding: 10px 14px;
          margin: 2px 0 12px;
          font-size: 0.88rem;
        }
        .sr-alert-banner::before {
          content: "ALERT";
          color: var(--sr-red);
          font-size: 0.66rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          border: 1px solid rgba(255, 69, 58, 0.5);
          border-radius: 4px;
          padding: 2px 6px;
          flex: none;
        }

        .sr-alloc {
          display: flex;
          height: 8px;
          border-radius: 999px;
          overflow: hidden;
          border: 1px solid var(--sr-border);
          margin: 2px 0 16px;
          background: var(--sr-surface-2);
        }
        .sr-alloc i { display: block; height: 100%; }
        .sr-alloc i.low { background: var(--sr-green); }
        .sr-alloc i.watch { background: var(--sr-orange); }
        .sr-alloc i.dangerous { background: var(--sr-red); }

        /* ---- Awaiting-upload empty state ---- */
        .sr-empty {
          border: 1px dashed var(--sr-border-strong);
          border-radius: 8px;
          padding: 28px 24px;
          color: var(--sr-muted);
          background: var(--sr-surface-2);
        }
        .sr-empty .sr-empty-title {
          color: var(--sr-text);
          font-size: 1rem;
          margin-bottom: 6px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        @media (max-width: 820px) {
          .block-container { padding-top: 18px; padding-bottom: 36px; }
          h1 { font-size: 1.8rem; }
          .sr-frame { padding: 18px 16px; }
          .sr-readout { grid-template-columns: 1fr; gap: 16px; padding: 18px 16px; }
          .sr-gauge { width: 128px; height: 128px; }
          .sr-gauge-val { font-size: 2.6rem; }
          .sr-verdict-text { font-size: 0.96rem; }
          .sr-tiles { grid-template-columns: 1fr; gap: 8px; }
          .sr-steps .sr-step { padding: 14px 16px; }
          [data-testid="stMetric"] { padding: 10px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    safe_level = str(level).lower()
    css_class = {
        "low": "sr-low",
        "watch": "sr-watch",
        "dangerous": "sr-dangerous",
    }.get(safe_level, "sr-label")
    return f'<span class="{css_class}">{safe_level}</span>'


def render_header(policy: dict) -> None:
    lockbox = policy["lockbox_metrics"]
    st.markdown(
        f"""
        <div class="sr-frame sr-crosshair">
          <div class="sr-console-title">
            <div>
              <div class="sr-eyebrow">Frozen policy console</div>
              <h1>Seismic Risk Console</h1>
              <p>
                Score coal-mine shift records with the frozen Logistic Regression
                policy and inspect the safety-first threshold behind each alert.
              </p>
            </div>
            <div class="sr-meta">
              model<br><b>{policy["model"]}</b><br>
              threshold<br><b>{policy["operating_threshold"]:.3f}</b><br>
              watch floor<br><b>{policy["watch_floor"]:.2f}</b>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Lockbox recall",
        f"{lockbox['hazardous_recall']:.3f}",
        help="Recall: share of truly hazardous shifts the model catches. "
        "Higher means fewer missed hazards.",
    )
    metric_cols[1].metric(
        "Lockbox F2",
        f"{lockbox['hazardous_f2']:.3f}",
        help="F2: a balanced score that weights recall about 2x precision — "
        "right for a safety tool where misses cost more than false alarms.",
    )
    metric_cols[2].metric(
        "Lockbox AUC",
        f"{lockbox['roc_auc']:.3f}",
        help="AUC: how well the model ranks hazardous above non-hazardous shifts "
        "(0.5 = guessing, 1.0 = perfect).",
    )
    metric_cols[3].metric(
        "Precision",
        f"{lockbox['hazardous_precision']:.3f}",
        help="Precision: of the shifts flagged dangerous, the share that were "
        "truly hazardous.",
    )


def render_start_here() -> None:
    st.markdown(
        """
        <div class="sr-steps">
          <div class="sr-step">
            <span class="sr-step-no">01</span>
            <strong>Try one shift</strong>
            <p>Use the first tab if you do not have a CSV yet. Pick an example row,
            edit the monitoring values, then read the score below the form.</p>
          </div>
          <div class="sr-step">
            <span class="sr-step-no">02</span>
            <strong>Upload many shifts</strong>
            <p>Use the CSV tab for real files. Download the sample template, keep
            the same column names, replace the rows, then upload it.</p>
          </div>
          <div class="sr-step">
            <span class="sr-step-no">03</span>
            <strong>Check the output</strong>
            <p>Look for risk score, risk level, and dangerous flag. Low is routine,
            watch needs attention, dangerous crosses the frozen alert threshold.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_sample_row(samples: pd.DataFrame) -> tuple[int, pd.Series]:
    selected_index = st.selectbox(
        "Example row",
        options=list(range(len(samples))),
        format_func=lambda index: f"Example shift {index + 1}",
        help="Pick a bundled sample row, then adjust any fields below.",
    )
    return selected_index, samples.iloc[selected_index]


def build_manual_feature_row(sample: pd.Series, sample_index: int) -> pd.DataFrame:
    st.markdown("##### Monitoring inputs")
    st.caption(
        "Categorical assessments are shown below; expand the panel to fine-tune the "
        "numeric signals. The console above updates as you edit."
    )

    categorical_cols = st.columns(4)
    categorical_values = {}
    for index, (column, options) in enumerate(CATEGORICAL_OPTIONS.items()):
        default_value = str(sample[column])
        categorical_values[column] = categorical_cols[index].selectbox(
            f"{CATEGORICAL_LABELS[column]} ({column})",
            options=options,
            index=options.index(default_value) if default_value in options else 0,
            help=CATEGORICAL_HELP[column],
            key=f"input_{sample_index}_{column}",
        )

    numeric_values = {}
    with st.expander("Adjust monitoring inputs", expanded=False):
        for group_name, columns in NUMERIC_GROUPS:
            st.markdown(f"###### {group_name}")
            cols = st.columns(4)
            for offset, column in enumerate(columns):
                numeric_values[column] = cols[offset].number_input(
                    f"{NUMERIC_LABELS[column]} ({column})",
                    value=float(sample[column]),
                    step=1.0,
                    format="%.3f",
                    help=NUMERIC_HELP[column],
                    key=f"input_{sample_index}_{column}",
                )

    row = {**categorical_values, **numeric_values}
    ordered_row = {column: row[column] for column in get_required_columns()}
    return pd.DataFrame([ordered_row])


def render_single_score(bundle: dict) -> None:
    st.caption(
        "Pick an example shift, then read the live risk console. Expand the inputs "
        "below to tune the monitoring signals."
    )
    sample_index, sample = select_sample_row(get_sample_input())
    result_slot = st.container()
    features = build_manual_feature_row(sample, sample_index)

    try:
        prediction = score_features(features, bundle)
    except ScoringInputError as exc:
        result_slot.error(str(exc))
        return

    threshold = float(bundle["threshold"])
    watch_floor = float(bundle.get("watch_floor", 0.04))
    result = prediction.iloc[0]
    level = str(result["risk_level"]).lower()
    score = int(result["risk_score"])
    probability = float(result["predicted_probability"])
    flag = int(result["dangerous_flag"])
    ring = {
        "low": "var(--sr-green)",
        "watch": "var(--sr-orange)",
        "dangerous": "var(--sr-red)",
    }.get(level, "var(--sr-faint)")

    if level == "dangerous":
        verdict = (
            f"Hazardous shift predicted. The calibrated probability {probability:.2f} "
            f"is at or above the {threshold:.3f} alert threshold — treat as an "
            "actionable hazard alert."
        )
    elif level == "watch":
        verdict = (
            f"Elevated but sub-threshold. Probability {probability:.2f} sits in the "
            f"watch band ({watch_floor:.2f}-{threshold:.3f}) — heightened monitoring, "
            "not yet an alert."
        )
    else:
        verdict = (
            f"Baseline conditions. Probability {probability:.2f} is below the "
            f"{watch_floor:.2f} watch floor."
        )

    with result_slot:
        st.markdown(
            f"""
            <div class="sr-readout sr-crosshair">
              <div class="sr-gauge" style="--val: {score}; --ring: {ring};">
                <div class="sr-gauge-inner">
                  <span class="sr-gauge-val">{score}</span>
                  <span class="sr-gauge-unit">risk / 100</span>
                </div>
              </div>
              <div>
                <div class="sr-verdict-eyebrow">Shift assessment</div>
                {risk_badge(level)}
                <p class="sr-verdict-text">{verdict}</p>
                <div class="sr-telemetry">
                  <div class="sr-readout-field" title="Calibrated probability that the next shift is hazardous (0 to 1).">
                    <span class="k">probability</span>
                    <span class="v">{probability:.3f}</span></div>
                  <div class="sr-readout-field" title="Alert threshold: at or above this probability a shift is flagged dangerous. Set from a 10:1 miss-to-false-alarm cost.">
                    <span class="k">alert threshold</span>
                    <span class="v">{threshold:.3f}</span></div>
                  <div class="sr-readout-field" title="1 = flagged dangerous (probability is at or above the threshold), otherwise 0.">
                    <span class="k">dangerous flag</span>
                    <span class="v">{flag}</span></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Raw output"):
            st.dataframe(
                prediction.loc[:, list(PREDICTION_COLUMNS)],
                width="stretch",
                hide_index=True,
            )


def render_batch_score(bundle: dict) -> None:
    st.subheader("Batch score CSV")
    st.caption(
        f"Use this when you already have multiple shift rows. Required columns match "
        f"`{DEFAULT_SAMPLE_INPUT_PATH.name}`."
    )
    st.markdown(
        """
        1. **Download the example CSV** below — it already has the right columns.
        2. Keep the header row exactly the same.
        3. Replace or add shift rows with your values.
        4. Upload the completed CSV here.
        5. Check the summary cards and download the scored output.
        """
    )
    with DEFAULT_SAMPLE_INPUT_PATH.open("rb") as sample_file:
        st.download_button(
            "Download example CSV (template)",
            sample_file,
            file_name="seismic_sample_input.csv",
            mime="text/csv",
        )
    st.caption("This is what a valid file looks like — keep these exact column names:")
    st.dataframe(get_sample_input().head(2), width="stretch", hide_index=True)

    with st.expander("CSV schema", expanded=False):
        schema_rows = []
        for column in get_required_columns():
            if column in CATEGORICAL_OPTIONS:
                schema_rows.append(
                    {
                        "column": column,
                        "kind": "categorical",
                        "accepted values": ", ".join(CATEGORICAL_OPTIONS[column]),
                    }
                )
            else:
                schema_rows.append(
                    {
                        "column": column,
                        "kind": "numeric",
                        "accepted values": "number",
                    }
                )
        st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

    uploaded = st.file_uploader("Upload shift feature CSV", type=["csv"])
    if uploaded is None:
        st.markdown(
            """
            <div class="sr-empty sr-crosshair">
              <div class="sr-empty-title">Awaiting upload</div>
              Download the template, keep the header row, replace the shift rows, then
              upload here to score the whole batch at once.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Preview expected format"):
            st.dataframe(get_sample_input().head(3), width="stretch", hide_index=True)
        return

    try:
        features = read_csv_input(uploaded)
        predictions = score_features(features, bundle)
    except ScoringInputError as exc:
        st.error(str(exc))
        return

    summary = prediction_summary(predictions)
    total = max(summary["low"] + summary["watch"] + summary["dangerous"], 1)
    danger = summary["dangerous"]
    danger_cls = "sr-tile dangerous alert" if danger else "sr-tile dangerous"
    banner = (
        f'<div class="sr-alert-banner">{danger} shift'
        f'{"s" if danger != 1 else ""} flagged dangerous — review before the shift '
        "starts.</div>"
        if danger
        else ""
    )
    st.markdown(
        f"""
        {banner}
        <div class="sr-tiles">
          <div class="sr-tile low"><div class="k">low</div>
            <div class="v">{summary["low"]}</div></div>
          <div class="sr-tile watch"><div class="k">watch</div>
            <div class="v">{summary["watch"]}</div></div>
          <div class="{danger_cls}"><div class="k">dangerous</div>
            <div class="v">{danger}</div></div>
        </div>
        <div class="sr-alloc">
          <i class="low" style="width: {summary["low"] / total * 100:.1f}%"></i>
          <i class="watch" style="width: {summary["watch"] / total * 100:.1f}%"></i>
          <i class="dangerous" style="width: {danger / total * 100:.1f}%"></i>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The table below is your original CSV plus prediction columns. Download it "
        "when you want to keep or share the scored results."
    )
    st.dataframe(predictions, width="stretch", hide_index=True)
    st.download_button(
        "Download scored CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="seismic_risk_predictions.csv",
        mime="text/csv",
    )


def render_model_evidence(policy: dict) -> None:
    st.subheader("Model evidence")
    st.write(
        "The policy was selected by development cross-validation using hazardous-class "
        "F2, then checked once on the lockbox split. Recall and F2 are emphasized because "
        "missed hazardous shifts are worse than false alarms in this demo setting."
    )

    with st.expander("What do these terms mean?"):
        st.markdown(
            """
- **Risk score** — the hazard probability shown as 0–100.
- **Threshold (0.080)** — at or above this, a shift is flagged **dangerous**.
- **Watch floor (0.04)** — below this is **low**; between the two is **watch**.
- **Recall** — share of truly hazardous shifts the model catches (a miss is the costly error).
- **Precision** — of the shifts flagged dangerous, how many were truly hazardous.
- **F2** — an accuracy score weighting recall about 2× precision (fits a safety tool).
- **AUC** — how well the model ranks hazardous above non-hazardous (0.5 = guessing, 1.0 = perfect).
            """
        )

    cv = policy["cv_metrics"]
    lockbox = policy["lockbox_metrics"]
    evidence = pd.DataFrame(
        [
            {
                "split": "dev CV mean",
                "recall": cv["recall_mean"],
                "f2": cv["f2_mean"],
                "auc": cv["auc_mean"],
                "precision": None,
            },
            {
                "split": "lockbox",
                "recall": lockbox["hazardous_recall"],
                "f2": lockbox["hazardous_f2"],
                "auc": lockbox["roc_auc"],
                "precision": lockbox["hazardous_precision"],
            },
        ]
    )
    st.dataframe(evidence, width="stretch", hide_index=True)

    image_cols = st.columns(2)
    if DIAGNOSTICS_PATH.exists():
        image_cols[0].image(str(DIAGNOSTICS_PATH), caption="Final policy diagnostics")
    if TREND_PATH.exists():
        image_cols[1].image(str(TREND_PATH), caption="Risk-level trend")


def render_methodology(policy: dict) -> None:
    st.subheader("Methodology")
    watch_floor = float(policy.get("watch_floor", 0.30))
    threshold = float(policy["operating_threshold"])
    st.markdown(
        f"""
        - Dataset: UCI Seismic Bumps, 2,584 shift records and 18 features.
        - Task: predict whether the next 8-hour shift is hazardous.
        - Class balance: about 6.6% hazardous, so accuracy is not the main metric.
        - Model: `{policy["model"]}` with `{policy["hyperparams"]["class_weight"]}` class weighting.
        - Decision rule: `{policy["danger_rule"]}`.
        - Risk levels: `low` below {watch_floor:.2f}, `watch` from {watch_floor:.2f} to
          {threshold:.3f}, `dangerous` at {threshold:.3f} or above.

        Limitation: the source dataset has no explicit timestamps, so the original
        model uses stratified splits rather than true temporal validation.
        """
    )

    st.markdown("#### Calibration & cost policy")
    calibration = get_calibration_summary()
    if calibration is not None and not calibration.empty:
        rows = calibration.set_index("stage")
        improved = ""
        if {"uncalibrated", "calibrated"}.issubset(rows.index):
            before = float(rows.loc["uncalibrated", "brier"])
            after = float(rows.loc["calibrated", "brier"])
            improved = (
                f" Brier score improves from {before:.3f} to {after:.3f} "
                "(lower is better)."
            )
        method = get_bundle().get("calibration_method", "isotonic")
        st.markdown(
            f"Predicted probabilities are calibrated (`{method}`), so the 0-100 risk "
            f"score approximates real hazard frequency rather than the raw, "
            f"class-weight-distorted output.{improved}"
        )
    else:
        st.caption(
            "Run `scripts/evaluate_calibration.py` to generate calibration diagnostics."
        )

    if "cost_matrix" in policy:
        cost = policy["cost_matrix"]
        st.markdown(
            f"The operating threshold follows an explicit cost assumption: a missed "
            f"hazard costs `{cost['ratio']:.0f}x` a false alarm "
            f"(FN={cost['fn']:.0f}, FP={cost['fp']:.0f}). Cost-optimal threshold on "
            f"dev out-of-fold scores: `{policy.get('cost_optimal_threshold', float('nan')):.3f}`; "
            f"active basis: `{policy.get('threshold_basis', 'f2')}`."
        )

    cis = get_metric_cis()
    if cis is not None and not cis.empty:
        st.markdown("#### Lockbox metrics with 95% confidence intervals")
        display = cis.assign(
            estimate=lambda d: d["point_estimate"].map("{:.3f}".format),
            ci=lambda d: d.apply(
                lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1
            ),
        )[["metric", "estimate", "ci"]]
        st.dataframe(display, width="stretch", hide_index=True)
        st.caption(
            "Wide intervals reflect the small hazardous sample (~26 cases) in the "
            "lockbox split — a reason to treat single-point metrics cautiously."
        )

    if RELIABILITY_PATH.exists():
        st.image(
            str(RELIABILITY_PATH),
            caption="Reliability diagram and calibrated score distribution (lockbox)",
        )


def main() -> None:
    inject_styles()
    policy = get_policy()
    bundle = get_bundle()
    render_header(policy)
    render_start_here()

    score_tab, batch_tab, evidence_tab, methodology_tab = st.tabs(
        ["Try one shift", "Upload CSV", "Model evidence", "Methodology"]
    )
    with score_tab:
        render_single_score(bundle)
    with batch_tab:
        render_batch_score(bundle)
    with evidence_tab:
        render_model_evidence(policy)
    with methodology_tab:
        render_methodology(policy)


if __name__ == "__main__":
    main()
