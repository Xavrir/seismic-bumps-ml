"""Public Streamlit demo for the frozen Seismic Bumps hazard policy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.scoring import (
    CATEGORICAL_OPTIONS,
    DEFAULT_SAMPLE_INPUT_PATH,
    NUMERIC_COLUMNS,
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
    page_icon="",
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
def get_sample_input() -> pd.DataFrame:
    return load_sample_input()


@st.cache_data(show_spinner=False)
def get_required_columns() -> list[str]:
    return load_required_columns()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
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
          --z-sticky: 100;
          --z-overlay: 200;
          --z-tooltip: 300;
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
        <div class="sr-frame">
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
          <div class="sr-disclaimer">
            Research demo only. This is not certified operational mine-safety software.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Lockbox recall", f"{lockbox['hazardous_recall']:.3f}")
    metric_cols[1].metric("Lockbox F2", f"{lockbox['hazardous_f2']:.3f}")
    metric_cols[2].metric("Lockbox AUC", f"{lockbox['roc_auc']:.3f}")
    metric_cols[3].metric("Precision", f"{lockbox['hazardous_precision']:.3f}")


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
    st.subheader("Score one shift")
    st.caption("Score updates from the selected example values and edited fields.")

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
    for group_name, columns in NUMERIC_GROUPS:
        st.markdown(f"##### {group_name}")
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
    st.info(
        "Use this tab when you want to test one shift manually. The current result "
        "appears below the input fields."
    )
    sample_index, sample = select_sample_row(get_sample_input())
    features = build_manual_feature_row(sample, sample_index)

    try:
        prediction = score_features(features, bundle)
    except ScoringInputError as exc:
        st.error(str(exc))
        return

    result = prediction.iloc[0]
    st.markdown("#### Current shift score")
    st.caption(
        "Read this section after changing fields. Risk score is 0-100, risk level "
        "is the human-readable alert band, and dangerous flag is 1 only when the "
        "frozen threshold is crossed."
    )
    left, right = st.columns([1, 2])
    left.metric("Risk score", f"{int(result['risk_score'])}/100")
    left.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
    right.dataframe(
        prediction.loc[
            :,
            [
                "predicted_probability",
                "risk_score",
                "risk_level",
                "dangerous_flag",
            ],
        ],
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
        1. Download the sample input CSV.
        2. Keep the header row exactly the same.
        3. Replace or add shift rows with your values.
        4. Upload the completed CSV here.
        5. Check the summary cards and download the scored output.
        """
    )
    with DEFAULT_SAMPLE_INPUT_PATH.open("rb") as sample_file:
        st.download_button(
            "Download sample input CSV",
            sample_file,
            file_name="seismic_sample_input.csv",
            mime="text/csv",
        )

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
        st.dataframe(get_sample_input().head(3), width="stretch", hide_index=True)
        st.caption("Preview of the expected upload format.")
        return

    try:
        features = read_csv_input(uploaded)
        predictions = score_features(features, bundle)
    except ScoringInputError as exc:
        st.error(str(exc))
        return

    summary = prediction_summary(predictions)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Low", summary["low"])
    summary_cols[1].metric("Watch", summary["watch"])
    summary_cols[2].metric("Dangerous", summary["dangerous"])
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
    st.markdown(
        f"""
        - Dataset: UCI Seismic Bumps, 2,584 shift records and 18 features.
        - Task: predict whether the next 8-hour shift is hazardous.
        - Class balance: about 6.6% hazardous, so accuracy is not the main metric.
        - Model: `{policy["model"]}` with `{policy["hyperparams"]["class_weight"]}` class weighting.
        - Decision rule: `{policy["danger_rule"]}`.
        - Risk levels: `low` below 0.30, `watch` from 0.30 to threshold, `dangerous` at threshold or above.

        Limitation: the source dataset has no explicit timestamps, so the original
        model uses stratified splits rather than true temporal validation.
        """
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
