"""Analyze Seismic Risk Console user-testing responses into PPT-ready output.

Ingests the Google Form responses CSV and produces:
  * SUS (System Usability Scale) score per participant + mean, with a grade band.
  * Usefulness item means + overall.
  * Per-task difficulty means and success rates.
  * Charts in reports/figures/user_testing/ (clean light style for slides).
  * A markdown summary table + a collated qualitative-responses file.

Column matching is tolerant: for each canonical code (e.g. ``sus1``) it accepts a
column named exactly ``sus1`` OR any header containing ``[sus1]`` (so a raw Google
Forms export works once questions are titled with the ``[code]`` prefix).

Run from the project root (use the project venv that has pandas + matplotlib):
    python scripts/analyze_user_testing.py
    python scripts/analyze_user_testing.py --input path/to/responses.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- canonical codes -------------------------------------------------------
BG_CODES = ["pid", "major", "year", "ml_fam", "dash_fam", "device"]
TASKS = ["t1", "t2", "t3", "t4", "t5"]
SUS_CODES = [f"sus{i}" for i in range(1, 11)]
USE_CODES = [f"use{i}" for i in range(1, 6)]
OPEN_CODES = ["q_confuse", "q_like", "q_suggest", "q_clarity"]
OPEN_LABELS = {
    "q_confuse": "Paling membingungkan / sulit",
    "q_like": "Yang disukai",
    "q_suggest": "Saran perbaikan",
    "q_clarity": "Kejelasan output risiko",
}
USE_LABELS = {
    "use1": "Paham arti risk level",
    "use2": "Output berguna menilai bahaya",
    "use3": "Percaya sebagai alat bantu",
    "use4": "Evidence/Methodology membantu",
    "use5": "Penjelasan transparan",
}
TASK_LABELS = {
    "t1": "T1 Memahami fungsi",
    "t2": "T2 Baca gauge/level/verdict",
    "t3": "T3 Ubah input",
    "t4": "T4 Upload CSV",
    "t5": "T5 Recall & threshold",
}
SUS_ITEM_LABELS = {
    1: "Would use often",
    2: "Not too complex",
    3: "Easy to use",
    4: "No tech help needed",
    5: "Well integrated",
    6: "Consistent (not messy)",
    7: "Quick to learn",
    8: "Not awkward",
    9: "Felt confident",
    10: "Little to learn first",
}
SUCCESS_MAP = {"selesai": 1.0, "sebagian": 0.5, "gagal": 0.0}

# Fallback: match a raw Google Forms header by its question text (case-insensitive
# substring) when the [code] prefix is missing (e.g. questions edited by hand).
TEXT_FALLBACK = {
    "major": ["program studi"],
    "year": ["jenjang kuliah", "tahun / jenjang"],
    "ml_fam": ["familiar kamu dengan machine learning"],
    "dash_fam": ["aplikasi data", "dashboard"],
    "device": ["perangkat yang dipakai"],
    "q_confuse": ["paling membingungkan", "atau sulit? kenapa"],
    "q_like": ["yang kamu sukai", "kamu sukai"],
    "q_suggest": ["saran perbaikan"],
    "q_clarity": ["output risiko sudah jelas", "jelas/mudah dipahami"],
}

# Brand palette on a light canvas (readable on any slide).
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_INK = "#2f93ff", "#2a9d4a", "#e08a00", "#e0382c", "#201d1d"


def _resolve(df: pd.DataFrame, code: str) -> pd.Series | None:
    """Return the column for a canonical code (exact name or ``[code]`` in header)."""
    if code in df.columns:
        return df[code]
    pat = re.compile(r"\[" + re.escape(code) + r"\]")
    for col in df.columns:
        if pat.search(str(col)):
            return df[col]
    for needle in TEXT_FALLBACK.get(code, []):
        for col in df.columns:
            if needle.lower() in str(col).lower():
                return df[col]
    return None


def _numeric(df: pd.DataFrame, code: str) -> pd.Series:
    series = _resolve(df, code)
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#e6e3e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def compute_sus(df: pd.DataFrame) -> pd.Series:
    """Per-participant SUS 0-100 (odd items positive, even items negative)."""
    cols = {code: _numeric(df, code) for code in SUS_CODES}
    missing = [c for c, s in cols.items() if s.empty]
    if missing:
        print(f"  [warn] SUS skipped — missing items: {', '.join(missing)}")
        return pd.Series(dtype=float)
    frame = pd.DataFrame(cols)
    odd = [f"sus{i}" for i in (1, 3, 5, 7, 9)]
    even = [f"sus{i}" for i in (2, 4, 6, 8, 10)]
    contrib = (frame[odd] - 1).sum(axis=1) + (5 - frame[even]).sum(axis=1)
    return contrib * 2.5


def sus_grade(score: float) -> str:
    if score >= 80.3:
        return "A (Excellent)"
    if score >= 74:
        return "B (Good)"
    if score >= 68:
        return "C (OK, above average)"
    if score >= 51:
        return "D (Poor)"
    return "F (Awful)"


def plot_sus(pids, sus, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pids, sus.values, color=C_BLUE, edgecolor=C_INK, linewidth=0.6)
    mean = float(np.nanmean(sus.values))
    ax.axhline(mean, color=C_INK, ls="-", lw=1.2, label=f"mean {mean:.1f}")
    ax.axhline(68, color=C_ORANGE, ls="--", lw=1, label="industry avg 68")
    for x, v in zip(pids, sus.values):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=C_INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("SUS score (0-100)")
    ax.set_title("System Usability Scale per participant")
    ax.legend(loc="lower right", fontsize=8)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(outdir / "sus_per_participant.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_sus_items(df: pd.DataFrame, outdir: Path) -> None:
    """One chart for all 10 SUS items, normalized so higher always = better usability."""
    cols = {c: _numeric(df, c) for c in SUS_CODES}
    if any(s.empty for s in cols.values()):
        return
    labels, values = [], []
    for i, code in enumerate(SUS_CODES, start=1):
        mean = float(cols[code].mean())
        contrib = (mean - 1) if i % 2 == 1 else (5 - mean)  # 0..4, higher = better
        labels.append(SUS_ITEM_LABELS[i])
        values.append(contrib / 4 * 100)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [C_GREEN if v >= 75 else C_ORANGE if v >= 50 else C_RED for v in values]
    ax.barh(labels, values, color=colors, edgecolor=C_INK, linewidth=0.6)
    for y, v in enumerate(values):
        ax.text(v + 1.5, y, f"{v:.0f}", va="center", fontsize=9, color=C_INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("favorability (0-100, higher = better usability)")
    ax.set_title("SUS per item (normalized so higher = better)")
    ax.invert_yaxis()
    _style_axis(ax)
    ax.grid(axis="x", color="#e6e3e3", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "sus_items.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_usefulness(use_means, outdir: Path) -> None:
    labels = [USE_LABELS[c] for c in USE_CODES]
    values = [use_means.get(c, np.nan) for c in USE_CODES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color=C_GREEN, edgecolor=C_INK, linewidth=0.6)
    for y, v in enumerate(values):
        if not np.isnan(v):
            ax.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=9, color=C_INK)
    ax.set_xlim(1, 5.4)
    ax.set_xlabel("mean rating (1-5)")
    ax.set_title("Perceived usefulness (mean per item)")
    ax.invert_yaxis()
    _style_axis(ax)
    ax.grid(axis="x", color="#e6e3e3", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "usefulness_items.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_tasks(diff_means, success_rates, outdir: Path) -> None:
    labels = [TASK_LABELS[t] for t in TASKS]
    diffs = [diff_means.get(t, np.nan) for t in TASKS]
    succ = [success_rates.get(t, np.nan) * 100 for t in TASKS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = [C_RED if d >= 3.5 else C_ORANGE if d >= 2.5 else C_GREEN for d in diffs]
    ax1.bar(range(len(labels)), diffs, color=colors, edgecolor=C_INK, linewidth=0.6)
    for x, v in enumerate(diffs):
        ax1.text(x, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=C_INK)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylim(1, 5.4)
    ax1.set_ylabel("mean difficulty (1 easy - 5 hard)")
    ax1.set_title("Task difficulty")
    _style_axis(ax1)

    ax2.bar(range(len(labels)), succ, color=C_BLUE, edgecolor=C_INK, linewidth=0.6)
    for x, v in enumerate(succ):
        ax2.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=C_INK)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("success rate (%)")
    ax2.set_title("Task success (Selesai=1, Sebagian=0.5)")
    _style_axis(ax2)

    fig.tight_layout()
    fig.savefig(outdir / "task_metrics.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_background(df: pd.DataFrame, outdir: Path) -> None:
    ml = _numeric(df, "ml_fam").dropna()
    dash = _numeric(df, "dash_fam").dropna()
    device = _resolve(df, "device")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, data, title, color in [
        (axes[0], ml, "ML familiarity (1-5)", C_BLUE),
        (axes[1], dash, "Dashboard familiarity (1-5)", C_GREEN),
    ]:
        ax.hist(data, bins=np.arange(0.5, 6.5, 1), color=color, edgecolor=C_INK, linewidth=0.6)
        ax.set_xticks(range(1, 6))
        ax.set_title(title)
        ax.set_ylabel("participants")
        _style_axis(ax)

    if device is not None:
        counts = device.astype(str).value_counts()
        axes[2].bar(counts.index, counts.values, color=C_ORANGE, edgecolor=C_INK, linewidth=0.6)
        axes[2].set_title("Device used")
        axes[2].set_ylabel("participants")
        axes[2].tick_params(axis="x", labelrotation=15)
    _style_axis(axes[2])

    fig.suptitle("Participant background", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "background_summary.png", dpi=150, facecolor="white")
    plt.close(fig)


def write_summary(df, pids, sus, use_means, diff_means, success_rates, path: Path) -> None:
    n = len(df)
    lines = ["# Hasil Analisis Pengujian Pengguna (ringkasan otomatis)", ""]
    lines.append(f"Jumlah responden: **{n}**")
    lines.append("")

    if not sus.empty:
        mean = float(np.nanmean(sus.values))
        lines += [
            "## Usability — SUS",
            f"- **Rata-rata SUS: {mean:.1f} / 100 → {sus_grade(mean)}** "
            "(rujukan rata-rata industri ≈ 68).",
            "",
            "| Peserta | SUS |",
            "|---|---|",
        ]
        for pid, v in zip(pids, sus.values):
            lines.append(f"| {pid} | {v:.0f} |")
        lines.append("")

    if use_means:
        overall = float(np.nanmean(list(use_means.values())))
        lines += [
            "## Usefulness (1-5)",
            f"- **Rata-rata kegunaan keseluruhan: {overall:.2f} / 5 "
            f"({overall / 5 * 100:.0f}%).**",
            "",
            "| Item | Rata-rata |",
            "|---|---|",
        ]
        for c in USE_CODES:
            if c in use_means:
                lines.append(f"| {USE_LABELS[c]} | {use_means[c]:.2f} |")
        lines.append("")

    lines += [
        "## Tugas — kesulitan & keberhasilan",
        "",
        "| Tugas | Kesulitan (1-5) | Keberhasilan |",
        "|---|---|---|",
    ]
    for t in TASKS:
        d = diff_means.get(t, float("nan"))
        s = success_rates.get(t, float("nan"))
        lines.append(f"| {TASK_LABELS[t]} | {d:.1f} | {s * 100:.0f}% |")
    lines += [
        "",
        "Grafik: `reports/figures/user_testing/` "
        "(sus_per_participant, usefulness_items, task_metrics, background_summary).",
        "",
        "> Dihasilkan oleh `scripts/analyze_user_testing.py`. Tempelkan tabel + grafik ke "
        "`template-hasil-analisis.md` lalu ke PPT.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_qualitative(df, pids, path: Path) -> None:
    lines = ["# Umpan Balik Kualitatif (per pertanyaan)", "",
             "Untuk *thematic coding*: kelompokkan jawaban serupa, lalu rangkum temanya.", ""]
    for code in OPEN_CODES:
        series = _resolve(df, code)
        lines.append(f"## {OPEN_LABELS[code]}")
        if series is None:
            lines += ["_(kolom tidak ditemukan)_", ""]
            continue
        for pid, val in zip(pids, series.tolist()):
            text = str(val).strip()
            if text and text.lower() != "nan":
                lines.append(f"- **{pid}:** {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="docs/user-testing/responses_template.csv")
    parser.add_argument("--outdir", default="reports/figures/user_testing")
    parser.add_argument("--summary", default="docs/user-testing/analysis_summary.md")
    parser.add_argument("--qual", default="docs/user-testing/qualitative_responses.md")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pid_series = _resolve(df, "pid")
    pids = (
        pid_series.astype(str).tolist()
        if pid_series is not None
        else [f"P{i + 1}" for i in range(len(df))]
    )

    sus = compute_sus(df)
    use_means = {
        c: float(_numeric(df, c).mean()) for c in USE_CODES if not _numeric(df, c).empty
    }
    diff_means = {
        t: float(_numeric(df, f"{t}_diff").mean())
        for t in TASKS
        if not _numeric(df, f"{t}_diff").empty
    }
    success_rates = {}
    for t in TASKS:
        s = _resolve(df, f"{t}_status")
        if s is not None:
            mapped = s.astype(str).str.strip().str.lower().map(SUCCESS_MAP)
            success_rates[t] = float(mapped.mean())

    if not sus.empty:
        plot_sus(pids, sus, outdir)
        plot_sus_items(df, outdir)
    if use_means:
        plot_usefulness(use_means, outdir)
    if diff_means and success_rates:
        plot_tasks(diff_means, success_rates, outdir)
    plot_background(df, outdir)

    write_summary(df, pids, sus, use_means, diff_means, success_rates, Path(args.summary))
    write_qualitative(df, pids, Path(args.qual))

    # console recap
    print("=== USER TESTING ANALYSIS ===")
    print(f"Responses: {len(df)}")
    if not sus.empty:
        m = float(np.nanmean(sus.values))
        print(f"SUS mean: {m:.1f} -> {sus_grade(m)}")
    if use_means:
        print(f"Usefulness mean: {np.nanmean(list(use_means.values())):.2f} / 5")
    for t in TASKS:
        if t in diff_means:
            sr = success_rates.get(t, float("nan")) * 100
            print(f"  {TASK_LABELS[t]:<28} diff {diff_means[t]:.1f}  success {sr:.0f}%")
    print(f"Charts -> {outdir}/")
    print(f"Summary -> {args.summary}")
    print(f"Qualitative -> {args.qual}")


if __name__ == "__main__":
    main()
