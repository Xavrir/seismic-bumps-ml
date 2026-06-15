"""Shared dark "console" matplotlib theme for figures embedded in the Streamlit app.

Calling :func:`apply_console_theme` switches matplotlib to the Seismic Risk Console
palette (dark surfaces, light text) so generated figures match the dark UI instead of
clashing as light-on-dark images. It only mutates ``rcParams`` — it never touches data,
the model, or any metric.

The palette mirrors the CSS tokens in ``streamlit_app.py::inject_styles``.
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

PALETTE = {
    "bg": "#201d1d",
    "surface": "#262323",
    "text": "#fdfcfc",
    "muted": "#b8b3b3",
    "faint": "#8c8787",
    "grid": "#3a3636",
    "blue": "#2f93ff",
    "green": "#30d158",
    "orange": "#ff9f0a",
    "red": "#ff453a",
}


def apply_console_theme() -> None:
    """Set matplotlib rcParams to the dark console palette (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["faint"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "legend.facecolor": PALETTE["surface"],
            "legend.edgecolor": PALETTE["faint"],
            "legend.labelcolor": PALETTE["text"],
            "axes.prop_cycle": cycler(
                color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
            ),
            "font.family": "monospace",
        }
    )
