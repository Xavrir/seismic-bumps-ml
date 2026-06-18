"""Risk-level mapping utilities for hazardous shift prediction."""

from __future__ import annotations

from typing import Literal

RiskLevel = Literal["low", "watch", "dangerous"]

def to_risk_score(probability: float) -> int:
    p = max(0.0, min(1.0, float(probability)))
    return int(round(p * 100))

def to_danger_flag(probability: float, threshold: float) -> int:
    return int(float(probability) >= float(threshold))

def to_risk_level(
    probability: float,
    threshold: float,
    watch_floor: float = 0.30,
) -> RiskLevel:
    """Map a calibrated hazard probability to a formal risk band.

    Bands (with calibrated probabilities, these read as real hazard frequencies):
    - ``dangerous``: ``p >= threshold`` — the cost-policy operating point; treat
      as an actionable hazard alert.
    - ``watch``: ``watch_floor <= p < threshold`` — elevated but sub-threshold;
      heightened monitoring, not yet an alert.
    - ``low``: ``p < watch_floor`` — baseline conditions.

    ``threshold`` comes from the frozen policy (F2- or cost-derived); the gap
    between ``watch_floor`` and ``threshold`` sets the expected alert volume.
    """
    p = float(probability)
    t = float(threshold)
    wf = float(watch_floor)
    if p >= t:
        return "dangerous"
    if p >= wf:
        return "watch"
    return "low"
