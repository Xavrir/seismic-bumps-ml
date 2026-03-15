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
    p = float(probability)
    t = float(threshold)
    wf = float(watch_floor)
    if p >= t:
        return "dangerous"
    if p >= wf:
        return "watch"
    return "low"
