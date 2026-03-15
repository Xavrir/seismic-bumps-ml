from src.models.risk_levels import to_danger_flag, to_risk_level, to_risk_score


def test_to_risk_score_clamps_range():
    assert to_risk_score(-0.2) == 0
    assert to_risk_score(0.0) == 0
    assert to_risk_score(0.555) == 56
    assert to_risk_score(1.5) == 100


def test_to_danger_flag_uses_threshold_boundary():
    assert to_danger_flag(0.39, threshold=0.40) == 0
    assert to_danger_flag(0.40, threshold=0.40) == 1
    assert to_danger_flag(0.95, threshold=0.40) == 1


def test_to_risk_level_mapping():
    threshold = 0.55
    watch_floor = 0.30
    assert to_risk_level(0.10, threshold=threshold, watch_floor=watch_floor) == "low"
    assert to_risk_level(0.30, threshold=threshold, watch_floor=watch_floor) == "watch"
    assert to_risk_level(0.54, threshold=threshold, watch_floor=watch_floor) == "watch"
    assert (
        to_risk_level(0.55, threshold=threshold, watch_floor=watch_floor) == "dangerous"
    )
