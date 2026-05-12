"""单元测试 — LinearEstimator + EMAEstimator。"""

import pytest

from slopenav.slope.ema import compute_ema_slope
from slopenav.slope.linear import compute_linear_slope


# ── LinearEstimator ───────────────────────────────────────────────────────────


def test_linear_perfect_increase():
    points = [(0, 0.0), (1, 0.1), (2, 0.2), (3, 0.3)]
    slope = compute_linear_slope(points)
    assert slope == pytest.approx(0.1, abs=1e-9)


def test_linear_perfect_decrease():
    points = [(0, 0.5), (1, 0.4), (2, 0.3)]
    slope = compute_linear_slope(points)
    assert slope == pytest.approx(-0.1, abs=1e-9)


def test_linear_flat():
    points = [(0, 0.7), (1, 0.7), (2, 0.7), (3, 0.7)]
    slope = compute_linear_slope(points)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_linear_single_point_returns_zero():
    assert compute_linear_slope([(0, 0.5)]) == 0.0


def test_linear_empty_returns_zero():
    assert compute_linear_slope([]) == 0.0


def test_linear_two_points():
    points = [(0, 0.6), (1, 0.8)]
    slope = compute_linear_slope(points)
    assert slope == pytest.approx(0.2, abs=1e-9)


# ── EMAEstimator ──────────────────────────────────────────────────────────────


def test_ema_increasing_trend_positive():
    points = [(i, 0.5 + i * 0.1) for i in range(6)]
    slope = compute_ema_slope(points)
    assert slope > 0


def test_ema_decreasing_trend_negative():
    points = [(i, 0.9 - i * 0.1) for i in range(6)]
    slope = compute_ema_slope(points)
    assert slope < 0


def test_ema_flat_near_zero():
    points = [(i, 0.7) for i in range(5)]
    slope = compute_ema_slope(points)
    assert abs(slope) < 1e-6


def test_ema_single_point_returns_zero():
    assert compute_ema_slope([(0, 0.5)]) == 0.0


def test_ema_alpha_sensitivity():
    """alpha=1.0 时完全跟随最新值，slope = 最后两点之差。"""
    points = [(0, 0.5), (1, 0.7), (2, 0.9)]
    slope = compute_ema_slope(points, alpha=1.0)
    assert slope == pytest.approx(0.2, abs=1e-9)
