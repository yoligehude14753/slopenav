"""单元测试 — DecisionTree 纯函数，9 条规则全覆盖。"""

from slopenav.decision.tree import decide
from slopenav.domain.models import VerdictProgress


def _vp(net=0, stability=0.9, persistent=0) -> VerdictProgress:
    return VerdictProgress(
        flipped_positive=max(0, net) * ["Q"],
        flipped_negative=max(0, -net) * ["Q"],
        persistent_failures=[{"question": f"Q{i}"} for i in range(persistent)],
        stability_score=stability,
        net_progress=net,
    )


def _decide(**kwargs) -> str:
    defaults = dict(
        n_evals=3,
        current_score=0.5,
        best_score=0.5,
        linear_slope=0.0,
        ema_slope=0.0,
        effective_high_slope=0.05,
        min_threshold=0.80,
        pivot_count=0,
        max_pivots=2,
        require_min_evals=1,
        vp=None,
    )
    defaults.update(kwargs)
    d = decide(**defaults)
    return d.action, d.reason


# ── Rule 0: 数据不足 ──────────────────────────────────────────────────────────


def test_rule0_need_more_data():
    action, reason = _decide(n_evals=1, require_min_evals=3, current_score=0.5)
    assert action == "continue"
    assert reason == "need_slope_data"


def test_rule0_excellent_on_first_eval():
    action, reason = _decide(n_evals=1, current_score=0.91, require_min_evals=1)
    assert action == "deliver"
    assert reason == "first_eval_excellent"


# ── Rule 1: 优秀分数 ──────────────────────────────────────────────────────────


def test_rule1_excellent_stable_delivers():
    action, _ = _decide(current_score=0.90, n_evals=3, vp=_vp(stability=0.9))
    assert action == "deliver"


def test_rule1_excellent_unstable_continues():
    action, reason = _decide(current_score=0.90, n_evals=2, vp=_vp(stability=0.3))
    assert action == "continue"
    assert "unstable" in reason


# ── Rule 2: Verdict 回退 ──────────────────────────────────────────────────────


def test_rule2_verdict_regression_continues():
    action, reason = _decide(
        current_score=0.70,
        vp=_vp(net=-3, stability=0.3),
    )
    assert action == "continue"
    assert "verdict_regression" in reason


# ── Rule 3: 足够好 + 平坦 ─────────────────────────────────────────────────────


def test_rule3_good_enough_flat_delivers():
    action, reason = _decide(
        current_score=0.87,
        linear_slope=0.01,
        ema_slope=0.01,
    )
    assert action == "deliver"
    assert "good_enough" in reason


# ── Rule 4: 高斜率继续 ───────────────────────────────────────────────────────


def test_rule4_high_linear_slope_continues():
    action, reason = _decide(linear_slope=0.10, effective_high_slope=0.05)
    assert action == "continue"
    assert "high_slope" in reason


def test_rule4_high_ema_slope_continues():
    action, reason = _decide(
        ema_slope=0.05, linear_slope=0.0, effective_high_slope=0.05
    )
    assert action == "continue"


# ── Rule 5: 超合格线 + 稳定 ──────────────────────────────────────────────────


def test_rule5_above_threshold_stable_delivers():
    action, reason = _decide(
        current_score=0.82,
        linear_slope=0.02,
        effective_high_slope=0.05,
        vp=_vp(stability=0.90),
    )
    assert action == "deliver"


def test_rule5_above_threshold_flat_delivers():
    action, reason = _decide(
        current_score=0.82,
        linear_slope=0.02,
        effective_high_slope=0.05,
        vp=None,
    )
    assert action == "deliver"


# ── Rule 6: Verdict 停滞 ─────────────────────────────────────────────────────


def test_rule6_persistent_failures_near_threshold_delivers():
    action, reason = _decide(
        current_score=0.75,
        n_evals=5,
        vp=_vp(stability=0.5, persistent=4),
        min_threshold=0.80,
    )
    assert action == "deliver"
    assert "persistent_failures_near_threshold" in reason


def test_rule6_persistent_failures_stagnant_pivots():
    action, reason = _decide(
        current_score=0.60,
        n_evals=5,
        vp=_vp(stability=0.5, persistent=4),
        pivot_count=0,
        max_pivots=2,
        min_threshold=0.80,
    )
    assert action == "pivot"
    assert "persistent_failures_stagnant" in reason


# ── Rule 7: 耐心耗尽 ─────────────────────────────────────────────────────────


def test_rule7_patience_exhausted_delivers():
    action, reason = _decide(
        n_evals=11,
        best_score=0.76,
        current_score=0.70,
        min_threshold=0.80,
        linear_slope=0.001,
        ema_slope=0.001,
        effective_high_slope=0.05,
    )
    assert action == "deliver"
    assert "patience_exhausted" in reason


# ── Rule 8: 负斜率 ───────────────────────────────────────────────────────────


def test_rule8_negative_slope_pivots():
    action, reason = _decide(
        linear_slope=-0.05,
        ema_slope=-0.02,
        current_score=0.60,
        pivot_count=0,
        max_pivots=2,
    )
    assert action == "pivot"
    assert "stagnant" in reason


def test_rule8_max_pivots_exhausted_delivers():
    action, reason = _decide(
        linear_slope=-0.05,
        ema_slope=-0.02,
        pivot_count=2,
        max_pivots=2,
    )
    assert action == "deliver"
    assert "max_pivots_exhausted" in reason


# ── Rule 6: persistent_failures_pivots_exhausted branch ─────────────────────


def test_rule6_persistent_failures_pivots_exhausted():
    """Rule 6: 持续失败 + pivots 耗尽 + 分数未达近阈值 → deliver。"""
    action, reason = _decide(
        current_score=0.55,  # < min_threshold * 0.90 (0.72)
        min_threshold=0.80,
        pivot_count=2,
        max_pivots=2,
        vp=_vp(persistent=3),
        n_evals=4,
    )
    assert action == "deliver"
    assert "pivots_exhausted" in reason


# ── Rule 9: 弱正斜率 ─────────────────────────────────────────────────────────


def test_rule9_weak_positive_slope_continues():
    action, reason = _decide(
        linear_slope=0.02,
        ema_slope=0.01,
        effective_high_slope=0.05,
        current_score=0.65,
        min_threshold=0.80,
    )
    assert action == "continue"
    assert "weak_positive" in reason
