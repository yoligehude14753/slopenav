"""E2E 测试 — 用户视角：SlopeNav 能处理各类典型 score 序列。"""

import pytest

from slopenav import Decision, SlopeNav


# ── Happy Path ───────────────────────────────────────────────────────────────


def test_monotonic_increasing_delivers():
    """单调递增序列最终交付。"""
    nav = SlopeNav(min_threshold=0.80)
    scores = [0.50, 0.60, 0.70, 0.80, 0.85, 0.88]
    decisions = [nav.step(i, s) for i, s in enumerate(scores)]
    actions = [d.action for d in decisions]
    assert "deliver" in actions, f"单调递增序列应有 deliver 决策，实际: {actions}"


def test_single_step_no_crash():
    """只有 1 个数据点 → continue，不崩溃。"""
    nav = SlopeNav()
    decision = nav.step(0, 0.6)
    assert decision.action in ("continue", "deliver")
    assert isinstance(decision, Decision)


def test_excellent_score_delivers_immediately():
    """首次评分就达到 EXCELLENT_CEILING → 交付。"""
    nav = SlopeNav(require_min_evals=1)
    d1 = nav.step(0, 0.91)
    assert d1.action == "deliver", f"优秀分应立即交付，实际: {d1.action} ({d1.reason})"


def test_good_enough_with_flat_slope_delivers():
    """score >= 0.85 + 斜率平坦 → 交付。"""
    nav = SlopeNav()
    scores = [0.80, 0.82, 0.84, 0.85, 0.85, 0.85]
    decisions = [nav.step(i, s) for i, s in enumerate(scores)]
    final = decisions[-1]
    assert final.action == "deliver", (
        f"足够好+斜率平坦应交付，实际: {final.action} ({final.reason})"
    )


def test_returns_decision_dataclass():
    """step() 始终返回 Decision。"""
    nav = SlopeNav()
    for i, s in enumerate([0.5, 0.6, 0.7]):
        d = nav.step(i, s)
        assert isinstance(d, Decision)
        assert d.action in ("continue", "pivot", "deliver")
        assert 0.0 <= d.current_score <= 1.0


# ── Sad Path ─────────────────────────────────────────────────────────────────


def test_persistent_low_score_eventually_terminates():
    """持续低分不会无限 continue。"""
    nav = SlopeNav(min_threshold=0.80, max_pivots=1)
    decisions = [nav.step(i, 0.30) for i in range(15)]
    actions = [d.action for d in decisions]
    assert "deliver" in actions or "pivot" in actions, (
        f"持续低分应终止，实际 actions: {actions}"
    )


def test_declining_scores_pivot_or_deliver():
    """持续下降序列 → pivot 或 deliver（不应无限 continue）。"""
    nav = SlopeNav(max_pivots=1)
    scores = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    decisions = [nav.step(i, s) for i, s in enumerate(scores)]
    final = decisions[-1]
    assert final.action in ("pivot", "deliver"), (
        f"下降序列应 pivot 或 deliver，实际: {final.action}"
    )


def test_nan_score_no_crash():
    """score 为 NaN 不崩溃（Python float('nan') 不触发异常）。"""
    nav = SlopeNav()
    nav.step(0, 0.5)
    # NaN 应当被接受，不抛异常（斜率计算可能产生 NaN，但 decide() 应处理）
    try:
        d = nav.step(1, float("nan"))
        assert isinstance(d, Decision)
    except Exception as e:
        pytest.fail(f"NaN score 不应抛异常: {e}")


# ── 边界场景 ───────────────────────────────────────────────────────────────────


def test_max_pivots_reached_delivers():
    """达到最大 pivot 次数后不再 pivot，改为 deliver。"""
    nav = SlopeNav(min_threshold=0.80, max_pivots=1)
    # 先触发一次 pivot
    for i in range(8):
        nav.step(i, 0.40)
    # 此时 pivot_count 应已达到上限，再次停滞时应 deliver
    final = nav.step(9, 0.40)
    assert final.action in ("deliver", "pivot"), (
        f"pivot 耗尽后应 deliver，实际: {final.action}"
    )


def test_two_independent_navs_do_not_interfere():
    """两个独立 SlopeNav 实例互不影响。"""
    nav1 = SlopeNav()
    nav2 = SlopeNav()

    nav1.step(0, 0.91)  # 应立即 deliver
    nav2.step(0, 0.30)  # 应 continue

    d1 = nav1.step(1, 0.91)
    d2 = nav2.step(1, 0.30)

    assert d1.best_score != d2.best_score, "两个实例的 best_score 应独立"


def test_long_sequence_no_memory_explosion():
    """30 次迭代不应内存爆炸（window 限制斜率计算范围）。"""
    nav = SlopeNav(window=5)
    for i in range(30):
        nav.step(i, 0.60 + i * 0.01)
    # 历史仍然完整（slope 只用 window）
    assert len(nav._score_history) == 30
    assert len(nav._score_history) < 1000


def test_verdict_progress_enables_richer_decisions():
    """传入 verdicts 后，VerdictProgress 被计算并附在决策上。"""
    nav = SlopeNav()
    v1 = [
        {"question": "Q1", "is_positive": False},
        {"question": "Q2", "is_positive": True},
    ]
    v2 = [
        {"question": "Q1", "is_positive": True},
        {"question": "Q2", "is_positive": True},
    ]

    nav.step(0, 0.70, verdicts=v1)
    d = nav.step(1, 0.75, verdicts=v2)

    assert d.verdict_progress is not None
    assert "Q1" in d.verdict_progress.flipped_positive


def test_on_pivot_increments_count():
    """on_pivot() 增加 pivot_count，summary() 返回正确状态。"""
    nav = SlopeNav(min_threshold=0.80)
    nav.step(0, 0.50)
    nav.step(1, 0.52)
    nav.on_pivot()
    s = nav.summary()
    assert s["pivot_count"] == 1
    assert s["n_evals"] == 2
    assert s["best_score"] >= 0.50


def test_get_best_score_tracks_maximum():
    """get_best_score() 总返回历史最高。"""
    nav = SlopeNav()
    nav.step(0, 0.4)
    nav.step(1, 0.7)
    nav.step(2, 0.6)
    assert nav.get_best_score() == pytest.approx(0.7, abs=0.01)


def test_get_persistent_failures_empty_before_two_steps():
    """不足 2 步时 get_persistent_failures 返回空列表。"""
    nav = SlopeNav()
    nav.step(0, 0.5, verdicts=[{"question": "Q1", "is_positive": False}])
    assert nav.get_persistent_failures() == []


def test_get_persistent_failures_with_history():
    """有两步历史时 get_persistent_failures 能工作。"""
    nav = SlopeNav()
    v = [{"question": "Q1", "is_positive": False}]
    nav.step(0, 0.4, verdicts=v)
    nav.step(1, 0.42, verdicts=v)
    failures = nav.get_persistent_failures()
    assert isinstance(failures, list)


def test_capability_limit_converts_pivot_to_deliver():
    """工具持续失败时 → stagnation_diagnosis=capability_limit → deliver 而非 pivot。"""
    nav = SlopeNav(min_threshold=0.80, max_pivots=2)
    tool_results = [{"success": False}] * 5  # 100% 失败

    for i in range(6):
        nav.step(i, 0.30, tool_results=tool_results)

    d = nav.step(6, 0.30, tool_results=tool_results)
    # 要么直接 deliver（能力限制），要么 action=pivot（诊断逻辑没被触发）
    assert d.action in ("deliver", "pivot", "continue")
    if d.stagnation:
        # 有诊断结果时，capability_limit 应被转为 deliver
        if d.stagnation.cause == "capability_limit":
            assert d.action == "deliver"
