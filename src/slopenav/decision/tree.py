"""DecisionTree — 9 条规则的纯函数决策树。

所有参数显式传入，无状态，无副作用，可完全单元测试。
决策树由 SlopeNav（有状态外壳）调用，传入已计算好的中间值。
"""

from __future__ import annotations

from typing import Optional

from slopenav.domain.models import Decision, VerdictProgress

# ── 常量（论文 Table 1）────────────────────────────────────────────────────────

EXCELLENT_CEILING = 0.88  # 直接交付的优秀分数线
HIGH_SLOPE_LINEAR = 0.05  # 线性斜率"显著上升"阈值
HIGH_SLOPE_EMA = 0.03  # EMA 斜率"显著上升"阈值
VERDICT_STABILITY_DELIVER = 0.85  # verdict 稳定度到此 → 可交付
GOOD_ENOUGH_THRESHOLD = 0.85  # 足够好 + 斜率平坦 → 交付
PATIENCE_EXHAUSTED_ITERS = 10  # 超过此轮数后强制交付


def decide(
    n_evals: int,
    current_score: float,
    best_score: float,
    linear_slope: float,
    ema_slope: float,
    effective_high_slope: float,
    min_threshold: float,
    pivot_count: int,
    max_pivots: int,
    require_min_evals: int,
    vp: Optional[VerdictProgress],
) -> Decision:
    """纯函数决策。返回 Decision(action, reason, ...)。

    规则按优先级从高到低排列（Rule 1 最高，Rule 9 最低）。

    Args:
        n_evals: 已记录的评分点数量。
        current_score: 最新一次分数。
        best_score: 历史最高分。
        linear_slope: 当前线性回归斜率。
        ema_slope: 当前 EMA 末端斜率。
        effective_high_slope: 自适应的"显著斜率"阈值（考虑问题数量）。
        min_threshold: 合格线（默认 0.80）。
        pivot_count: 已执行的 pivot 次数。
        max_pivots: 最大允许 pivot 次数。
        require_min_evals: 最少需要几个数据点才开始决策。
        vp: verdict 级别进展（None 表示数据不足）。
    """

    def _d(action: str, reason: str) -> Decision:
        return Decision(
            action=action,
            reason=reason,
            slope_linear=linear_slope,
            slope_ema=ema_slope,
            current_score=current_score,
            best_score=best_score,
            verdict_progress=vp,
        )

    # Rule 0: 数据不足
    if n_evals < max(2, require_min_evals):
        if current_score >= EXCELLENT_CEILING:
            return _d("deliver", "first_eval_excellent")
        if require_min_evals <= 1 and current_score >= min_threshold:
            return _d("deliver", "first_eval_above_threshold")
        return _d("continue", "need_slope_data")

    # Rule 1: 优秀分数 + 稳定 verdict → 立即交付
    if current_score >= EXCELLENT_CEILING:
        if vp and vp.stability_score >= 0.7:
            return _d("deliver", "excellent_score_stable")
        if n_evals >= 3:
            return _d("deliver", "excellent_score")
        return _d("continue", "excellent_but_unstable_verifying")

    # Rule 2: Verdict 回退保护（近期丢失的 verdict > 获得的）
    if vp and vp.net_progress < -1 and current_score < min_threshold:
        return _d("continue", "verdict_regression_detected")

    # Rule 3: 足够好 + 斜率平坦 → 交付
    if (
        current_score >= GOOD_ENOUGH_THRESHOLD
        and linear_slope < 0.05
        and ema_slope < 0.03
    ):
        return _d("deliver", "good_enough_score")

    # Rule 4: 任一斜率显著上升 → 继续
    if linear_slope > effective_high_slope or ema_slope > HIGH_SLOPE_EMA:
        return _d("continue", "high_slope_improving")

    # Rule 5: 超过合格线 + 斜率平坦 → 交付
    if current_score >= min_threshold and linear_slope <= effective_high_slope:
        if vp and vp.stability_score >= VERDICT_STABILITY_DELIVER:
            return _d("deliver", "above_threshold_stable_verdicts")
        if vp is None or vp.net_progress >= 0:
            return _d("deliver", "above_threshold_slope_flat")

    # Rule 6: Verdict 级别停滞（同样的问题持续失败 ≥3 轮）
    if vp and len(vp.persistent_failures) >= 3 and n_evals >= 4:
        if current_score >= min_threshold * 0.90:
            return _d("deliver", "persistent_failures_near_threshold")
        if pivot_count >= max_pivots:
            return _d("deliver", "persistent_failures_pivots_exhausted")
        return _d("pivot", "persistent_failures_stagnant")

    # Rule 7: 超过耐心上限 → 交付最优
    if n_evals >= PATIENCE_EXHAUSTED_ITERS and best_score >= min_threshold * 0.90:
        return _d("deliver", "patience_exhausted_deliver_best")

    # Rule 8: 双斜率均非正 → 诊断停滞（调用者处理诊断）
    if linear_slope <= 0.0 and ema_slope <= 0.0:
        if pivot_count >= max_pivots:
            return _d("deliver", "max_pivots_exhausted")
        return _d("pivot", "stagnant_or_declining")

    # Rule 9: 弱正斜率但长期在合格线以下 → 耗尽耐心
    return _d("continue", "weak_positive_slope")
