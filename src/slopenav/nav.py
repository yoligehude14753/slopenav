"""SlopeNav — 有状态外壳，封装双斜率迭代决策。

对外接口极简：
    nav = SlopeNav()
    decision = nav.step(iteration=0, score=0.72, verdicts=[...])

内部依赖：
    slope/linear.py  → compute_linear_slope
    slope/ema.py     → compute_ema_slope
    verdicts/progress.py → compute_verdict_progress
    decision/tree.py → decide (纯函数)
    diagnose/diagnoser.py → diagnose_stagnation（pivot 时触发）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from slopenav.decision.tree import (
    EXCELLENT_CEILING,
    HIGH_SLOPE_LINEAR,
    decide,
)
from slopenav.diagnose.diagnoser import diagnose_stagnation
from slopenav.domain.models import Decision, VerdictProgress
from slopenav.slope.ema import compute_ema_slope
from slopenav.slope.linear import compute_linear_slope
from slopenav.verdicts.progress import compute_verdict_progress


class SlopeNav:
    """QAG-aware 双斜率迭代决策器。

    使用示例::

        from slopenav import SlopeNav, Decision

        nav = SlopeNav(min_threshold=0.80, max_pivots=2)
        for i, (score, verdicts) in enumerate(iteration_results):
            decision = nav.step(iteration=i, score=score, verdicts=verdicts)
            if decision.action == "deliver":
                deliver(best_content)
                break
            if decision.action == "pivot":
                change_strategy()

    Attributes:
        min_threshold: 合格分数线（默认 0.80）。
        max_pivots: 最大 pivot 次数（默认 2）。
        window: 斜率计算的滑动窗口大小（默认 5）。
    """

    def __init__(
        self,
        min_threshold: float = 0.80,
        max_pivots: int = 2,
        window: int = 5,
        require_min_evals: int = 1,
        ema_alpha: float = 0.4,
    ) -> None:
        self.min_threshold = min_threshold
        self.max_pivots = max_pivots
        self.window = window
        self.require_min_evals = require_min_evals
        self.ema_alpha = ema_alpha

        self._score_history: List[Tuple[int, float]] = []
        self._verdict_history: Dict[int, List[Any]] = {}
        self._best_score: float = 0.0
        self._best_iteration: int = 0
        self._pivot_count: int = 0

    # ── 主接口 ──────────────────────────────────────────────────────────────

    def step(
        self,
        iteration: int,
        score: float,
        verdicts: Optional[List[Any]] = None,
        tool_results: Optional[List[Any]] = None,
    ) -> Decision:
        """记录一次评分并返回决策。

        Args:
            iteration: 当前迭代编号（从 0 开始）。
            score: 当前评分，[0, 1]。
            verdicts: 评分器产出的 verdict 列表（可选，提供则启用 verdict 级别分析）。
            tool_results: 工具调用结果（可选，提供则在 pivot 时用于诊断停滞原因）。

        Returns:
            Decision: action ∈ {"continue", "pivot", "deliver"}
        """
        # 记录
        self._score_history.append((iteration, score))
        if verdicts is not None:
            self._verdict_history[iteration] = verdicts
        if score > self._best_score:
            self._best_score = score
            self._best_iteration = iteration

        # 计算斜率
        recent = self._score_history[-self.window:]
        linear_slope = compute_linear_slope(recent)
        ema_slope = compute_ema_slope(recent, alpha=self.ema_alpha)

        # 自适应 high_slope（问题数越多，每道题翻转的影响越小）
        effective_high_slope = self._adaptive_high_slope(linear_slope)

        # 计算 verdict 进展
        vp: Optional[VerdictProgress] = None
        if len(self._verdict_history) >= 2:
            sorted_iters = sorted(self._verdict_history.keys())
            vp = compute_verdict_progress(
                prev_verdicts=self._verdict_history[sorted_iters[-2]],
                curr_verdicts=self._verdict_history[sorted_iters[-1]],
                verdict_history=self._verdict_history,
            )

        # 纯函数决策
        decision = decide(
            n_evals=len(self._score_history),
            current_score=score,
            best_score=self._best_score,
            linear_slope=linear_slope,
            ema_slope=ema_slope,
            effective_high_slope=effective_high_slope,
            min_threshold=self.min_threshold,
            pivot_count=self._pivot_count,
            max_pivots=self.max_pivots,
            require_min_evals=self.require_min_evals,
            vp=vp,
        )
        decision.iteration = iteration

        # pivot 时执行停滞诊断
        if decision.action == "pivot":
            scores_only = [s for _, s in self._score_history]
            diagnosis = diagnose_stagnation(
                score_history=scores_only,
                tool_results=tool_results or [],
                min_threshold=self.min_threshold,
            )
            decision.stagnation = diagnosis

            # 能力限制或评分盲区 → 改为交付
            if diagnosis.cause in ("capability_limit", "eval_blind_spot"):
                decision.action = "deliver"
                decision.reason = f"stagnation_{diagnosis.cause}"
            else:
                self._pivot_count += 1

        return decision

    def on_pivot(self) -> None:
        """外部主动通知 pivot（重置 EMA，计数器 +1）。"""
        self._pivot_count += 1

    def get_best_score(self) -> float:
        return self._best_score

    def get_persistent_failures(self) -> List[Dict]:
        """获取持续未通过的检查项（用于 pivot prompt 生成）。"""
        if len(self._verdict_history) < 2:
            return []
        sorted_iters = sorted(self._verdict_history.keys())
        vp = compute_verdict_progress(
            prev_verdicts=self._verdict_history[sorted_iters[-2]],
            curr_verdicts=self._verdict_history[sorted_iters[-1]],
            verdict_history=self._verdict_history,
        )
        return vp.persistent_failures if vp else []

    def summary(self) -> Dict:
        """返回当前状态摘要（用于调试和日志）。"""
        return {
            "n_evals": len(self._score_history),
            "best_score": self._best_score,
            "pivot_count": self._pivot_count,
            "score_history": [s for _, s in self._score_history],
        }

    # ── 内部辅助 ────────────────────────────────────────────────────────────

    def _adaptive_high_slope(self, base_slope: float) -> float:
        """根据问题数量自适应调整 high_slope 阈值。"""
        if not self._verdict_history:
            return HIGH_SLOPE_LINEAR
        latest_iter = max(self._verdict_history.keys())
        n_questions = len(self._verdict_history[latest_iter])
        if n_questions > 0:
            q_step = 1.0 / n_questions
            return max(0.01, min(HIGH_SLOPE_LINEAR, q_step * 0.6))
        return HIGH_SLOPE_LINEAR
