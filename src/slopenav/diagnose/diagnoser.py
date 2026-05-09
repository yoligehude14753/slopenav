"""StagnationDiagnoser — 区分停滞原因。

停滞有两种根本原因：
  capability_limit  — agent 能力不足，工具持续失败，再 pivot 也无济于事
  eval_blind_spot   — 评分器无法识别已产出的文件/结果，agent 其实已完成
  unclear           — 信号混杂，无法确定

用途：当 DecisionTree 判断为 "pivot" 时，先调用 diagnose()。
  - capability_limit → 改为 "deliver"（交付已有最佳结果）
  - eval_blind_spot  → 改为 "deliver"（评分器有盲区，强制交付）
  - unclear          → 正常 pivot
"""

from __future__ import annotations

from typing import Any, Dict, List

from slopenav.domain.models import StagnationDiagnosis


def diagnose_stagnation(
    score_history: List[float],
    tool_results: List[Any],
    min_threshold: float = 0.80,
) -> StagnationDiagnosis:
    """分析停滞原因。

    Args:
        score_history: 最近若干轮的分数序列（时间顺序）。
        tool_results: 最近迭代的工具调用结果列表（duck typing）。
        min_threshold: 质量合格线。

    Returns:
        StagnationDiagnosis
    """
    if len(score_history) < 3:
        return StagnationDiagnosis(cause="unclear", confidence=0.3, detail="history too short")

    recent = score_history[-3:]
    avg_recent = sum(recent) / len(recent)

    def _get(r: Any, key: str, default=None):
        if isinstance(r, dict):
            return r.get(key, default)
        return getattr(r, key, default)

    file_producers = [
        r for r in tool_results
        if _get(r, "success") and (
            _get(r, "path") or _get(r, "file_url") or
            _get(r, "files") or _get(r, "filename")
        )
    ]
    tool_failures = [r for r in tool_results if _get(r, "success") is False]
    total_tools = len([r for r in tool_results if isinstance(r, dict) or hasattr(r, "__dict__")])

    # 有文件产出但分数卡在低位 → 评分器盲区
    if file_producers and avg_recent < min_threshold * 0.8:
        return StagnationDiagnosis(
            cause="eval_blind_spot",
            confidence=0.8,
            detail=f"files={len(file_producers)}, avg_score={avg_recent:.3f}",
        )

    # 工具失败率 >70% → 能力限制
    if total_tools > 0 and len(tool_failures) > total_tools * 0.7:
        return StagnationDiagnosis(
            cause="capability_limit",
            confidence=0.85,
            detail=f"failures={len(tool_failures)}/{total_tools}",
        )

    # 分数极低且无成功工具
    if len(score_history) >= 5:
        recent_5 = score_history[-5:]
        score_range = max(recent_5) - min(recent_5)
        successful = [r for r in tool_results if _get(r, "success")]
        if score_range < 0.05 and avg_recent < min_threshold * 0.7 and not successful:
            return StagnationDiagnosis(
                cause="capability_limit",
                confidence=0.75,
                detail=f"flat={score_range:.3f}, avg={avg_recent:.3f}, no_tools",
            )

    return StagnationDiagnosis(cause="unclear", confidence=0.4, detail="mixed signals")
