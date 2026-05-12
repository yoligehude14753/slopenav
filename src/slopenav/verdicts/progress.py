"""VerdictProgress — 计算相邻两次迭代的 verdict 级别变化。

输入：两次迭代的 verdict 快照列表（duck typing，只需要 question + is_positive）。
输出：VerdictProgress（翻转情况、稳定性、持续失败）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from slopenav.domain.models import VerdictProgress


def compute_verdict_progress(
    prev_verdicts: List[Any],
    curr_verdicts: List[Any],
    verdict_history: Dict[int, List[Any]],
    persistent_failure_threshold: int = 3,
) -> VerdictProgress:
    """计算两次迭代之间的 verdict 变化。

    Args:
        prev_verdicts: 上一迭代的 verdict 列表（需有 question, is_positive 属性或键）。
        curr_verdicts: 当前迭代的 verdict 列表。
        verdict_history: {iteration: verdicts} 全量历史，用于计算持续失败。
        persistent_failure_threshold: 连续失败几次算"持续失败"。

    Returns:
        VerdictProgress
    """

    def _get(v: Any, key: str, default=None):
        if isinstance(v, dict):
            return v.get(key, default)
        return getattr(v, key, default)

    prev_map = {
        _get(v, "question", ""): _get(v, "is_positive", True) for v in prev_verdicts
    }
    curr_map = {
        _get(v, "question", ""): _get(v, "is_positive", True) for v in curr_verdicts
    }

    common_qs = set(prev_map.keys()) & set(curr_map.keys())
    flipped_pos = [q for q in common_qs if not prev_map[q] and curr_map[q]]
    flipped_neg = [q for q in common_qs if prev_map[q] and not curr_map[q]]

    stable_count = sum(1 for q in common_qs if prev_map[q] == curr_map[q])
    stability = stable_count / len(common_qs) if common_qs else 0.5

    persistent = _find_persistent_failures(
        verdict_history, persistent_failure_threshold
    )

    return VerdictProgress(
        flipped_positive=flipped_pos,
        flipped_negative=flipped_neg,
        persistent_failures=persistent,
        stability_score=stability,
        net_progress=len(flipped_pos) - len(flipped_neg),
    )


def _find_persistent_failures(
    verdict_history: Dict[int, List[Any]],
    threshold: int,
) -> List[Dict]:
    if len(verdict_history) < 2:
        return []

    sorted_iters = sorted(verdict_history.keys())
    check_window = sorted_iters[-min(threshold, len(sorted_iters)) :]

    def _get(v: Any, key: str, default=None):
        if isinstance(v, dict):
            return v.get(key, default)
        return getattr(v, key, default)

    failure_counts: Dict[str, int] = {}
    failure_reasons: Dict[str, str] = {}
    failure_cats: Dict[str, str] = {}

    for it in check_window:
        for v in verdict_history[it]:
            q = _get(v, "question", "")
            if not _get(v, "is_positive", True):
                failure_counts[q] = failure_counts.get(q, 0) + 1
                failure_reasons[q] = _get(v, "reason", "")
                failure_cats[q] = _get(v, "category", "")

    real_threshold = min(threshold, len(check_window))
    return [
        {
            "question": q,
            "count": c,
            "reason": failure_reasons.get(q, ""),
            "category": failure_cats.get(q, ""),
        }
        for q, c in failure_counts.items()
        if c >= real_threshold
    ]
