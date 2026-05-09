"""slopenav — Dual-slope iteration decision algorithm for AI agent quality convergence.

快速开始::

    from slopenav import SlopeNav, Decision

    nav = SlopeNav(min_threshold=0.80, max_pivots=2)
    for iteration, score in enumerate(scores):
        decision: Decision = nav.step(iteration=iteration, score=score)
        if decision.action == "deliver":
            break
        if decision.action == "pivot":
            pass  # 切换策略
"""

from slopenav.domain.models import (
    Decision,
    SlopeResult,
    StagnationDiagnosis,
    VerdictProgress,
    VerdictSnapshot,
)
from slopenav.nav import SlopeNav

__all__ = [
    "SlopeNav",
    "Decision",
    "VerdictProgress",
    "VerdictSnapshot",
    "SlopeResult",
    "StagnationDiagnosis",
]

__version__ = "0.1.0"
