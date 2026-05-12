"""slopenav 领域模型 — 纯数据类，零外部依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VerdictSnapshot:
    """单次迭代的 verdict 快照（由 QAGEvaluator 或任意评分器产出）。"""

    question: str
    is_positive: bool
    category: str = ""
    reason: str = ""


@dataclass
class VerdictProgress:
    """相邻两次迭代之间的 verdict 级别变化。"""

    flipped_positive: List[str] = field(default_factory=list)  # fail→pass 问题
    flipped_negative: List[str] = field(default_factory=list)  # pass→fail 问题
    persistent_failures: List[Dict] = field(
        default_factory=list
    )  # 连续失败 N+ 次的问题
    stability_score: float = 0.0  # 0=混乱, 1=完全稳定
    net_progress: int = 0  # flipped_positive - flipped_negative


@dataclass
class SlopeResult:
    """双斜率计算结果。"""

    linear: float = 0.0  # 线性回归斜率
    ema: float = 0.0  # EMA 差分斜率


@dataclass
class StagnationDiagnosis:
    """停滞诊断结果。"""

    cause: str = "unclear"  # "capability_limit" | "eval_blind_spot" | "unclear"
    confidence: float = 0.0
    detail: str = ""


@dataclass
class Decision:
    """SlopeNav.step() 的输出 — 单步决策。"""

    action: str  # "continue" | "pivot" | "deliver"
    reason: str  # 机器可读的决策原因标签
    slope_linear: float = 0.0
    slope_ema: float = 0.0
    current_score: float = 0.0
    best_score: float = 0.0
    verdict_progress: Optional[VerdictProgress] = None
    stagnation: Optional[StagnationDiagnosis] = None
    iteration: int = 0
