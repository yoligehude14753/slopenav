"""EMAEstimator — 指数移动平均斜率估计。

用 EMA 序列的最后两个值之差作为"当前动量"。
比线性回归对最新变化更敏感，适合检测突然上升或下降。
"""

from __future__ import annotations

from typing import List, Tuple


def compute_ema_slope(
    points: List[Tuple[int, float]],
    alpha: float = 0.4,
) -> float:
    """计算 EMA 序列末端斜率（最后两点之差）。

    Args:
        points: [(iteration_index, score), ...] 按时间顺序排列。
        alpha: EMA 平滑系数，越大对最新数据越敏感（默认 0.4）。

    Returns:
        EMA 末端斜率（正=近期上升，负=近期下降）。
        点数 < 2 时返回 0.0。
    """
    if len(points) < 2:
        return 0.0

    scores = [s for _, s in points]
    ema = scores[0]
    ema_vals: List[float] = [ema]
    for s in scores[1:]:
        ema = alpha * s + (1 - alpha) * ema
        ema_vals.append(ema)

    return ema_vals[-1] - ema_vals[-2]
