"""LinearEstimator — 最小二乘线性回归斜率估计。

纯 Python 实现（无 numpy 依赖），可选 numpy 加速。
输入：最近 window 个 (iteration, score) 点。
输出：斜率 k（每迭代 score 变化量）。
"""

from __future__ import annotations

from typing import List, Tuple


def compute_linear_slope(points: List[Tuple[int, float]]) -> float:
    """用普通最小二乘法计算斜率。

    Args:
        points: [(iteration_index, score), ...] 按时间顺序排列。

    Returns:
        斜率 k（正=上升，负=下降，0=水平）。
        点数 < 2 时返回 0.0。
    """
    n = len(points)
    if n < 2:
        return 0.0

    # 使用序号 0..n-1 而非原始 iteration，避免迭代号不连续导致偏差
    xs = list(range(n))
    ys = [s for _, s in points]

    x_sum = sum(xs)
    y_sum = sum(ys)
    xy_sum = sum(x * y for x, y in zip(xs, ys))
    x2_sum = sum(x * x for x in xs)

    denom = n * x2_sum - x_sum * x_sum
    if denom == 0:
        return 0.0

    return (n * xy_sum - x_sum * y_sum) / denom
