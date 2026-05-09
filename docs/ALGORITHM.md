# slopenav · 核心算法说明

> 版本：v1.0 · 日期：2026-05-08

---

## 1. 形式化定义

### 1.1 时间序列

agent 在 iteration $i \in \mathbb{N}$ 产生质量分 $s_i \in [0,1]$ 和 verdict 集合 $V_i = \{(q_j, b_{ij})\}$（$b_{ij} \in \{0, 1\}$ 表示 verdict $j$ 是否通过）。

历史序列：
$$
S_n = \big((i, s_i)\big)_{i=0}^{n-1}, \quad \mathcal{V}_n = \{V_i\}_{i=0}^{n-1}
$$

### 1.2 决策器

$$
D: (S_n, \mathcal{V}_n) \to \{\text{continue, pivot, deliver}\} \times \mathcal{R}
$$

其中 $\mathcal{R}$ 是决策原因（用于日志和论文 case study）。

---

## 2. 双斜率估计

### 2.1 线性斜率（OLS over window）

取最近 $w$ 个点 $(x_k, y_k)_{k=0}^{w-1}$（$x_k = k$，$y_k = s_{n-w+k}$）：

$$
\hat{\beta}_\text{lin} = \frac{w \sum x_k y_k - \sum x_k \sum y_k}{w \sum x_k^2 - (\sum x_k)^2}
$$

### 2.2 EMA 斜率

定义 EMA 序列 $e_k = \alpha s_k + (1 - \alpha) e_{k-1}$，$e_0 = s_0$。

$$
\hat{\beta}_\text{ema} = e_n - e_{n-1}
$$

**为什么用两种**：
- 线性斜率对趋势鲁棒但延迟（看 $w$ 个点）
- EMA 斜率对最新变化敏感但有噪声
- **双斜率冗余**：任一指示"还在改进"就 continue；都说"平了"才考虑停

### 2.3 自适应高斜率阈值

QAG 类二值评分有"量化"特性：单题翻转 ≈ $\frac{1}{N_q}$（$N_q$ 是问题数）。

$$
\beta^* = \max\big(0.01,\; \min(0.05,\; \frac{0.6}{N_q})\big)
$$

意思是：单题翻转的 0.6 倍以下的斜率不算"强劲改进"。这避免了"问题多时单题翻转占比小，被误判为停滞"的问题。

---

## 3. Verdict-Level Progress

### 3.1 Flipped 集合

对相邻两轮 $V_{n-1}, V_n$：
- **Flipped Positive**：$\{q : b_{n-1, q} = 0, b_{n, q} = 1\}$（fail → pass）
- **Flipped Negative**：$\{q : b_{n-1, q} = 1, b_{n, q} = 0\}$（pass → fail）

净进展：$\Delta = |\text{F}^+| - |\text{F}^-|$

### 3.2 稳定性

$$
\text{Stability} = \frac{|\{q : b_{n-1, q} = b_{n, q}\}|}{|Q_n \cap Q_{n-1}|}
$$

接近 1 = verdict 稳定；接近 0 = 评估器自身在抖。

### 3.3 持续失败检测

对最近 $T$ 轮（$T = 3$）：
$$
\text{Persistent}_T = \{q : \forall i \in [n-T, n], b_{i,q} = 0\}
$$

这些是"无论怎么改 generator，评估器都说不通过"的问题，是 pivot 信号。

---

## 4. 决策树（核心算法）

```
function decide(S_n, V_n):
    n = |S_n|
    s_n = current score
    β_lin = compute_linear_slope(S_n)
    β_ema = compute_ema_slope(S_n)
    vp = compute_verdict_progress(V_n)
    
    # Rule 0: 数据不足
    if n < max(2, require_min_evals):
        if s_n ≥ EXCELLENT_CEILING (= 0.88):
            return Decision("deliver", "first_eval_excellent", ...)
        if require_min_evals ≤ 1 and s_n ≥ min_threshold:
            return Decision("deliver", "first_eval_above_threshold", ...)
        return Decision("continue", "need_slope_data", ...)
    
    # Rule 1: 优秀 + 稳定 → 立即交付
    if s_n ≥ EXCELLENT_CEILING:
        if vp.stability ≥ 0.7:
            return Decision("deliver", "excellent_stable", ...)
        if n ≥ 3:
            return Decision("deliver", "excellent_score", ...)
        return Decision("continue", "excellent_but_unstable", ...)
    
    # Rule 2: Verdict 回退保护
    if vp.net_progress < -1 and s_n < min_threshold:
        return Decision("continue", "verdict_regression", ...)
    
    # Rule 2.5: 良好+稳定（相邻于阈值，避免过度优化）
    if s_n ≥ 0.85 and β_lin < 0.05 and β_ema < 0.03:
        return Decision("deliver", "good_enough_score", ...)
    
    # Rule 3: 强劲斜率 → 继续
    β* = adaptive_high_slope(N_q)
    if β_lin > β* or β_ema > HIGH_SLOPE_EMA (= 0.03):
        return Decision("continue", "high_slope_improving", ...)
    
    # Rule 4: 阈值上 + 平稳 → 交付
    if s_n ≥ min_threshold and β_lin ≤ β*:
        if vp.stability ≥ VERDICT_STABILITY_DELIVER (= 0.85):
            return Decision("deliver", "above_threshold_stable", ...)
        if vp.net_progress ≥ 0:
            return Decision("deliver", "above_threshold_flat_slope", ...)
    
    # Rule 5: Verdict 持续失败
    if |vp.persistent| ≥ 3 and n ≥ 4:
        if s_n ≥ 0.90 × min_threshold:
            return Decision("deliver", "persistent_near_threshold", ...)
        if pivot_count ≥ max_pivots:
            return Decision("deliver", "persistent_pivots_exhausted", ...)
        return Decision("pivot", "persistent_failures_stagnant", ...)
    
    # Rule 6: 分数平稳（紧带宽）
    if n ≥ 5:
        recent = S_n[-5:]
        range_ = max(recent) - min(recent)
        q_step = 1 / max(N_q, 1)
        if range_ < q_step × 1.5:
            if s_n ≥ 0.95 × min_threshold:
                return Decision("deliver", "plateau_near_threshold", ...)
            if pivot_count ≥ max_pivots:
                return Decision("deliver", "plateau_pivots_exhausted", ...)
            return Decision("pivot", "plateau_below_threshold", ...)
    
    # Rule 7: 耐心耗尽（多轮无显著结果）
    if n ≥ 10 and best_score ≥ 0.90 × min_threshold:
        return Decision("deliver", "patience_exhausted", ...)
    
    # Rule 8: 负斜率 → 诊断
    if β_lin ≤ 0 and β_ema ≤ 0:
        if pivot_count ≥ max_pivots:
            return Decision("deliver", "max_pivots_negative_slope", ...)
        return Decision("pivot", "stagnant_or_declining", ...)
    
    # Rule 9: 弱正斜率，带耐心限制
    weak = sum(1 for s in S_n[-8:] if s < min_threshold)
    if weak ≥ 8:
        return Decision("deliver", "weak_slope_patience_exhausted", ...)
    return Decision("continue", "weak_positive_slope", ...)
```

**论文中的关键洞察**：决策树的 9 条规则形成一个**优先级有向图**，每条规则覆盖现有方法漏掉的一种场景。规则之间正交（同一时刻最多触发一条），可独立做消融实验（拿掉某条 rule，观察对 task success 的影响）。

---

## 5. 停滞诊断算法

```
function diagnose_stagnation(S_n, context):
    tool_results = context.tool_results
    file_producers = [r for r in tool_results if r.success and has_file(r)]
    tool_failures = [r for r in tool_results if r.success is False]
    n_total = len(tool_results)
    
    if n < 3:
        return "unclear"
    
    avg_recent = mean(S_n[-3:])
    
    # Eval blind spot：工具产文件但分还低
    if file_producers and avg_recent < min_threshold × 0.8:
        return "eval_blind_spot"
    
    # Capability limit：工具失败率高
    if n_total > 0 and len(tool_failures) > n_total × 0.7:
        return "capability_limit"
    
    # 分数平 + 完全没成功的工具
    if n ≥ 5:
        recent = S_n[-5:]
        range_ = max(recent) - min(recent)
        successful = [r for r in tool_results if r.success]
        if range_ < 0.05 and avg_recent < min_threshold × 0.7 and not successful:
            return "capability_limit"
    
    return "unclear"
```

### 诊断 → 决策耦合

| 诊断 | 决策修正 |
|------|---------|
| capability_limit | pivot 改为 deliver(best)（继续也徒劳） |
| eval_blind_spot | pivot 改为 deliver(best)（评估器问题，不是 generator 问题） |
| unclear | 维持原决策 |

这是 SlopeNav 与朴素 stopping 算法的核心差异：**当负斜率发生时，先问"为什么"再决定"怎么办"**。

---

## 6. 关键设计选择的论文叙述

### 6.1 为什么不用一个 RL agent 学习停止

- 训练成本高，外推性差
- SlopeNav 是"白盒"决策树，论文里能逐条解释每条 rule 的意图，便于读者复现和修改
- 工程上更容易调（每条 rule 的阈值可独立调）

### 6.2 为什么 verdict-level 比 score-level 更重要

实证假设（在 EXPERIMENTS.md 验证）：
- Score 只是 verdict 的**汇总**；同一个 score 可能对应不同的 verdict 模式
- 当 verdict 模式发生变化时（即使 score 没变），意味着 generator 的策略真的换了
- Verdict-level analysis 提供"高频信号"，比"低频"的 score 更早预测停滞

### 6.3 为什么需要双斜率（线性 + EMA）

- 单一线性回归对 outlier 敏感（一个噪声点会让 slope 翻转）
- 单一 EMA 平滑过头，落后真实变化
- 双斜率"或"逻辑：任一指示改进就 continue，提供冗余防错

---

## 7. 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| `record()` | O(1) | O(1) 增量 |
| `compute_linear_slope` | O(w) | O(w) |
| `compute_ema_slope` | O(w) | O(w) |
| `compute_verdict_progress` | O(\|V\|) | O(\|V\|) |
| `decide()` | O(w + \|V\|) | O(1) 临时 |

实测目标：`step()` P99 ≤ 1ms（Python 标准 numpy）。
