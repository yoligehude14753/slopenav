# slopenav · PRD

> 版本：v1.0 · 日期：2026-05-08 · 状态：草稿待确认

---

## 6W2H 需求分析

### What — 做什么

SlopeNav 是一个**接受任意质量分序列、输出迭代决策**的算法库。

```python
nav = SlopeNav(min_threshold=0.80)
for iteration in range(max_iters):
    output = generator.run()
    score, verdicts = some_evaluator.eval(output)  # 任意打分器
    decision = nav.step(score=score, verdicts=verdicts, iteration=iteration)
    if decision.action == "deliver":
        break
    elif decision.action == "pivot":
        generator.switch_strategy()
    # else: continue
```

输出 `Decision`：
```python
Decision(
    action: Literal["continue", "pivot", "deliver"],
    reason: str,                # 决策原因（用于日志和论文 case study）
    slope_linear: float,
    slope_ema: float,
    verdict_progress: VerdictProgress,
    diagnosis: Optional[str],   # capability_limit / eval_blind_spot / unclear
)
```

### Why — 为什么做

**核心痛点**：现有 agent iteration loop 用以下三种朴素策略停：
1. **固定 max-iterations**（max=3）：浪费早收敛任务的机会，撑不到难任务
2. **绝对阈值**（score ≥ 0.8）：对评分器噪声敏感，分数抖动会假交付
3. **LLM 自我判定**："我觉得做完了" → 高假交付率

SlopeNav 用**信号处理 + verdict 级进展**做决策，比上述三者更稳健。

### Who — 谁来用

**主要用户**：
- 做 self-refining agents 的工程师（Reflexion / Self-Refine 框架的延伸）
- 研究 agent 收敛性的研究者
- 训练 RLHF 的研究者（reward 信号的 stopping rule）

**反面用户**：
- 没有"质量分时间序列"的场景（如 single-shot QA）
- 不需要迭代的任务

### Where — 在哪用

- **形式**：纯 Python 包（无 LLM 依赖），`pip install slopenav`
- **嵌入位置**：任何 generator-evaluator-loop 中，replace 原有 stopping logic
- **与 QAG-Gate 关系**：可独立使用，也可消费 QAG-Gate 输出

### When — 什么时候用

- 每次 iteration 完成 + 打分后调用一次
- 计算延迟：< 1ms（纯 numpy / 内存操作）

### Which — 哪种方案

| 方案 | 描述 | 选择 |
|------|------|------|
| A. 固定 max-iter | 简单但浪费算力 / 漏交付 | × |
| B. 单一阈值 | 对噪声敏感 | × |
| C. 用 LLM 决策 | 又一次 LLM 调用，成本高且不稳 | × |
| D. **双斜率 + verdict 进展（SlopeNav）** | 量化 + 高频信号 | ✓ |
| E. RL agent 自学停止 | 重，需要训练 | 远期可探 |

### How — 怎么做（核心流程）

```
[输入] iteration, score, verdicts, content_snapshot
   ↓
[阶段1] record(iteration, score, content, verdicts)
   ├── 更新 score_history
   ├── 更新 EMA score
   ├── 记录 best_score
   └── 持久化 verdict_history
   ↓
[阶段2] compute_slope() → linear_slope（最近 window 个点的 OLS）
[阶段3] compute_ema_slope() → ema_slope（EMA 序列的差分）
[阶段4] compute_verdict_progress() → 翻转、稳定性、持续失败
   ↓
[阶段5] 决策树：
   if 优秀 + 稳定 → deliver
   elif verdict 回退 → continue
   elif 强劲斜率 → continue
   elif 阈值上 + 平稳 → deliver
   elif verdict 持续失败 → pivot or deliver
   elif 分数平稳 → pivot or deliver
   elif 多轮无进展 → deliver(best)
   elif 负斜率 → diagnose → pivot or deliver
   else continue
   ↓
[输出] Decision
```

### How Much — 成本与收益

**开发成本**：
- 代码解耦：1 周
- 合成 + 真实数据生成：2 周
- 实验跑通 + paper draft：3 周
- 合计 ~112 人时

**运行成本**：
- 几乎零（无 LLM 调用，pure numpy）

**预期收益**：
- 学术贡献：信号处理 + verdict-level 在 LLM agent 上的首次系统化
- 通用性：可挂任何评分器，受众广
- 论文方向：methodology paper（NeurIPS / ICLR）

### How Well — 质量标准

| 维度 | 标准 |
|------|------|
| 节省 vs 固定 max-iter | 在保持 success rate ≥ baseline 时，平均 iter 数减少 ≥ 30% |
| 延迟 | step() 调用 ≤ 1ms |
| 决策一致性 | 同输入序列重跑，决策完全一致（pure function） |
| 兼容性 | 不依赖 QAG-Gate，接受任何 score sequence |

---

## 验收标准（业务目标三问）

### 主路径
- [ ] 安装后 5 行代码可跑通
- [ ] 给定一个递增 score 序列 → 在合适时机输出 `deliver`
- [ ] 给定一个停滞序列 → 输出 `pivot` 或 `deliver(best)`

### 失败路径
- [ ] 序列只有 1 个点 → 输出 `continue`，不抛异常
- [ ] score 含 None / NaN → 跳过该点 + warning
- [ ] verdicts 缺失 → 仅用分数决策，diagnosis=unclear

### 完整状态
- [ ] 9 种典型决策场景都有单测覆盖
- [ ] capability_limit / eval_blind_spot 诊断都有触发用例

---

## 排期

| 里程碑 | 计划完成 |
|--------|----------|
| 架构 + 算法定稿 | W1 末 |
| 测试用例确认 | W2 末 |
| 实现 + 单测 100% | W4 末 |
| 合成数据生成器就位 | W4 末 |
| 主实验完成 | W6 末 |
| Paper draft v1 | W6 末 |
| arXiv + GitHub public | W11 |
| 投稿 | W12 |
