# slopenav · 实验 Plan

> 版本：v1.0 · 日期：2026-05-08 · 遵循 `15-evidence-first.mdc`

---

## 一、实验四要素

### 假设 H1（主假设）
> SlopeNav 在保持 task success rate ≥ 固定 max-iter (=3) 的前提下，**平均迭代次数减少 ≥ 30%**。
> **支持阈值**：iter_avg(SlopeNav) ≤ 0.7 × iter_avg(fixed=3) 且 success_SlopeNav ≥ success_fixed
> **反驳阈值**：iter_avg(SlopeNav) > iter_avg(fixed=3) 或 success_SlopeNav < success_fixed - 5%

### 假设 H2
> SlopeNav 比单阈值停止策略 (score ≥ 0.80) **假交付率减少 ≥ 50%**。
> 假交付定义：决策为 deliver 但 oracle 后续 iteration 还能涨 ≥ 0.05。

### 假设 H3
> SlopeNav 相对 oracle baseline 的 regret ≤ 10%。
> Regret = oracle_max_score - slopenav_delivered_score（对每个 trace 取 mean）

### 假设 H4（消融）
> 移除 verdict-level analysis 后，假交付率显著上升（Δ ≥ 0.10）。

### 假设 H5（消融）
> 移除停滞诊断（直接 pivot）后，capability_limit 类任务上的算力浪费上升 ≥ 30%。

### 假设 H6（性能）
> `decide()` 调用 P99 ≤ 1ms（Python 3.12, numpy 1.x）。

---

## 二、变量

### 操控
- 停止策略：SlopeNav / fixed-3 / fixed-5 / fixed-10 / single-threshold-0.80 / oracle / Reflexion 自评停
- 任务难度：easy / mid / hard（影响 generator 生成 trace 的形状）
- 评估器噪声：σ ∈ {0, 0.05, 0.10}（注入到合成 score 序列）

### 不变
- generator 模型：固定（合成数据用脚本，真实数据用 easychat 历史 runs）
- 评估器（生成 verdicts）：固定 QAG-Gate v0.1
- 硬件：M4 Pro 64GB

---

## 三、度量

| 指标 | 定义 | 单位 |
|------|------|------|
| **avg_iters** | 决策为 deliver 时平均 iteration 数 | 次 |
| **success_rate** | 在 max_iter=10 内达到 score ≥ 0.80 的任务比例 | % |
| **false_deliver_rate** | deliver 后 oracle 显示后续可涨 ≥ 0.05 的比例 | % |
| **regret** | oracle_max - delivered_score | [0,1] |
| **wasted_iters** | capability_limit 任务上多跑的 iteration 数 | 次 |
| **decide_latency_p99** | `decide()` 调用延迟 P99 | ms |

---

## 四、阈值

| 假设 | 支持 | 反驳 | 不确定 |
|------|------|------|--------|
| H1 | iter减30% & success不降 | iter没减或success降5% | 之间 |
| H2 | false_deliver降50% | false_deliver没降 | 降<50% |
| H3 | regret ≤ 10% | regret > 20% | 10-20% |
| H4 | Δfalse_deliver ≥ 0.10 | <0 | 0-0.10 |
| H5 | wasted_iters增30% | <0 | 0-30% |
| H6 | P99 ≤ 1ms | P99 > 2ms | 之间 |

---

## 五、数据集

## 五、实验批次设计（渐进式，按批次 go/no-go）

> **原则**：SlopeNav 自身不调 LLM（纯算法），合成数据零成本；
> 只有用 QAG-Gate 标注真实 trace 时才产生 API 费用。
> 因此先把合成数据跑通，再逐步引入真实数据。

### P0：算法冒烟（200 条合成 trace，¥0）

**目的**：确认 `decide()` 不崩溃、输出合理、9 条规则都能触发。  
**操作**：每类 trace 各取 25 条（共 200 条），跑 SlopeNav 和 fixed-3，检查输出。

| 检查项 | 通过标准 |
|--------|---------|
| `decide()` 无 exception | 100% 通过 |
| 9 条规则都至少触发 1 次 | 全部触发 |
| deliver_action_rate ∈ (0, 1) | 不全 continue 也不全 deliver |

**不通过 → 修算法后重跑 P0，零成本可无限重跑。**

---

### P1：试点（500 条合成 trace × 4 策略，¥0）

**目的**：H1 方向验证，确认 SlopeNav 比 fixed-3 在 avg_iters 上有改善。  
**数据**：每类 trace 各取 62-63 条，保证各类型代表性。

| 假设 | 通过条件 |
|------|---------|
| H1 方向 | iter_avg(SlopeNav) < iter_avg(fixed-3) |
| H6 性能 | decide() P99 ≤ 1ms |

**不通过 → 分析哪类 trace 上 SlopeNav 反而更多 iter，调整决策树参数。**

---

### P2：真实数据验证（100 条真实 trace，¥200）

**目的**：确认合成数据的结论在真实 easychat 历史 run 上是否成立。  
**操作**：从 easychat 历史抽取 100 条 iteration ≥ 3 的 run，用 QAG-Gate 生成 verdicts（¥200），用 oracle 标注 ground truth。

| 指标 | 通过条件 |
|------|---------|
| H1 真实数据 | iter_avg(SlopeNav) < iter_avg(fixed-3) |
| 方向与合成一致 | 不要求幅度，方向一致即可 |

**不通过 → 需要分析真实 trace 与合成 trace 的分布差异，可能需要调整合成生成器。**

---

### P3：完整实验（2000 合成 + 300 真实 trace × 6 策略 × 3 seeds，¥800）

**目的**：论文级数据。仅在 P2 通过后执行。

**数据规模（完整）**：

| 类型 | 数量 | 描述 |
|------|------|------|
| **early-converge** | 300 | iter 1-2 就达到 0.85 |
| **steady-improve** | 300 | 单调递增 iter 5-7 越线 |
| **plateau-then-deliver** | 300 | 平台期应 pivot |
| **plateau-then-pivot** | 300 | pivot 后涨到 0.85 |
| **declining** | 200 | 持续下降，应早停 |
| **noisy-improving** | 200 | 大趋势上涨但 σ=0.10 |
| **capability-limit** | 200 | 工具持续失败 |
| **eval-blind-spot** | 200 | 文件已产出但 score 卡在 0.4 |

每条 trace：10 iterations × score + verdicts（10 个 questions 的 binary）

**消融**：no-verdict-level / no-diagnosis / no-ema / no-adaptive-slope

---

### 数据开源

`benchmarks/synthetic/` 完整开源（包括生成脚本和种子）。  
真实数据脱敏后开源 `benchmarks/real/`，每条 trace 仅含 score 序列 + verdict 模式 + oracle 标注，不含原始 task 文本。

---

## 六、Baselines 实现

| Baseline | 实现 |
|----------|------|
| **fixed-N** | 跑 N iter 后强制 deliver（N ∈ {3, 5, 10}） |
| **single-threshold** | score ≥ 0.80 立即 deliver |
| **double-threshold** | score ≥ 0.80 且 score ≥ 上轮 → deliver |
| **Reflexion-style** | 用一个 LLM 调用问"是否完成"（这里 mock 为"score ≥ 0.75 时 70% 概率说完成"） |
| **oracle** | 事后看，挑 max(score) 对应的 iter |

所有 baseline 在 `slopenav/baselines/` 下实现，与主算法同等待遇做实验。

---

## 七、实验目录结构

```
slopenav/benchmarks/
└── 2026-05-slopenav-vs-baselines/
    ├── PLAN.md
    ├── ENV.md
    ├── run.sh
    ├── synthetic/
    │   ├── generate.py
    │   ├── data/
    │   │   ├── traces.jsonl              # 2000 条
    │   │   └── ground_truth.jsonl        # oracle 标注
    │   └── seeds.json
    ├── real/
    │   ├── extract_from_easychat.py
    │   ├── data/
    │   │   └── traces.jsonl              # ~300 条
    │   └── anonymize.py
    ├── runs/
    │   ├── slopenav-01.json
    │   ├── slopenav-02.json
    │   ├── fixed-3-01.json
    │   ├── single-threshold-01.json
    │   └── ...
    ├── ablations/
    │   ├── no-verdict-level.json
    │   ├── no-diagnosis.json
    │   └── no-ema.json
    ├── RESULT.md
    └── REPRO.md
```

---

## 八、`run.sh`（分批次，按 PHASE 参数控制）

```bash
#!/usr/bin/env bash
# 渐进式实验：SlopeNav 自身零 LLM 成本，仅真实数据标注阶段有 API 费用
# 用法：
#   PHASE=p0 bash run.sh   # 算法冒烟（¥0，200 合成 trace）
#   PHASE=p1 bash run.sh   # 试点（¥0，500 合成 trace）
#   PHASE=p2 bash run.sh   # 真实数据（¥200，100 真实 trace）
#   PHASE=p3 bash run.sh   # 完整实验（¥800，2000+300 trace）

set -euo pipefail

PHASE=${PHASE:-p0}
echo ">>> Running phase: $PHASE"

python -m venv .venv && source .venv/bin/activate
pip install -e ".[bench]" -q

case "$PHASE" in
  p0)
    N_SYNTHETIC=200; USE_REAL=false; SEEDS="01"; DO_ABLATION=false
    echo ">>> P0 冒烟：¥0，算法正确性确认"
    ;;
  p1)
    N_SYNTHETIC=500; USE_REAL=false; SEEDS="01"; DO_ABLATION=false
    echo ">>> P1 试点：¥0，H1 方向验证"
    echo ">>> 通过条件：iter_avg(SlopeNav) < iter_avg(fixed-3)"
    ;;
  p2)
    N_SYNTHETIC=500; USE_REAL=true; N_REAL=100; SEEDS="01"; DO_ABLATION=false
    echo ">>> P2 真实数据：¥200（QAG-Gate 标注 100 条 trace）"
    echo ">>> 通过条件：真实数据上方向与合成一致"
    ;;
  p3)
    N_SYNTHETIC=2000; USE_REAL=true; N_REAL=300; SEEDS="01 02 03"; DO_ABLATION=true
    echo ">>> P3 完整实验：¥800，仅在 P2 通过后运行"
    ;;
  *)
    echo "未知 PHASE: $PHASE（可选 p0/p1/p2/p3）"; exit 1
    ;;
esac

# 1. 生成合成数据（纯脚本，¥0）
python -m slopenav.bench.generate_synthetic \
    --n "$N_SYNTHETIC" \
    --output "data/$PHASE/synthetic/traces.jsonl" \
    --seed 42
echo ">>> 生成 $N_SYNTHETIC 条合成 trace，费用 ¥0"

# 2. 提取真实数据（P2/P3 才运行，触发 QAG-Gate API 调用）
if [ "$USE_REAL" = true ]; then
    [ -z "${OPENAI_API_KEY:-}" ] && echo "OPENAI_API_KEY missing（P2/P3 需要 QAG-Gate 打分）" && exit 1
    python -m slopenav.bench.extract_real \
        --input "${EASYCHAT_RUNS:-$HOME/.easychat/runs}" \
        --n "$N_REAL" \
        --output "data/$PHASE/real/traces.jsonl"
    echo ">>> 提取 $N_REAL 条真实 trace，已用 QAG-Gate 打分（估计 ¥200/¥800）"
fi

# 3. 跑各 baseline
TRACE_INPUTS="data/$PHASE/synthetic/traces.jsonl"
[ "$USE_REAL" = true ] && TRACE_INPUTS="$TRACE_INPUTS data/$PHASE/real/traces.jsonl"

for method in slopenav fixed-3 fixed-5 single-threshold oracle; do
    for seed in $SEEDS; do
        OUT="data/$PHASE/runs/${method}-${seed}.json"
        [ -f "$OUT" ] && echo ">>> 已存在 $OUT，跳过" && continue
        python -m slopenav.bench.run \
            --method "$method" \
            --traces $TRACE_INPUTS \
            --output "$OUT" \
            --seed "$seed"
    done
done

# 4. 消融（仅 p3）
if [ "$DO_ABLATION" = true ]; then
    for ablation in no-verdict-level no-diagnosis no-ema no-adaptive-slope; do
        python -m slopenav.bench.run \
            --method "slopenav-$ablation" \
            --traces "data/$PHASE/synthetic/traces.jsonl" \
            --output "data/$PHASE/ablations/${ablation}.json"
    done
fi

# 5. 性能 benchmark（任何阶段都跑，¥0）
python -m slopenav.bench.perf --iterations 10000 --output "data/$PHASE/runs/perf.json"

# 6. 分析 + go/no-go
python -m slopenav.bench.analyze \
    --phase "$PHASE" \
    --runs "data/$PHASE/runs/*.json" \
    --ground-truth "data/$PHASE/synthetic/ground_truth.jsonl" \
    --output "data/$PHASE/RESULT.md"

echo ""
echo ">>> 结果见 data/$PHASE/RESULT.md"
echo ">>> 请检查 go/no-go 条件后再决定是否进入下一批次"
```

---

## 九、最小样本量

- **主实验**：2000 合成 trace × 8 method × 3 seed = 48000 决策计算
- **真实**：300 trace × 8 method = 2400
- **性能**：10000 次 `decide()` 调用（满足 P99 计算需要）

---

## 十、属性测试（property-based testing）

基于 hypothesis 库，验证决策树的关键性质：

```python
@given(score_seq=lists(floats(0, 1), min_size=1, max_size=20))
def test_monotonic_increasing_eventually_delivers(score_seq):
    """单调递增的序列最终一定 deliver"""
    sorted_seq = sorted(score_seq)
    nav = SlopeNav()
    for i, s in enumerate(sorted_seq):
        d = nav.step(i, s)
        if d.action == "deliver":
            return
    if sorted_seq[-1] >= 0.80:
        pytest.fail("Monotonically increasing to ≥0.80 should deliver")

@given(score_seq=lists(floats(0, 0.3), min_size=10))
def test_low_scores_eventually_pivot_or_deliver_best(score_seq):
    """持续低分一定不会 forever-continue"""
    nav = SlopeNav(max_pivots=2)
    actions = []
    for i, s in enumerate(score_seq):
        d = nav.step(i, s)
        actions.append(d.action)
        if d.action == "deliver":
            return
    pytest.fail(f"Stagnant low scores never delivered: {actions}")
```

---

## 十一、RESULT.md 模板

```markdown
# SlopeNav vs Baselines 实验结果

## 假设回答
- H1: 支持/反驳 — iter_avg=X.X (vs fixed-3=Y.Y), success=A% (vs B%)
- H2: ...

## 主表

| 方法 | avg_iters | success_rate | false_deliver | regret |
|------|-----------|--------------|---------------|--------|
| SlopeNav | | | | |
| fixed-3 | | | | |
| fixed-5 | | | | |
| single-threshold | | | | |
| reflexion-mock | | | | |
| oracle | - | - | 0 | 0 |

## 按场景拆分

| 场景 | SlopeNav avg_iters | fixed-3 avg_iters | success diff |
|------|---|---|---|
| early-converge | | | |
| steady-improve | | | |
| plateau-then-pivot | | | |
| capability-limit | | | |
| ... | | | |

## 消融

| 配置 | avg_iters | false_deliver | regret |
|------|-----------|---------------|--------|
| Full SlopeNav | | | |
| - verdict-level | | | |
| - diagnosis | | | |
| - EMA slope | | | |
| - adaptive slope | | | |

## 性能

| 指标 | 值 |
|------|---|
| decide() P50 | μs |
| decide() P99 | μs |
| 内存峰值 | KB |
```
