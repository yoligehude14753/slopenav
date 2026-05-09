# slopenav · 测试 Plan

> 版本：v1.0 · 日期：2026-05-08

---

## 一、测试金字塔

SlopeNav 是纯算法库，所以测试金字塔比 QAG-Gate 更"下沉"：

```
        ┌─────────────────┐
        │ Property (~10)  │   hypothesis-based 性质测试
        ├─────────────────┤
        │ Integration ~10 │   完整 record + decide loop
        ├─────────────────┤
        │   Arch (~5)     │   Fitness Functions
        ├─────────────────┤
        │   Unit (~60)    │   决策树每条规则、各算法子模块
        └─────────────────┘
```

**没有 E2E**：因为 SlopeNav 没有外部副作用，integration 即足够。

---

## 二、业务目标三问

### 主路径
> 用户用 5 行代码集成 SlopeNav 到自己的 generator-evaluator loop，能在合理时机收到 `deliver` 决策且不漏交付。

### 失败路径
- score 序列只有 1 个点 → `continue`，不抛
- score = NaN → 跳过该点 + warning（不让 NaN 污染斜率）
- verdicts = None → 仅用分数决策，diagnosis = unclear
- 极端长序列（10000 点）→ 仍然 O(w) 复杂度

### 完整状态
- 9 条决策规则各有触发用例
- 3 种 diagnosis 各有触发用例
- 4 种 baseline（fixed/single-threshold/reflexion/oracle）都可作为对比

---

## 三、功能完整性清单

```markdown
## 功能完整性清单：SlopeNav

### 主流程
- [ ] `nav.step(iter, score)` 单步调用 → 返回 Decision
- [ ] Decision.action ∈ {continue, pivot, deliver}
- [ ] 决策原因（reason）是非空字符串，对应规则名
- [ ] best_iteration / best_score / best_content 跟踪正确

### 失败路径
- [ ] score 序列长度 = 1 → continue（数据不足）
- [ ] 序列仅 2 点但都 ≥ excellent_ceiling → deliver
- [ ] verdicts = None → decide 仍能跑，diagnosis=unclear
- [ ] 极端低分长序列 → 最终 deliver(best) 而非死循环
- [ ] reset_pivot() 后 pivot_count 归零

### 边界场景
- [ ] 全部 score 相同 → 第 5 轮触发 plateau，pivot 或 deliver
- [ ] verdicts 在两轮之间完全不重叠 → stability=0.5（默认）
- [ ] persistent_failures 数量精确等于 PERSISTENT_FAILURE_THRESHOLD - 1 → 不触发
- [ ] persistent_failures = THRESHOLD → 触发
- [ ] pivot_count == max_pivots → 不再 pivot，转 deliver

### 状态集（决策规则触发条件）
- [ ] Rule 0: 数据不足
- [ ] Rule 1: 优秀+稳定
- [ ] Rule 2: verdict 回退
- [ ] Rule 2.5: 良好+斜率平
- [ ] Rule 3: 强劲斜率
- [ ] Rule 4: 阈值上+稳定
- [ ] Rule 5: 持续失败
- [ ] Rule 6: 分数平稳
- [ ] Rule 7: 耐心耗尽
- [ ] Rule 8: 负斜率
- [ ] Rule 9: 弱正斜率

### 诊断（与 Rule 8 联动）
- [ ] capability_limit 触发：tool_failures > 70%
- [ ] eval_blind_spot 触发：file produced 但 score < 0.64
- [ ] unclear：其他

### 性能
- [ ] decide() 调用 P99 ≤ 1ms（10000 次实测）
- [ ] 内存峰值 ≤ 100KB（不含 verdicts 序列化）
```

---

## 四、单元测试设计（~60 个）

### 4.1 `slope/`（15 个）

```python
def test_linear_slope_zero_for_flat_series()
def test_linear_slope_positive_for_increasing()
def test_linear_slope_negative_for_decreasing()
def test_linear_slope_window_smaller_than_data()
def test_linear_slope_single_point_returns_zero()
def test_linear_slope_handles_window_2()

def test_ema_slope_first_point_returns_zero()
def test_ema_slope_alpha_high_responsive()
def test_ema_slope_alpha_low_smooth()

def test_adaptive_high_slope_default_when_no_questions()
def test_adaptive_high_slope_decreases_with_more_questions()
def test_adaptive_high_slope_clamped_to_min_max()

def test_question_step_estimation()
def test_window_truncation_correctness()
def test_slope_with_outlier_robust_via_ema()
```

### 4.2 `verdicts/`（10 个）

```python
def test_flipped_positive_detected()
def test_flipped_negative_detected()
def test_stability_score_perfect()
def test_stability_score_chaotic()
def test_stability_score_partial()
def test_persistent_failure_below_threshold_not_detected()
def test_persistent_failure_at_threshold_detected()
def test_persistent_failure_with_long_history()
def test_net_progress_positive()
def test_net_progress_negative()
```

### 4.3 `decision/tree.py`（每条规则一个，~15 个）

```python
def test_rule0_excellent_first_eval_delivers()
def test_rule0_above_threshold_first_eval_delivers()
def test_rule0_low_first_eval_continues()
def test_rule1_excellent_stable_delivers()
def test_rule1_excellent_unstable_continues_with_few_data()
def test_rule2_verdict_regression_continues()
def test_rule2_5_good_enough_flat_delivers()
def test_rule3_high_slope_continues()
def test_rule4_above_threshold_stable_verdicts_delivers()
def test_rule5_persistent_failures_pivots()
def test_rule5_persistent_failures_pivots_exhausted_delivers()
def test_rule6_score_plateau_below_threshold_pivots()
def test_rule7_patience_exhausted_delivers_best()
def test_rule8_negative_slope_pivots()
def test_rule9_weak_positive_continues()
```

### 4.4 `diagnose/`（10 个）

```python
def test_diagnose_capability_limit_high_tool_failure()
def test_diagnose_capability_limit_no_successful_tools()
def test_diagnose_eval_blind_spot_files_produced_low_score()
def test_diagnose_unclear_no_tool_results()
def test_diagnose_unclear_mixed()
def test_diagnose_capability_limit_overrides_pivot_to_deliver()
def test_diagnose_eval_blind_spot_overrides_pivot_to_deliver()
def test_diagnose_with_few_score_points_returns_unclear()
def test_diagnose_below_threshold_3_avg()
def test_diagnose_with_only_failed_tools_capability_limit()
```

### 4.5 `nav.py`（10 个）

```python
def test_record_updates_best_score()
def test_record_updates_ema()
def test_record_persists_verdicts()
def test_pivot_resets_ema()
def test_max_pivots_enforced()
def test_step_combines_record_and_decide()
def test_to_dict_round_trips_via_from_dict()
def test_concurrent_record_safe()  # 同 nav 不允许并发，测试 explicit error
def test_reset_pivot_count_zero()
def test_get_best_content_when_no_snapshot_returns_none()
```

---

## 五、Fitness Functions（~5 个）

```python
# tests/arch/test_no_external_deps.py
def test_no_llm_imports():
    """slopenav 任何文件不能 import openai/anthropic/langchain/requests"""

def test_only_numpy_external_dep():
    """pyproject.toml 的依赖只能是 numpy + typing-extensions"""

# tests/arch/test_pure_function.py
def test_decide_is_deterministic():
    """同输入序列 100 次重跑，决策完全一致"""

def test_decide_no_global_state():
    """两个独立 SlopeNav 实例之间不互相影响"""

# tests/arch/test_module_boundaries.py
def test_external_imports_from_init_only():
    """外部用户只从 slopenav 顶层导入"""
```

---

## 六、集成测试（~10 个）

```python
def test_full_loop_early_converge():
    """递增到 0.85 → 在 iter 5-7 deliver"""

def test_full_loop_plateau_then_pivot():
    """先涨后平 → 触发 pivot"""

def test_full_loop_capability_limit():
    """工具持续失败 → diagnose + deliver(best)"""

def test_full_loop_eval_blind_spot():
    """文件 produced 但 score 低 → diagnose + deliver(best)"""

def test_full_loop_max_pivots_then_deliver():
    """连续 2 次 pivot 后 → 必 deliver"""

def test_full_loop_with_qag_gate_verdicts():
    """传入 QAG-Gate 风格的 verdicts → verdict_progress 正确"""

def test_full_loop_with_custom_verdict_class():
    """duck typing：自定义 verdict 类（有相同字段）也能跑"""

def test_full_loop_serialization_round_trip():
    """to_dict → from_dict 后继续 decide，行为一致"""

def test_full_loop_long_horizon_30_iters():
    """30 iter 长序列下不会内存溢出（window 限制生效）"""

def test_full_loop_reflexion_integration():
    """examples/with_reflexion.py 跑通"""
```

---

## 七、Property-based 测试（~10 个）

```python
@given(seq=lists(floats(0,1), min_size=1, max_size=20))
def test_monotonic_increasing_eventually_delivers_if_above_threshold(seq):
    sorted_seq = sorted(seq)
    nav = SlopeNav()
    for i, s in enumerate(sorted_seq):
        d = nav.step(i, s)
        if d.action == "deliver":
            assert s >= 0.0  # always true; just sanity
            return
    if sorted_seq[-1] >= 0.80:
        pytest.fail("Should have delivered")

@given(seq=lists(floats(0, 0.5), min_size=10))
def test_persistently_low_eventually_terminates(seq):
    nav = SlopeNav()
    for i, s in enumerate(seq):
        d = nav.step(i, s)
        if d.action == "deliver":
            return
    # 必须 pivot 至少一次或 deliver
    assert any(...)

@given(seq=lists(floats(0,1), min_size=2))
def test_decision_doesnt_crash_on_random(seq):
    nav = SlopeNav()
    for i, s in enumerate(seq):
        d = nav.step(i, s)
        assert d.action in {"continue", "pivot", "deliver"}

@given(seq=lists(floats(0,1), min_size=5))
def test_pivot_count_never_exceeds_max(seq):
    nav = SlopeNav(max_pivots=2)
    pivots = 0
    for i, s in enumerate(seq):
        d = nav.step(i, s)
        if d.action == "pivot":
            pivots += 1
            nav.reset_pivot()  # only reset after we ack
    assert nav.pivot_count <= 2
```

---

## 八、性能测试

```python
def test_decide_p99_under_1ms():
    """在 10000 次随机调用下，decide() P99 ≤ 1ms"""
    nav = SlopeNav()
    for i in range(100):  # warmup
        nav.step(i, random.random())
    
    latencies = []
    for i in range(10000):
        t0 = time.perf_counter_ns()
        nav.decide()
        latencies.append(time.perf_counter_ns() - t0)
    
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    assert p99 < 1_000_000, f"P99 = {p99/1000:.2f}μs > 1ms"
```

---

## 九、Flaky 政策

SlopeNav 是纯算法，**不允许任何 Flaky 测试**。出现 Flaky → 直接视为 bug 修复。
