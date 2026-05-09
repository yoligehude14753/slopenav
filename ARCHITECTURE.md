# slopenav · ARCHITECTURE

> 版本：v1.0 · 日期：2026-05-08

---

## 一、统一哲学

**纯算法库**：SlopeNav 不调用 LLM，不依赖外部网络，不持久化（除非用户主动 dump）。这让它可在任何 generator-evaluator loop 中嵌入，且测试和复现成本极低。

---

## 二、业务域识别

### 核心名词
- **ScorePoint**：(iteration, score)
- **Verdict**：(question, is_positive, weight)
- **VerdictHistory**：iteration → verdict 列表
- **Slope**：质量分对 iteration 的导数（线性 / EMA 两种估计）
- **VerdictProgress**：(flipped_positive, flipped_negative, stability_score, persistent_failures)
- **Decision**：(action, reason, slope_*, verdict_progress, diagnosis)
- **Diagnosis**：capability_limit / eval_blind_spot / unclear

### 数据归属

| 数据 | 唯一写入方 |
|------|----------|
| score_history | `SlopeNav.record()` |
| verdict_history | `SlopeNav.record()` |
| best_score | `SlopeNav.record()` |
| Decision | `DecisionEngine.decide()` |
| Diagnosis | `StagnationDiagnoser.diagnose()` |

---

## 三、目录结构

```
slopenav/
├── README.md
├── pyproject.toml
├── LICENSE                          # Apache 2.0
├── docs/
│   ├── ALGORITHM.md
│   ├── EXPERIMENTS.md
│   ├── TESTING.md
│   └── PAPER.md
├── src/
│   └── slopenav/
│       ├── __init__.py              # 公共 API
│       ├── domain/
│       │   ├── decision.py          # Decision, VerdictProgress
│       │   ├── verdict.py           # Verdict (与 qag-gate 对齐结构)
│       │   ├── score_point.py       # ScorePoint
│       │   └── diagnosis.py         # Diagnosis 枚举
│       ├── slope/
│       │   ├── linear.py            # OLS 线性回归斜率
│       │   ├── ema.py               # EMA 斜率
│       │   └── adaptive.py          # 自适应高斜率阈值（按 question 数）
│       ├── verdicts/
│       │   ├── progress.py          # flipped, stability, persistent
│       │   └── transitions.py       # verdict diff 工具
│       ├── decision/
│       │   ├── tree.py              # 决策树主逻辑
│       │   └── policies.py          # 各 action 的触发条件
│       ├── diagnose/
│       │   ├── stagnation.py        # 停滞原因诊断
│       │   └── strategies.py        # 各诊断策略
│       └── nav.py                   # SlopeNav 主类（Orchestrator）
├── tests/
│   ├── arch/
│   │   ├── test_no_external_deps.py # 不依赖 LLM / 网络
│   │   └── test_pure_function.py    # 决策必须是 pure function
│   ├── unit/
│   │   ├── test_linear_slope.py
│   │   ├── test_ema_slope.py
│   │   ├── test_verdict_progress.py
│   │   ├── test_decision_tree.py    # 9 种场景
│   │   └── test_stagnation_diagnose.py
│   ├── integration/
│   │   └── test_nav_full_loop.py    # 完整 record → decide
│   └── property/
│       └── test_monotonicity.py     # 性质测试：单调输入 → 单调决策
├── benchmarks/
│   └── 2026-05-slopenav-vs-baselines/
│       ├── PLAN.md
│       ├── ENV.md
│       ├── run.sh
│       ├── synthetic_data/
│       ├── real_data/               # 来自 easychat 历史 runs
│       └── RESULT.md
└── examples/
    ├── basic.py
    ├── with_qag_gate.py             # 演示与 QAG-Gate 集成
    ├── with_custom_evaluator.py
    └── with_reflexion.py            # 集成到 Reflexion
```

---

## 四、对外公共接口

```python
# src/slopenav/__init__.py
from slopenav.nav import SlopeNav
from slopenav.domain.decision import Decision, VerdictProgress
from slopenav.domain.verdict import Verdict
from slopenav.domain.diagnosis import Diagnosis

__version__ = "0.1.0"
__all__ = ["SlopeNav", "Decision", "VerdictProgress", "Verdict", "Diagnosis"]
```

主类签名：
```python
class SlopeNav:
    def __init__(
        self,
        window: int = 5,
        min_threshold: float = 0.80,
        max_pivots: int = 2,
        excellent_ceiling: float = 0.88,
        ema_alpha: float = 0.4,
        require_min_evals: int = 1,
    ): ...

    def record(
        self,
        iteration: int,
        score: float,
        verdicts: Optional[Sequence[Verdict]] = None,
        content_snapshot: Optional[str] = None,
    ) -> None: ...

    def decide(self) -> Decision: ...

    def diagnose(self, context: dict | None = None) -> Diagnosis: ...

    def step(
        self,
        iteration: int,
        score: float,
        verdicts: Optional[Sequence[Verdict]] = None,
    ) -> Decision:
        """Convenience: record + decide in one call."""
        self.record(iteration, score, verdicts)
        return self.decide()

    def best_iteration(self) -> int: ...
    def best_content(self) -> Optional[str]: ...
    def reset_pivot(self) -> None: ...
```

---

## 五、技术选型 ADR

### ADR-2026-05-12-no-external-deps

**决策**：核心依赖只有 numpy（用于斜率计算），不依赖 pandas / scipy。

**理由**：
- 使用场景是 hot loop 内调用，import 时间和内存占用敏感
- numpy 已是 Python 数据科学标配
- scipy 一些函数我们不需要，不引入

### ADR-2026-05-13-pure-function-decision

**决策**：`decide()` 必须是 pure function（同输入 → 同输出，无副作用）。

**理由**：
- 论文要复现：给定 score 序列，决策序列必须确定
- 测试简单：不需要 mock 时间、随机数
- 用户可信任：不会因 SlopeNav 的内部状态变化而行为不同

**Fitness Function**：`test_pure_function.py` 给同一输入跑 100 次，断言 100 次决策完全一致。

### ADR-2026-05-14-no-llm-anywhere

**决策**：SlopeNav 任何代码路径**禁止** import openai/anthropic/任何 LLM 库。

**理由**：
- 这是 SlopeNav 与现有所有"agent self-eval"工作的核心差异
- 一旦引入 LLM，论文的"零额外推理成本"卖点崩塌

**Fitness Function**：`test_no_external_deps.py`

---

## 六、与 QAG-Gate 的接口契约

```python
# slopenav/domain/verdict.py
@dataclass(frozen=True)
class Verdict:
    """
    与 qag-gate 的 Verdict 结构对齐，但无 import 依赖。
    Duck typing：只要有这些字段，SlopeNav 就能消费。
    """
    question: str
    category: str
    answer: bool
    is_positive: bool
    reason: str = ""
    weight: float = 1.0
    score_value: float = 1.0
```

**重要**：SlopeNav 完全不 import qag-gate；它只声明"我接受任何有这些字段的对象"。

`examples/with_qag_gate.py` 演示二者集成（本质是把 qag-gate 的 `Verdict` 直接传进来，duck typing 通过）。

---

## 七、Fitness Functions

```python
# tests/arch/test_no_external_deps.py
def test_no_llm_imports_anywhere():
    for f in Path("src/slopenav").rglob("*.py"):
        c = f.read_text()
        for forbidden in ["openai", "anthropic", "langchain", "requests"]:
            assert f"import {forbidden}" not in c, f"{f} imports {forbidden}"

def test_only_numpy_external_dep():
    pyproject = Path("pyproject.toml").read_text()
    deps = parse_toml_deps(pyproject)
    allowed = {"numpy", "typing-extensions"}
    assert set(deps) <= allowed

# tests/arch/test_pure_function.py
def test_decide_is_deterministic():
    nav1 = SlopeNav()
    nav2 = SlopeNav()
    seq = [(0, 0.5), (1, 0.6), (2, 0.7), (3, 0.75), (4, 0.8)]
    
    for it, s in seq:
        nav1.record(it, s); nav2.record(it, s)
    
    for _ in range(100):
        d1 = nav1.decide()
        d2 = nav2.decide()
        assert d1 == d2
```

---

## 八、可观测性

虽然 SlopeNav 不强制持久化，但提供可选的 `to_dict()` 让用户 dump：

```python
nav.to_dict()
# {
#   "score_history": [(0, 0.5), (1, 0.6), ...],
#   "verdict_history": {0: [...], 1: [...]},
#   "best_iteration": 4,
#   "best_score": 0.85,
#   "ema_score": 0.78,
#   "pivot_count": 1,
# }
```

对应 `from_dict()` 可恢复，方便 checkpoint / debug。
