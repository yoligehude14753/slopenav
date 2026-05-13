# slopenav

**A stopping criterion for long-running coding agents.** Decides between `continue` / `pivot` / `deliver` on every iteration of a Self-Refine, Reflexion, Claude Code SDK, Codex CLI, AutoGen / CrewAI / Letta loop. Zero required dependencies, pure Python ≥ 3.11, ~500 LOC.

[![PyPI](https://img.shields.io/pypi/v/slopenav?cacheSeconds=300)](https://pypi.org/project/slopenav/)
[![Python](https://img.shields.io/pypi/pyversions/slopenav?cacheSeconds=300)](https://pypi.org/project/slopenav/)
[![CI](https://github.com/yoligehude14753/slopenav/actions/workflows/ci.yml/badge.svg)](https://github.com/yoligehude14753/slopenav/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/yoligehude14753/slopenav/branch/main/graph/badge.svg)](https://codecov.io/gh/yoligehude14753/slopenav)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-orange)](docs/PAPER.md)

## Why this exists

Self-Refine ships with a fixed iteration budget of 4. Reflexion typically uses 3–5. Claude Code, Codex CLI, Cursor Composer, and Aider effectively run until a hand-coded heuristic or a token cap fires. None of these adapt to the actual trajectory of the agent — easy tasks burn tokens past convergence, hard tasks get cut off before they would have improved.

SlopeNav consumes a sequence of quality scores plus optional per-criterion binary verdicts and emits one of `continue` / `pivot` / `deliver`. It tracks two slopes in parallel (windowed OLS for the long-run trend, EMA difference for local momentum), looks at per-criterion progress (which sub-criteria are persistently failing → cleanest pivot signal), and routes the inputs through a deterministic 9-rule decision tree that is small enough to read end-to-end.

## Quick Start

```python
from slopenav import SlopeNav, Decision

nav = SlopeNav()
for iteration, score in enumerate(scores):
    decision: Decision = nav.step(iteration, score, verdicts=verdicts[iteration])
    if decision.action == "deliver":
        break
    if decision.action == "pivot":
        # change strategy
        pass
```

## Install

```bash
pip install slopenav          # zero dependencies
pip install "slopenav[numpy]" # optional: faster slope computation
```

## Algorithm

Dual-slope estimation:
- **Linear regression** (sliding window, captures long-term trend)
- **EMA** (Exponential Moving Average, captures recent momentum)

9-rule decision tree:
1. Excellent + stable → deliver
2. Verdict regression → continue
3. Good-enough + flat slope → deliver
4. High slope → continue
5. Persistent failures (≥3 iters) → pivot
6. Score plateau → pivot or deliver
7. Patience exhausted → deliver best
8. Negative slope → diagnose (capability_limit vs eval_blind_spot)
9. Weak positive → continue with patience limit

## Paper

SlopeNav: Dual-Slope Convergence Tracking for Efficient Iterative AI Agents (arXiv preprint, 2026). See [`docs/PAPER.md`](docs/PAPER.md) for full draft.

## Validation

Evaluated on 100 real Self-Refine trajectories (FLASK instructions × `gpt-4o-mini`, every iteration scored by QAG-Gate):

| Strategy | Success Rate | Avg Iterations | Efficiency |
|----------|-------------|----------------|------------|
| **SlopeNav** | **0.930** | **1.72** | **0.541** |
| Fixed-4 (Self-Refine default) | 0.910 | 4.00 | 0.228 |
| Fixed-5 | 0.880 | 5.00 | 0.176 |
| Δ-threshold (tuned early-stop) | 0.950 | 1.43 | 0.664 |

**Honest negative comparison.** On *this* corpus the QAG scores are nearly flat (~0.85 at every iteration), so a plain Δ-threshold baseline outperforms SlopeNav on efficiency. SlopeNav's slope-and-verdict design pays off on trajectories where the score actually moves — e.g. the synthetic regimes in [`docs/PAPER.md`](docs/PAPER.md) §5.1 where SlopeNav reaches 2.0× the efficiency of Fixed-5. We report both rather than cherry-picking.

Full experiment scripts in [`benchmarks/`](benchmarks/).

## Works with

SlopeNav is evaluator-agnostic — it consumes any sequence of `(score, optional verdicts)`:

- **QAG-Gate** (recommended; verdicts unlock the persistent-failure pivot signal)
- **G-Eval, RAGAS, MT-Bench judge prompt** (scalar score only; verdict-level features degrade gracefully)
- Custom heuristics, unit-test pass rates, build-success flags

Drop-in for **Self-Refine, Reflexion, ReAct, AutoGen, CrewAI, Letta, LangGraph**, or any agent loop that already produces a per-iteration scalar.

Copy-paste-ready integration snippets (including a non-QAG-Gate G-Eval example) are in [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md).

## Development

```bash
git clone https://github.com/yoligehude14753/openall
cd openall/projects/slopenav
pip install -e ".[dev]"
pytest tests/
```
