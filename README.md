# slopenav

**Dual-slope iteration decision algorithm for AI agent quality convergence.**

Zero external dependencies. Pure Python ≥ 3.11.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)

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

Evaluated on 100 real Self-Refine trajectories (FLASK tasks × `gpt-4o-mini`, scored by QAG-Gate):

| Strategy | Success Rate | Avg Iterations | Efficiency |
|----------|-------------|----------------|------------|
| **SlopeNav** | **0.930** | **1.72** | **0.541** |
| Fixed-5 | 0.880 | 5.00 | 0.176 |
| Fixed-4 (Self-Refine default) | 0.910 | 4.00 | 0.228 |

3.07× efficiency vs Fixed-5 on this benchmark. Full experiment scripts in [`benchmarks/`](benchmarks/).

## Development

```bash
git clone https://github.com/yoligehude14753/openall
cd openall/projects/slopenav
pip install -e ".[dev]"
pytest tests/
```
