# Contributing to SlopeNav

## Development Setup

```bash
git clone https://github.com/[org]/slopenav.git
cd slopenav
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests (no external dependencies required)
pytest tests/ -v

# Property-based tests (Hypothesis)
pytest tests/property/ -v

# Coverage
pytest tests/ --cov=slopenav --cov-report=term-missing
```

## Architecture

SlopeNav is a pure Python library with **zero required dependencies**:

```
domain/    ← Models (VerdictSnapshot, Decision, etc.)
slope/     ← Slope computation (linear.py, ema.py) — pure functions
verdicts/  ← Verdict-level progress tracking — pure functions
decision/  ← decide() — pure function, 9-rule tree
diagnose/  ← Stagnation diagnosis — pure function
nav.py     ← SlopeNav — stateful wrapper over all the above
```

### Core Design Principle: Pure Functions

The `decide()` function in `decision/tree.py` is a **pure function**:
- Same inputs → always same output
- No side effects, no global state
- Fully unit-testable in isolation

`SlopeNav` (in `nav.py`) is the stateful wrapper that:
1. Maintains score history
2. Computes slopes and verdict progress
3. Calls `decide()` with computed values
4. Handles post-decision actions (stagnation diagnosis, pivot counting)

**Never add side effects to `decide()`.**

### Adding a New Decision Rule

1. Add constants to `decision/tree.py` if needed
2. Insert the rule at the appropriate priority position in `decide()`
3. Add unit test in `tests/unit/test_decision_tree.py` **first** (TDD)
4. Verify property tests still pass: `pytest tests/property/ -v`
5. Update `docs/adr/ADR-001-dual-slope-design.md` if the rule affects the core algorithm

### Modifying Slope Computation

The slope functions in `slope/` take a list of `(iteration, score)` pairs and return a float:

```python
def compute_linear_slope(history: list[tuple[int, float]]) -> float:
    """Returns slope in score/iteration units. Positive = improving."""
    ...
```

If you change the slope API, update:
- `tests/unit/test_slope.py`
- `tests/property/test_properties.py` (Properties 6, 7)
- `nav.py` where slopes are computed

## Commit Convention

```
feat: add verdict-level regression detection (Rule 2b)
fix: handle single-point history in EMA computation
test: add property test for EMA boundedness
bench: run P2 real data experiment
docs: update paper draft with P1 synthetic results
```
