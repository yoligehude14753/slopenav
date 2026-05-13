# Contributing to SlopeNav

Thanks for considering a contribution.

## Quick links

- Bug reports → [Issues](https://github.com/yoligehude14753/slopenav/issues)
- Feature requests → [Discussions](https://github.com/yoligehude14753/slopenav/discussions) (or open an Issue with the `enhancement` label)
- Security issues → see `SECURITY.md` (please do not file public issues for vulnerabilities)
- Integrations with other agent frameworks → [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md)

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

## Pull request workflow

1. Fork + branch off `main` (`feat/<slug>`, `fix/<slug>`, `docs/<slug>`).
2. Run `pytest tests/ -v` and `ruff check src tests` locally.
3. Open a PR against `main`; one logical change per PR, target ≤ 400 lines of diff.
4. Use Conventional Commits in the PR title (it becomes the squash commit message).
5. CI must be green; reviewer may ask for an extra property test if you touch `slope/`, `verdicts/`, or `decision/`.

## Good first issues

Look for issues labelled [`good-first-issue`](https://github.com/yoligehude14753/slopenav/labels/good-first-issue) and [`help-wanted`](https://github.com/yoligehude14753/slopenav/labels/help-wanted). If you would like to pick something up that doesn't have an issue yet, the following are concrete, small-scoped starters — each fits in a single PR:

- **New stagnation diagnosis rule**: add a `cause="oscillation"` branch in `diagnose/diagnoser.py` (verdicts flipping back and forth ≥3 times) with unit tests.
- **`SlopeNav.export_json()`**: serialise the full history + decision trace to JSON (and add a matching `from_json` constructor). Useful for debugging long runs.
- **More framework examples**: add an `examples/05_<framework>.py` that mirrors one section of `docs/INTEGRATIONS.md` end-to-end (Letta, LangGraph, CrewAI, Aider, …).
- **Translate docs to Chinese**: start with `README.md` → `README.zh-CN.md`, or `docs/INTEGRATIONS.md` → `docs/INTEGRATIONS.zh-CN.md`.
- **9-rule decision tree visualiser**: a tiny CLI / Jupyter helper that takes a list of `Decision` objects and renders an ASCII / matplotlib timeline annotated with which rule fired.
- **Type hints on `verdicts/progress.py`**: tighten the dict-shaped `persistent_failures` into a `TypedDict` or `@dataclass`, no behavioural change.
- **Pure-numpy slope path**: optional `slopenav[numpy]` extra already exists — wire it through `slope/linear.py` and `slope/ema.py` with a fallback if numpy is not installed, behind a benchmark in `benchmarks/`.
- **CI matrix expansion**: add Python 3.14 (or Windows / macOS) to `.github/workflows/ci.yml` and confirm green.

Comment on the issue (or open one) before you start so we can avoid duplicate work.
