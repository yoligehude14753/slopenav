# Changelog

All notable changes to `slopenav` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-13

First public release.

### Added

- **`SlopeNav`** — main entry point. Consumes a sequence of quality
  scores plus optional per-criterion binary verdicts and emits one of
  `continue`, `pivot`, or `deliver` per step.
- **Dual-slope computation** — windowed linear regression for global
  trend and EMA difference for local momentum.
- **Verdict-level progress signal** — tracks which sub-criteria are
  improving and which are persistently failing.
- **Deterministic 9-rule decision tree** — fully inspectable, no
  hidden randomness.
- **Stagnation diagnosis** — when `pivot` is emitted, the result
  includes a `reason` (e.g., `slope_flat`, `verdict_stuck`,
  `oscillating`) so the agent harness can choose a remediation
  strategy.
- **Evaluator-agnostic** — works with `qag-gate`, G-Eval, RAGAS, or
  any custom score sequence.
- **Zero required dependencies** — pure Python, stdlib only.
- Apache 2.0 license.

### Validated against

- **500 synthetic traces with oracle stops** — efficiency
  (success_rate / avg_iterations) = **0.234** vs. **0.118** for
  fixed-5 (2.0× improvement).
- **100 real `gpt-4o-mini` Self-Refine traces on FLASK** — efficiency
  = **0.541** vs. **0.176** for fixed-5 (3.07× ratio; success
  0.93 vs. 0.88; mean 1.72 vs. 5.0 iterations).
- **Honest negative finding** — when score trajectories are flat
  (real Self-Refine on `gpt-4o-mini`, mean ≈ 0.85 throughout), a
  simple Δ-threshold baseline reaches efficiency 0.664 — **higher
  than SlopeNav**. SlopeNav's slope-and-verdict design pays off when
  trajectories are informative; flat regimes are best handled by
  thresholding. See [`docs/PAPER.md`](docs/PAPER.md) §6 for details.
- **Verdict-targeted reprompt (n = 32 pivot traces)** — Δscore =
  +0.034 vs. generic reprompt, Wilcoxon one-sided *p* = 0.065 — not
  significant.

### Known limitations

- Out of the box, `SlopeNav` is parameter-tuned for trajectories with
  3–10 iterations. Very long trajectories (50+) may benefit from a
  larger window size; see `slopenav.config.SlopeConfig`.
- The `pivot` decision is advisory — the agent harness is responsible
  for choosing the actual remediation prompt. See
  `examples/04_pivot_detection.py` for an example.
- Public API is **stable from 0.1.x onward**.

[Unreleased]: https://github.com/yoligehude14753/slopenav/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yoligehude14753/slopenav/releases/tag/v0.1.0
