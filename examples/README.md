# slopenav examples

Self-contained scripts showing how to plug **slopenav** into agent loops
(Self-Refine, Reflexion, Claude Code, Codex, AutoGen, CrewAI, Letta, etc.).
Every example is < 80 lines of Python.

## Install

```bash
pip install slopenav
# only example 02 needs qag-gate + openai:
pip install "qag-gate[openai]"
```

## Common env vars

| Variable | Required by | Default | Notes |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | `02_with_qag_gate.py` | — | any OpenAI-compatible key |
| `OPENAI_BASE_URL` | `02_with_qag_gate.py` | OpenAI | optional override |
| `QAG_MODEL` | `02_with_qag_gate.py` | `gpt-4o-mini` | judge model |

## Examples

### `01_minimal_decision.py` — read the slopes for a clean trajectory
Feeds the hardcoded scores `[0.30, 0.50, 0.70, 0.85]` into SlopeNav and prints
the per-iteration `decision.action`, `decision.reason`, linear slope, EMA
slope. No LLM call.

```bash
python examples/01_minimal_decision.py
```

Expected output (truncated):

```
iter  score   action     reason                         lin      ema
----------------------------------------------------------------------
0     0.300   continue   need_slope_data                0.0000   0.0000
1     0.500   continue   high_slope_improving           0.2000   0.0800
2     0.700   continue   high_slope_improving           0.2000   0.1280
3     0.850   continue   high_slope_improving           0.1850   0.1370
final score: 0.850, decision: continue, best_seen: 0.850
```

### `02_with_qag_gate.py` — end-to-end agent loop
Each round: OpenAI generates → QAG-Gate scores → SlopeNav decides. Up to 5
iterations. Demonstrates how `decision.action == "deliver"` cleanly stops a
real agent.

```bash
OPENAI_API_KEY=sk-... python examples/02_with_qag_gate.py
```

Expected output (truncated):

```
iter=0 score=0.62 action=continue reason=need_slope_data
iter=1 score=0.78 action=continue reason=high_slope_improving
iter=2 score=0.85 action=deliver  reason=good_enough_score
final score: 0.85, decision: deliver at iter=2
```

### `03_threshold_baseline_comparison.py` — SlopeNav vs baselines
Runs the same synthetic trajectory under three stopping rules:

* **SlopeNav** — dual-slope decision tree
* **delta_threshold** — stop when `|Δscore| < ε`
* **fixed_budget** — always run N iterations

Prints stop iteration, final score, success, and `score / iters` efficiency.

```bash
python examples/03_threshold_baseline_comparison.py
```

Expected output (truncated):

```
Trajectory: [0.55, 0.68, 0.78, 0.83, 0.85, 0.86, 0.86, 0.87]
Threshold:  0.8

SlopeNav           stop@iter=5  iters_used=6 final_score=0.860 success=True  efficiency=0.143
delta_threshold    stop@iter=4  iters_used=5 final_score=0.850 success=True  efficiency=0.170
fixed_budget=5     stop@iter=4  iters_used=5 final_score=0.850 success=True  efficiency=0.170

final: best strategy (success, then efficiency) = delta_threshold ...
```

### `04_pivot_detection.py` — stagnation → pivot
Feeds a trajectory that hovers around 0.40 to show that both slopes go
non-positive after a few rounds, at which point SlopeNav emits `pivot` (and
later `deliver`, since `max_pivots=1` is exhausted in this run).

```bash
python examples/04_pivot_detection.py
```

Expected output (truncated):

```
iter  score   action     reason                           lin      ema      pivots
------------------------------------------------------------------------------
0     0.400   continue   need_slope_data                  0.0000   0.0000   0
1     0.420   continue   weak_positive_slope              0.0200   0.0080   0
2     0.410   continue   weak_positive_slope              0.0050   0.0010   0
3     0.400   pivot      stagnant_or_declining           -0.0010  -0.0040   1
4     0.390   deliver    stagnation_capability_limit     -0.0040  -0.0060   1
final: pivot detected at iter=3, deliver at iter=4, best=0.420
```

## More integrations

For copy-paste-ready snippets that wire SlopeNav into Claude Code SDK, Aider,
AutoGen, CrewAI, Letta, LangGraph — plus an evaluator-agnostic G-Eval example
showing how to use SlopeNav *without* QAG-Gate — see
[`docs/INTEGRATIONS.md`](../docs/INTEGRATIONS.md).

## Notes

- Examples 1, 3, 4 are deterministic and require no network access.
- Example 2 performs real OpenAI calls — set `OPENAI_API_KEY` first.
- Each script prints a single `final: ...` line for quick smoke testing.
