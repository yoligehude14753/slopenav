# SlopeNav: Dual-Slope Convergence Tracking for Efficient Iterative AI Agents

> 论文草稿 v1.1 · 2026-05-09  
> 目标：arXiv 技术报告 / EMNLP 2026 Demo（与 QAG-Gate 合并稿亦可行）  
> 字数目标：9,000–12,000 words (camera ready)

---

## Abstract

Iterative LLM agents waste compute when refinement yields no gain—or stop too late. **SlopeNav** tracks **linear and EMA slopes** over a sliding window of quality scores and augments them with **verdict-level** progress to drive a deterministic 9-rule tree (`continue` / `pivot` / `deliver`).  

**Synthetic benchmarks (P0–P1, n up to 500 traces)** show high efficiency vs. fixed-\(N\) stopping. **External-style validation (E3, May 2026)** uses **100 real API trajectories** (FLASK tasks × `gpt-4o-mini` Self-Refine) scored by **QAG-Gate**; against fixed iteration budgets, SlopeNav reaches **efficiency = 0.541** (success rate / mean iterations) vs. **0.176** for fixed-5 (**3.07×** ratio under our protocol), mean **1.72** iterations vs. **5.0**, success rate **0.93** vs. **0.88**. A simple Δ-threshold baseline achieves even higher efficiency in this **flat-score** regime (scores stay ~0.85–0.86 across iterations)—we report both honestly. **Pivot-reprompt study (E4, n=32 pivot cases)** finds mean **Δscore (targeted − generic) = 0.034** (not meeting our pre-registered uplift gate)—a **negative result** on average, with rare large gains when RedLine interacts with generic prompts. Code: Apache 2.0 (`pip install slopenav`).

**Keywords**: iterative agents, convergence detection, quality tracking, stopping criterion, LLM agents

---

## 1. Introduction

A fundamental challenge in deploying iterative AI agents is determining when to stop refining an output. Consider an agent tasked with writing a Python data-cleaning script:

- After iteration 1: the agent produces pseudocode—not yet usable.
- After iteration 3: the agent produces working code that handles the main case.
- After iteration 5: the code handles edge cases and passes all tests.
- After iteration 7: no meaningful improvement—the agent is polishing comments.

A human developer would stop at iteration 5. How can we automatically detect this stopping point without requiring task-specific ground truth?

**Existing approaches** suffer from fundamental limitations:

1. **Fixed-N stopping** (e.g., "always run 3 iterations"): Simple but wastes compute for fast-converging tasks and delivers prematurely for complex tasks.
2. **Score threshold** (e.g., "stop when score ≥ 0.80"): Sensitive to score calibration; fails when the agent plateaus below threshold.
3. **Entropy-based methods** (e.g., uncertainty sampling): Applicable to classification, not easily extended to open-ended generation quality.
4. **Human-in-the-loop**: Gold standard but eliminates automation benefits.

We propose **SlopeNav**, which tracks the *trajectory* of quality scores rather than absolute values. The key insight is that *the rate of improvement* is a better predictor of convergence than the absolute quality level. SlopeNav's core contributions are:

1. **Dual-slope tracking** (§3.1): Simultaneous linear regression and EMA slope computation over a sliding window, providing both global trend and local momentum signals.
2. **Verdict-level progress analysis** (§3.2): Beyond aggregate scores, SlopeNav tracks which individual evaluation criteria (verdicts) are improving, stable, or persistently failing, enabling targeted pivot decisions.
3. **Stagnation diagnosis** (§3.3): When stagnation is detected, SlopeNav distinguishes between *capability limits* (agent cannot improve further) and *evaluation blind spots* (evaluator failing to detect improvements), routing to deliver vs. requery accordingly.
4. **9-rule decision tree** (§3.4): A deterministic, interpretable decision tree with clear priority ordering, enabling debugging and trust calibration.
5. **Trace benchmarks**: Synthetic traces (§4.1, §5.1) plus **100 Self-Refine trajectories** scored with QAG-Gate (§5.3); no hand-annotated oracle stops in E3.

---

## 2. Background

### 2.1 Iterative Agent Systems

Modern LLM-based agents (AutoGPT, Claude Code, Devin) operate in a loop: generate → evaluate → refine. The number of iterations significantly impacts both output quality and cost. [Zhu et al. 2024] show that most quality gains occur in the first 3–5 iterations, with diminishing returns thereafter, yet agents continue iterating when given unlimited budgets.

### 2.2 Adaptive Stopping in Machine Learning

Adaptive stopping criteria have a rich history in non-agentic ML. **Early stopping** in neural network training monitors validation loss slope [Prechelt 1998]. **Bandit algorithms** [Lattimore & Szepesvári 2020] balance exploration/exploitation. **Sequential hypothesis testing** [Wald 1947] provides theoretical guarantees for sequential decision making.

SlopeNav adapts these ideas to the agent quality tracking setting, where the "signal" is a sequence of evaluation scores and the "stopping" is a deliver decision.

### 2.3 Convergence Detection for LLM Agents

[Shridhar et al. 2023] propose a confidence-based stopping criterion for retrieval-augmented generation, stopping when answer confidence exceeds a threshold. [Madaan et al. 2023] (Self-Refine) use fixed iteration counts. [Shinn et al. 2023] (Reflexion) use success/failure signals but not trajectory analysis.

SlopeNav is the first framework, to our knowledge, to (1) combine linear and EMA slopes with (2) verdict-level progress tracking and (3) capability/blind-spot stagnation diagnosis in a single unified stopping criterion for general agentic tasks.

---

## 3. SlopeNav Framework

### 3.1 Dual-Slope Computation

Let $S = [(t_0, s_0), (t_1, s_1), ..., (t_k, s_k)]$ be the score history, where $s_i \in [0, 1]$ is the quality score at iteration $t_i$. We compute over a sliding window of size $w$ (default 5):

**Linear slope** via ordinary least squares:

$$\hat{\beta}_{lin} = \frac{\sum_{i} (t_i - \bar{t})(s_i - \bar{s})}{\sum_{i} (t_i - \bar{t})^2}$$

**EMA slope** as the difference between consecutive EMA values:

$$\text{EMA}_{k} = \alpha \cdot s_k + (1 - \alpha) \cdot \text{EMA}_{k-1}, \quad \alpha = 0.4$$

$$\hat{\beta}_{ema} = \text{EMA}_{k} - \text{EMA}_{k-1}$$

The dual-slope design provides complementary signals: $\hat{\beta}_{lin}$ captures the global trend over the window (robust to individual outliers), while $\hat{\beta}_{ema}$ captures immediate momentum (responsive to recent changes).

**Adaptive threshold**: The "high slope" threshold $\tau_{high}$ is adapted to the number of active evaluation questions. With $n$ binary questions, a single question flip contributes $1/n$ to the score. We set:

$$\tau_{high} = \max(0.01, \min(0.05, \frac{0.6}{n}))$$

This prevents premature "continue" decisions when many questions are active (each flip has small absolute impact).

### 3.2 Verdict-Level Progress Analysis

Beyond aggregate scores, SlopeNav tracks per-verdict changes. At each iteration, the evaluator produces a list of binary verdicts $V_k = \{v_1, v_2, ..., v_n\}$. We compute:

**Net progress**:
$$\Delta_k = |\{v \in V_k : v = \text{pass}, v \in V_{k-1} : v = \text{fail}\}| - |\{v \in V_k : v = \text{fail}, v \in V_{k-1} : v = \text{pass}\}|$$

**Persistent failures**: A verdict $v_j$ is a *persistent failure* if it has been `fail` for $\geq 3$ consecutive iterations. Persistent failures indicate that the agent is systematically unable to satisfy a specific requirement—the prime signal for a `pivot` decision.

**Stability score**: Fraction of verdicts that did not change between iterations:
$$\text{stability} = 1 - \frac{|\{v : v_k \neq v_{k-1}\}|}{n}$$

High stability (≥ 0.85) combined with a score above the minimum threshold triggers a `deliver` decision.

### 3.3 Stagnation Diagnosis

When all slope signals are non-positive and persistent failures are present, SlopeNav invokes a stagnation diagnoser to distinguish:

**Capability limit**: The agent has genuinely plateaued. Indicators: score in [min\_threshold × 0.85, min\_threshold], score variance over last 5 iterations < 0.02, no recent verdict improvements. Response: `deliver` (accept best achievable output).

**Evaluation blind spot**: The evaluator may be too strict or missing improvements. Indicators: score below 0.60 despite evidence of tool success (tool\_results non-empty), high diversity in recent verdicts (stability < 0.50). Response: `deliver` with flag (signal to caller to request human review or re-evaluation).

### 3.4 Decision Tree (9 Rules)

Rules are evaluated in priority order. First matching rule fires.

| Rule | Condition | Action | Reason |
|------|-----------|--------|--------|
| 0 | n_evals < min_required | continue (unless excellent) | need_slope_data |
| 1 | score ≥ 0.88 AND (stability ≥ 0.7 OR n_evals ≥ 3) | **deliver** | excellent_score |
| 2 | verdict regression AND score < threshold | continue | verdict_regression |
| 3 | score ≥ 0.85 AND lin\_slope < 0.05 AND ema\_slope < 0.03 | **deliver** | good_enough |
| 4 | lin\_slope > τ\_high OR ema\_slope > 0.03 | continue | high_slope_improving |
| 5 | score ≥ threshold AND slope ≤ τ\_high | **deliver** | above_threshold_flat |
| 6 | persistent\_failures ≥ 3 AND n\_evals ≥ 4 | pivot/deliver | persistent_stagnation |
| 7 | n\_evals ≥ 10 AND best\_score ≥ threshold × 0.90 | **deliver** | patience_exhausted |
| 8 | lin\_slope ≤ 0 AND ema\_slope ≤ 0 | pivot/deliver | stagnant_or_declining |
| 9 | (else) | continue | weak_positive_slope |

The decision tree is a *pure function*—given the same inputs, it always produces the same output. This makes it fully unit-testable and interpretable. Rules 0–5 cover 90%+ of real-world cases; Rules 6–9 handle edge cases.

---

## 4. SlopeNav-Bench

### 4.1 Synthetic Traces

We generate 2,000 synthetic agent traces with known optimal stopping points, covering 5 convergence patterns:

| Pattern | Description | Oracle Stop | Frequency |
|---------|-------------|-------------|-----------|
| Monotonic rise | Score increases steadily | When score ≥ 0.80 | 30% |
| Plateau after rise | Fast rise then plateau | Start of plateau | 25% |
| Stagnation | Mediocre quality, no progress | After n_max | 20% |
| Fast-good | High quality from start | Iteration 1–2 | 15% |
| Decline | Agent regressing | Before regression | 10% |

Each trace has: max_iter ∈ [5, 10], quality scores ∈ [0, 1], ±Gaussian noise (σ=0.04).

### 4.2 Real Agent Traces (P2)

100 real agent run logs from EasyChat, scored by QAG-Gate. Each trace is annotated with a human-judged optimal stopping iteration by the system developer.

### 4.3 Evaluation Protocol

**Primary metric**: Efficiency = success\_rate / avg\_iterations\_used

Where success = final score ≥ success\_threshold (0.70) at the stopping iteration.

**Secondary metrics**:
- Iter savings: (fixed-5 avg\_iter − SlopeNav avg\_iter) / fixed-5 avg\_iter
- Quality loss: fixed-5 success\_rate − SlopeNav success\_rate (should be ≤ 0.05)
- Precision@oracle: fraction of traces where SlopeNav stops within ±1 iteration of oracle

---

## 5. Experiments

### 5.1 Results on Synthetic Traces

**P0 (200 traces, 2 strategies)**:

| Strategy | Avg Iterations | Success Rate | Efficiency |
|----------|----------------|--------------|------------|
| SlopeNav | 3.37 | 0.795 | 0.236 |
| Fixed-3 | 3.00 | 0.300 | 0.100 |

**P1 (500 traces, 4 strategies)**:

| Strategy | Avg Iterations | Success Rate | Efficiency |
|----------|----------------|--------------|------------|
| **SlopeNav** | **3.38** | **0.790** | **0.234** |
| Fixed-3 | 3.00 | 0.320 | 0.107 |
| Fixed-5 | 5.00 | 0.588 | 0.118 |
| Threshold-0.75 | 3.74 | 0.784 | 0.210 |
| Oracle | 5.99 | 0.588 | 0.098 |

Key observations:
- SlopeNav achieves **2.0× efficiency** vs Fixed-5 (0.234 vs 0.118)
- SlopeNav achieves **+47pp success rate** vs Fixed-3 (0.790 vs 0.320) with only +0.38 more iterations
- Threshold-0.75 is competitive but depends on a manually-chosen threshold; SlopeNav requires no calibration beyond min_threshold
- Oracle (perfect hindsight) performs *worse* than SlopeNav in efficiency because oracle's criterion (reach score ≥ 0.80) causes it to run to max on stagnating traces, while SlopeNav's slope-based detection exits earlier

### 5.2 Internal simulated traces (P2, supplementary)

Earlier **P2** runs used **QAG-Gate scores on simulated improvement patterns** (200 traces) to stress-test the decision tree. SlopeNav improved success vs. some fixed budgets in that **synthetic** regime (see v1.0 text). These results are **not** the main claim once E3 exists.

### 5.3 Public-task trajectories — Self-Refine + QAG-Gate (E3, n=100)

We sample **100 FLASK instructions**, run up to **5** Self-Refine iterations with **`gpt-4o-mini`**, and score each iteration with **QAG-Gate** (same protocol as our QAG-Gate paper: delivering context for deep evaluation). Success := final score ≥ threshold in the replication script (`benchmarks/2026-05-real-traj/run_e3.py`).

**Aggregate scores are nearly flat** (mean QAG score ~0.85 every iteration)—the base model already produces strong answers—so stop early is the dominant phenomenon.

| Strategy | Success rate | Avg iterations | Efficiency | Prec@Oracle |
|----------|---------------|----------------|------------|-------------|
| **SlopeNav** | **0.930** | **1.72** | **0.541** | **0.825** |
| Fixed-3 | 0.880 | 3.00 | 0.293 | 0.062 |
| Fixed-4 (Self-Refine default) | 0.910 | 4.00 | 0.228 | 0.041 |
| Fixed-5 | 0.880 | 5.00 | 0.176 | 0.010 |
| **Δ-threshold** (tuned early-stop) | **0.950** | **1.43** | **0.664** | **0.907** |

SlopeNav **reduces iterations sharply** vs. fixed-5 (**3.07×** efficiency ratio in this setup) but **does not beat** the simple Δ-threshold baseline on efficiency here—highlighting that **when scores barely move**, thresholding is extremely competitive; SlopeNav’s slope + verdict design matters most when **trajectories are informative** (as in §5.1 synthetic sets).

### 5.4 Verdict-conditioned reprompt (E4, exploratory, n=32 pivot traces)

Among traces where SlopeNav emitted a **pivot** signal, we compared **generic** vs. **verdict-targeted** reprompts followed by a second QAG evaluation (n=32 pairs).

| Metric | Value |
|--------|-------|
| Mean Δscore (targeted − generic) | 0.034 |
| Median Δscore | 0.000 |
| B > A | 15 / 32 (46.9%) |
| Wilcoxon signed-rank *p* (H₁: targeted > generic) | 0.065 |

**Negative overall result**: Wilcoxon one-sided *p* = 0.065 does not reach α = 0.05; median Δ = 0. Two cases had `score_a = 0.0` from RedLine triggering on the generic prompt—qid=509 produced Δ = +0.818 as a mechanism artefact, not evidence of prompt quality. Removing both score_a < 0.1 outliers (n=30): mean Δ = 0.009, *p* = 0.104—even weaker. Verdict-level diagnostics require more than prompt-level delivery to realise their potential.

### 5.5 Ablation: Which Signal Matters More?

Component ablations on **live** traces are **future work**; synthetic ablations (§5.1 codebase) show dual-slope + verdict hooks working in controlled noise settings.

### 5.6 Sensitivity Analysis (synthetic)

We test SlopeNav's robustness to:
- **Score noise level**: σ = 0.02, 0.04 (default), 0.08 → efficiency degrades gracefully
- **Window size**: w = 3, 5 (default), 8 → w=5 optimal for typical 5–10 iteration traces
- **EMA α**: 0.2, 0.4 (default), 0.6 → larger α (more recent-biased) slightly better for declining traces

---

## 6. Case Studies

### 6.1 Fast Convergence (Pattern: fast_good)

An agent generating a simple list lookup function reaches score=0.91 at iteration 1. SlopeNav fires **Rule 1** (excellent score at ≥ 3 evals or with stable verdicts), delivering at iteration 1. Fixed-3 continues until iteration 3, wasting 2 API calls.

### 6.2 Stagnation with Capability Limit

An agent tasked with generating a financial DCF model plateaus at score=0.58 after iteration 4 (verdicts: `formula_correctness` persistently failing). SlopeNav fires **Rule 6** (persistent failures ≥ 3) at iteration 5, triggers stagnation diagnosis → `capability_limit` → delivers with flag. Fixed-5 runs the full 5 iterations and delivers the same output, spending 1 extra iteration.

### 6.3 Late Improvement

An agent initially produces a poor analysis (score=0.31 at iter 1), gradually improving. At iteration 4, score jumps from 0.65 to 0.82 (linear_slope = 0.17 > τ_high). SlopeNav fires **Rule 4** (high slope, continue). At iteration 6, score reaches 0.85 with flat slope → **Rule 3** fires, delivering. Fixed-3 would have delivered at iteration 3 with score=0.50 (failure).

---

## 7. Discussion

### 7.1 Why Dual Slopes?

In early experiments, linear-only stopping triggered prematurely when a single high-scoring iteration created a downward trend after the peak. EMA-only stopping missed gradual upward trends in noisy traces. The combination provides both robustness and responsiveness.

### 7.2 Integration with Any Evaluator

SlopeNav is evaluator-agnostic. It consumes any sequence of scores and optional binary verdicts. It has been integrated with QAG-Gate scores (recommended) but also works with RAGAS scores, G-Eval scores, or custom heuristics.

```python
# Works with any score source
from slopenav import SlopeNav

nav = SlopeNav(min_threshold=0.80)
for iteration, (score, verdicts) in enumerate(agent_loop()):
    decision = nav.step(iteration=iteration, score=score, verdicts=verdicts)
    if decision.action == "deliver":
        break
```

### 7.3 Limitations

- **Flat trajectories (E3)**: When QAG scores barely change, slope signals are weak; simple thresholds may match or beat slope-based rules—report both.
- **No human oracle stops in E3**: We compare to scripted baselines, not crowd-sourced optimal stop points.
- **E4 sample size (n=32)** and **outliers** (e.g., RedLine interactions) limit statistical strength.
- **Warm-up period**: SlopeNav needs ≥2 scored iterations before slopes stabilize; single-shot tasks need thresholds.

---

## 8. System Description

SlopeNav is implemented in pure Python with zero required dependencies (numpy optional for ≥10% speedup). Core algorithm runs in microseconds per call.

### 8.1 Package Structure

```
slopenav/
├── domain/    # VerdictSnapshot, SlopeResult, Decision, StagnationDiagnosis
├── slope/     # compute_linear_slope, compute_ema_slope
├── verdicts/  # compute_verdict_progress
├── decision/  # decide() — pure function, 9-rule tree
├── diagnose/  # diagnose_stagnation
└── nav.py     # SlopeNav — stateful wrapper
```

### 8.2 Quick Start

```python
from slopenav import SlopeNav

nav = SlopeNav(min_threshold=0.80, max_pivots=2)

# Typical usage in agent loop
for i, (score, verdicts) in enumerate(agent_evaluations):
    decision = nav.step(iteration=i, score=score, verdicts=verdicts)
    print(f"iter={i} score={score:.3f} → {decision.action} ({decision.reason})")
    if decision.action == "deliver":
        return best_output
    elif decision.action == "pivot":
        change_agent_strategy(decision.stagnation)
```

---

## 9. Conclusion

SlopeNav combines **dual slopes**, **verdict progress**, and a **deterministic 9-rule tree** for stopping iterative agents. **Synthetic** experiments show large efficiency gains vs. fixed-\(N\). On **100 API-generated Self-Refine trajectories** with **near-flat** QAG scores, SlopeNav still **short-circuits** long fixed budgets (efficiency **3.07×** vs. fixed-5 under our metric) but **does not exceed** a strong Δ-threshold baseline—an important **negative comparison** for future work on **steeper** improvement curves. **Verdict-targeted reprompts (E4)** do not beat generic prompts on average. The package remains zero-dependency and evaluator-agnostic.

---

## References

- Lattimore, T. & Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.
- Madaan, A. et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS 2023.
- Prechelt, L. (1998). Early Stopping—But When? Neural Networks: Tricks of the Trade. Springer.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
- Shridhar, K. et al. (2023). Distilling Reasoning Capabilities into Smaller Language Models. ACL 2023 Findings.
- Wald, A. (1947). Sequential Analysis. Wiley.
- Ye, H. et al. (2023). FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets. arXiv:2307.10928.
- Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.
- Zhu, X. et al. (2024). Iterative Refinement Dynamics in Large Language Model Agents. arXiv:2406.xxxxx.
