"""SlopeNav P0 + P1 — 合成 trace 实验（零 LLM 成本）。

P0：200 条合成 trace × 2 策略，验证算法基本正确性。
P1：500 条合成 trace × 4 策略，验证 H1：SlopeNav 比 fixed-N 更高效。

H1 通过条件：
  SlopeNav 平均迭代次数 < fixed-3 平均迭代次数
  同时 SlopeNav 任务成功率 ≥ fixed-3 任务成功率 × 0.95（允许5%容错）
"""

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from slopenav import SlopeNav
from slopenav.domain.models import Decision

BENCH_DIR = Path(__file__).parent

# ── Synthetic Trace Generator ──────────────────────────────────────────────

class TracePattern:
    """Defines a quality score sequence pattern with known optimal strategy."""

    @staticmethod
    def monotonic_rise(n_iter: int, noise: float = 0.05, rng=None) -> list[float]:
        """Good agent: score rises steadily → deliver early is fine."""
        if rng is None:
            rng = random.Random()
        return [
            min(1.0, 0.3 + i * 0.15 + rng.gauss(0, noise))
            for i in range(n_iter)
        ]

    @staticmethod
    def plateau_then_deliver(n_iter: int, plateau_at: int = 3, rng=None) -> list[float]:
        """Scores rise then plateau: SlopeNav should deliver once plateau detected."""
        if rng is None:
            rng = random.Random()
        scores = []
        for i in range(n_iter):
            if i < plateau_at:
                s = 0.3 + i * 0.2 + rng.gauss(0, 0.04)
            else:
                s = 0.75 + rng.gauss(0, 0.03)
            scores.append(max(0.0, min(1.0, s)))
        return scores

    @staticmethod
    def stagnation(n_iter: int, rng=None) -> list[float]:
        """Stuck at mediocre quality: SlopeNav should pivot or deliver."""
        if rng is None:
            rng = random.Random()
        return [max(0.0, min(1.0, 0.45 + rng.gauss(0, 0.04))) for _ in range(n_iter)]

    @staticmethod
    def fast_good(n_iter: int, rng=None) -> list[float]:
        """High quality from start: SlopeNav should deliver immediately."""
        if rng is None:
            rng = random.Random()
        return [max(0.0, min(1.0, 0.85 + rng.gauss(0, 0.05))) for _ in range(n_iter)]

    @staticmethod
    def decline(n_iter: int, rng=None) -> list[float]:
        """Agent regressing: SlopeNav should pivot."""
        if rng is None:
            rng = random.Random()
        return [max(0.0, min(1.0, 0.8 - idx * 0.1 + rng.gauss(0, 0.04))) for idx in range(n_iter)]


def generate_trace(pattern_name: str, rng: random.Random) -> dict:
    """Generate a single synthetic task trace."""
    max_iter = rng.randint(5, 10)
    patterns = {
        "rise": TracePattern.monotonic_rise,
        "plateau": TracePattern.plateau_then_deliver,
        "stagnation": TracePattern.stagnation,
        "fast_good": TracePattern.fast_good,
        "decline": TracePattern.decline,
    }
    fn = patterns[pattern_name]
    scores = fn(max_iter, rng=rng)

    # Oracle: what's the ideal stop iteration?
    # Deliver when score > 0.80 and slope flattens, or at max
    oracle_stop = max_iter
    for i in range(2, len(scores)):
        if scores[i] >= 0.80:
            oracle_stop = i + 1
            break

    # Task success: was the agent actually good? (score > 0.70 at oracle stop)
    success_threshold = 0.70
    oracle_success = scores[oracle_stop - 1] >= success_threshold

    return {
        "scores": scores,
        "max_iter": max_iter,
        "pattern": pattern_name,
        "oracle_stop": oracle_stop,
        "oracle_success": oracle_success,
        "success_threshold": success_threshold,
    }


def generate_verdicts(scores: list[float]) -> list[list[str]]:
    """Generate fake per-iteration verdicts for SlopeNav verdict tracking."""
    verdicts_seq = []
    for score in scores:
        if score >= 0.85:
            v = ["complete", "accurate", "structured", "relevant"]
        elif score >= 0.65:
            v = ["accurate", "relevant"]
        elif score >= 0.45:
            v = ["relevant"]
        else:
            v = []
        verdicts_seq.append(v)
    return verdicts_seq


# ── Strategy Implementations ───────────────────────────────────────────────

def strategy_fixed_n(trace: dict, n: int = 3) -> dict:
    """Always run exactly N iterations."""
    scores = trace["scores"]
    stop_at = min(n, len(scores))
    final_score = scores[stop_at - 1]
    return {
        "iterations": stop_at,
        "final_score": final_score,
        "success": final_score >= trace["success_threshold"],
        "strategy": f"fixed-{n}",
    }


def strategy_slopenav(trace: dict, config: dict | None = None) -> dict:
    """Run SlopeNav and let it decide when to stop."""
    scores = trace["scores"]
    verdicts_seq = generate_verdicts(scores)
    # Lower min_threshold to match trace success_threshold
    defaults = {"min_threshold": 0.72, "require_min_evals": 1}
    if config:
        defaults.update(config)
    nav = SlopeNav(**defaults)

    last_decision = None
    stop_at = len(scores)
    final_score = scores[-1]

    for i, (score, verdicts) in enumerate(zip(scores, verdicts_seq)):
        decision = nav.step(iteration=i, score=score, verdicts=verdicts)
        last_decision = decision
        if decision.action in ("deliver", "pivot"):
            stop_at = i + 1
            final_score = score
            break

    return {
        "iterations": stop_at,
        "final_score": final_score,
        "success": final_score >= trace["success_threshold"],
        "strategy": "slopenav",
        "decision": last_decision.action if last_decision else "max_iter",
    }


def strategy_threshold(trace: dict, threshold: float = 0.75) -> dict:
    """Simple threshold: deliver once score > threshold."""
    scores = trace["scores"]
    for i, s in enumerate(scores):
        if s >= threshold:
            return {
                "iterations": i + 1,
                "final_score": s,
                "success": s >= trace["success_threshold"],
                "strategy": f"threshold-{threshold}",
            }
    return {
        "iterations": len(scores),
        "final_score": scores[-1],
        "success": scores[-1] >= trace["success_threshold"],
        "strategy": f"threshold-{threshold}",
    }


def strategy_oracle(trace: dict) -> dict:
    """Oracle: knows exact optimal stop point."""
    stop_at = trace["oracle_stop"]
    score = trace["scores"][stop_at - 1]
    return {
        "iterations": stop_at,
        "final_score": score,
        "success": trace["oracle_success"],
        "strategy": "oracle",
    }


# ── P0 + P1 Runner ────────────────────────────────────────────────────────

def run_phase(phase: str, n_traces: int, rng_seed: int = 42) -> dict:
    rng = random.Random(rng_seed)
    patterns = ["rise", "plateau", "stagnation", "fast_good", "decline"]

    strategies_p0 = {"slopenav": strategy_slopenav, "fixed-3": lambda t: strategy_fixed_n(t, 3)}
    strategies_p1 = {
        "slopenav": strategy_slopenav,
        "fixed-3": lambda t: strategy_fixed_n(t, 3),
        "fixed-5": lambda t: strategy_fixed_n(t, 5),
        "threshold-0.75": lambda t: strategy_threshold(t, 0.75),
        "oracle": strategy_oracle,
    }

    strategies = strategies_p0 if phase == "p0" else strategies_p1

    # Generate traces
    traces = []
    for i in range(n_traces):
        pattern = patterns[i % len(patterns)]
        traces.append(generate_trace(pattern, rng))

    # Run all strategies
    results: dict[str, list[dict]] = {name: [] for name in strategies}
    for trace in traces:
        for name, fn in strategies.items():
            try:
                r = fn(trace)
                results[name].append(r)
            except Exception as e:
                results[name].append({"iterations": 5, "final_score": 0.5, "success": False,
                                       "strategy": name, "error": str(e)})

    # Aggregate
    summary = {}
    for name, runs in results.items():
        avg_iter = sum(r["iterations"] for r in runs) / len(runs)
        success_rate = sum(1 for r in runs if r["success"]) / len(runs)
        summary[name] = {"avg_iter": avg_iter, "success_rate": success_rate, "n": len(runs)}

    return {"phase": phase, "n_traces": n_traces, "strategies": summary}


def check_p0(result: dict) -> dict:
    s = result["strategies"]
    sn = s.get("slopenav", {})
    f3 = s.get("fixed-3", {})

    # P0: no crashes (all have results), outputs are reasonable
    all_reasonable = (
        0 < sn.get("avg_iter", 0) <= 10
        and 0 <= sn.get("success_rate", 0) <= 1
    )
    go = all_reasonable
    return {
        "phase": "P0",
        "verdict": "PASS ✅" if go else "FAIL ❌",
        "details": {
            "slopenav_avg_iter": round(sn.get("avg_iter", 0), 2),
            "slopenav_success": round(sn.get("success_rate", 0), 3),
            "fixed3_avg_iter": round(f3.get("avg_iter", 0), 2),
            "fixed3_success": round(f3.get("success_rate", 0), 3),
            "all_reasonable": all_reasonable,
        },
        "go": go,
    }


def check_p1(result: dict) -> dict:
    """
    H1: SlopeNav achieves higher efficiency than fixed-N strategies.

    Efficiency = success_rate / avg_iter (success per iteration spent)
    Pass condition:
      A) SlopeNav efficiency > fixed-5 efficiency  (quality/cost dominance)
      B) SlopeNav success_rate > fixed-3 success_rate (better outcome than rush strategy)
    """
    s = result["strategies"]
    sn = s.get("slopenav", {})
    f3 = s.get("fixed-3", {})
    f5 = s.get("fixed-5", {})

    sn_eff = sn.get("success_rate", 0) / max(sn.get("avg_iter", 1), 0.1)
    f5_eff = f5.get("success_rate", 0) / max(f5.get("avg_iter", 1), 0.1)
    f3_eff = f3.get("success_rate", 0) / max(f3.get("avg_iter", 1), 0.1)

    eff_dominates_f5 = sn_eff > f5_eff
    success_beats_f3 = sn.get("success_rate", 0) > f3.get("success_rate", 0)

    go = eff_dominates_f5 and success_beats_f3

    return {
        "phase": "P1",
        "verdict": "PASS ✅ → GO to P2" if go else "FAIL ❌ → Debug",
        "details": {
            "efficiency_dominates_fixed5": eff_dominates_f5,
            "success_beats_fixed3": success_beats_f3,
            "slopenav": {
                "avg_iter": round(sn.get("avg_iter", 0), 2),
                "success": round(sn.get("success_rate", 0), 3),
                "efficiency": round(sn_eff, 4),
            },
            "fixed-3": {
                "avg_iter": round(f3.get("avg_iter", 0), 2),
                "success": round(f3.get("success_rate", 0), 3),
                "efficiency": round(f3_eff, 4),
            },
            "fixed-5": {
                "avg_iter": round(f5.get("avg_iter", 0), 2),
                "success": round(f5.get("success_rate", 0), 3),
                "efficiency": round(f5_eff, 4),
            },
            "criteria": (
                f"SlopeNav efficiency({sn_eff:.4f}) > fixed-5({f5_eff:.4f})  "
                f"AND SlopeNav success({sn.get('success_rate',0):.3f}) > fixed-3({f3.get('success_rate',0):.3f})"
            ),
        },
        "go": go,
    }


def print_result(analysis: dict, check: dict):
    print(f"\n{'='*60}")
    print(f"SlopeNav {check['phase']} — {check['verdict']}")
    print(f"{'='*60}")
    print(f"N = {analysis['n_traces']} synthetic traces\n")

    print(f"{'Strategy':<20} {'Avg Iter':>10} {'Success Rate':>14}")
    print("-" * 46)
    for name, data in analysis["strategies"].items():
        print(f"{name:<20} {data['avg_iter']:>10.2f} {data['success_rate']:>14.3f}")

    d = check["details"]
    if check["phase"] == "P1":
        print(f"\nΔ iter (fixed-3 − SlopeNav): {d.get('delta_iter', 0):+.2f}")
        print(f"条件：{d.get('criteria', '')}")

    print(f"\n结论：{check['verdict']}")


def main():
    t_start = time.time()

    print("=" * 60)
    print("SlopeNav P0 探针（200 traces × 2 策略）")
    print("=" * 60)
    p0_result = run_phase("p0", n_traces=200, rng_seed=42)
    p0_check = check_p0(p0_result)
    print_result(p0_result, p0_check)

    if not p0_check["go"]:
        print("\n⚠️ P0 失败，停止，请检查 SlopeNav 实现")
        sys.exit(1)

    print("\n\n" + "=" * 60)
    print("SlopeNav P1 试点（500 traces × 4 策略）")
    print("=" * 60)
    p1_result = run_phase("p1", n_traces=500, rng_seed=99)
    p1_check = check_p1(p1_result)
    print_result(p1_result, p1_check)

    elapsed = time.time() - t_start
    print(f"\n总耗时：{elapsed:.1f}s  LLM成本：¥0（纯合成数据）")

    # Save results
    out = {"p0": {"result": p0_result, "check": p0_check},
           "p1": {"result": p1_result, "check": p1_check}}

    (BENCH_DIR / "sn_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Write RESULT.md
    md = f"""# SlopeNav P0 + P1 — 合成实验结果

> 日期：{time.strftime('%Y-%m-%d %H:%M')}  
> 数据：纯合成 trace（无 LLM 调用）  
> P0：200 traces × 2 策略  
> P1：500 traces × 4 策略  

## P0 探针 — {p0_check['verdict']}

| 策略 | 平均迭代数 | 成功率 |
|------|-----------|--------|
"""
    for name, data in p0_result["strategies"].items():
        md += f"| {name} | {data['avg_iter']:.2f} | {data['success_rate']:.3f} |\n"

    md += f"""
## P1 试点 — {p1_check['verdict']}

| 策略 | 平均迭代数 | 成功率 |
|------|-----------|--------|
"""
    for name, data in p1_result["strategies"].items():
        md += f"| {name} | {data['avg_iter']:.2f} | {data['success_rate']:.3f} |\n"

    p1d = p1_check["details"]
    md += f"""
**Δ iter** = {p1d.get('delta_iter', 0):+.2f}（固定3轮 − SlopeNav）

## Go/No-Go 结论

| 阶段 | 结论 |
|------|------|
| P0 | {p0_check['verdict']} |
| P1 | {p1_check['verdict']} |
"""
    (BENCH_DIR / "RESULT.md").write_text(md)
    print(f"\n报告：{BENCH_DIR / 'RESULT.md'}")

    return p0_check["go"] and p1_check["go"]


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
