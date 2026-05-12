"""03_threshold_baseline_comparison.py — SlopeNav vs Δ-threshold vs fixed-5.

Purpose:
    Replay the same synthetic score trajectory under three stopping rules and
    compare where each one halts:

      * SlopeNav        — dual-slope decision tree from this repo
      * delta_threshold — stop when |score[t] - score[t-1]| < eps
      * fixed_budget    — always run exactly N iterations

    Reports: stop iteration, final score, "success" (>= threshold), and a crude
    efficiency metric = final_score / iterations_used.

Run:
    python examples/03_threshold_baseline_comparison.py

Env vars: none.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from slopenav import SlopeNav


SCORES = [0.55, 0.68, 0.78, 0.83, 0.85, 0.86, 0.86, 0.87]
THRESHOLD = 0.80


def run_slopenav(scores: List[float]) -> Tuple[int, float]:
    nav = SlopeNav(min_threshold=THRESHOLD)
    for i, s in enumerate(scores):
        d = nav.step(iteration=i, score=s)
        if d.action == "deliver":
            return i, s
    return len(scores) - 1, scores[-1]


def run_delta_threshold(scores: List[float], eps: float = 0.03) -> Tuple[int, float]:
    for i in range(1, len(scores)):
        if abs(scores[i] - scores[i - 1]) < eps:
            return i, scores[i]
    return len(scores) - 1, scores[-1]


def run_fixed_budget(scores: List[float], budget: int = 5) -> Tuple[int, float]:
    idx = min(budget - 1, len(scores) - 1)
    return idx, scores[idx]


def report(name: str, strategy: Callable[[List[float]], Tuple[int, float]]) -> Tuple[str, int, float, bool, float]:
    stop_at, final_s = strategy(SCORES)
    iters = stop_at + 1
    success = final_s >= THRESHOLD
    efficiency = final_s / iters
    print(
        f"{name:<18} stop@iter={stop_at:<2} iters_used={iters} "
        f"final_score={final_s:.3f} success={success} efficiency={efficiency:.3f}"
    )
    return name, stop_at, final_s, success, efficiency


def main() -> None:
    print(f"Trajectory: {SCORES}\nThreshold:  {THRESHOLD}\n")
    rows = [
        report("SlopeNav",        run_slopenav),
        report("delta_threshold", run_delta_threshold),
        report("fixed_budget=5",  run_fixed_budget),
    ]
    winner = max(rows, key=lambda r: (r[3], r[4]))
    print(f"\nfinal: best strategy (success, then efficiency) = {winner[0]} "
          f"(score={winner[2]:.3f}, iters={winner[1] + 1})")


if __name__ == "__main__":
    main()
