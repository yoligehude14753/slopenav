"""01_minimal_decision.py — Smallest possible SlopeNav loop.

Purpose:
    Feed a hardcoded score trajectory into SlopeNav and print the decision,
    reason, and both slopes for every iteration. Useful for understanding
    how the decision tree reacts to a clean ascending trajectory.

Run:
    python examples/01_minimal_decision.py

Env vars: none (no LLM call).
"""

from __future__ import annotations

from slopenav import SlopeNav


SCORES = [0.30, 0.50, 0.70, 0.85]


def main() -> None:
    nav = SlopeNav(min_threshold=0.80)
    final_decision = None

    print(f"{'iter':<5} {'score':<7} {'action':<10} {'reason':<30} {'lin':<8} {'ema':<8}")
    print("-" * 70)
    for i, score in enumerate(SCORES):
        d = nav.step(iteration=i, score=score)
        final_decision = d
        print(
            f"{i:<5} {score:<7.3f} {d.action:<10} {d.reason:<30} "
            f"{d.slope_linear:<8.4f} {d.slope_ema:<8.4f}"
        )

    summary = nav.summary()
    print(
        f"\nfinal score: {SCORES[-1]:.3f}, "
        f"decision: {final_decision.action}, "
        f"best_seen: {summary['best_score']:.3f}"
    )


if __name__ == "__main__":
    main()
