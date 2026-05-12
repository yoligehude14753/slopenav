"""04_pivot_detection.py — Detect stagnation and emit a pivot signal.

Purpose:
    Construct a trajectory that hovers around 0.4 for several rounds (no real
    improvement) and show that SlopeNav switches from "continue" to "pivot"
    once both slopes go non-positive. After the configured number of pivots,
    SlopeNav will instead deliver the best-seen result.

Run:
    python examples/04_pivot_detection.py

Env vars: none.
"""

from __future__ import annotations

from slopenav import SlopeNav


SCORES = [0.40, 0.42, 0.41, 0.40, 0.39, 0.41, 0.38]


def main() -> None:
    nav = SlopeNav(min_threshold=0.80, max_pivots=1)

    pivot_iter = None
    deliver_iter = None
    print(f"{'iter':<5} {'score':<7} {'action':<10} {'reason':<32} "
          f"{'lin':<8} {'ema':<8} pivots")
    print("-" * 78)
    for i, s in enumerate(SCORES):
        d = nav.step(iteration=i, score=s)
        if d.action == "pivot" and pivot_iter is None:
            pivot_iter = i
        if d.action == "deliver" and deliver_iter is None:
            deliver_iter = i
        print(
            f"{i:<5} {s:<7.3f} {d.action:<10} {d.reason:<32} "
            f"{d.slope_linear:<8.4f} {d.slope_ema:<8.4f} "
            f"{nav._pivot_count}"
        )
        if d.action == "deliver":
            break

    summary = nav.summary()
    print(
        f"\nfinal: pivot detected at iter={pivot_iter}, "
        f"deliver at iter={deliver_iter}, best={summary['best_score']:.3f}"
    )


if __name__ == "__main__":
    main()
