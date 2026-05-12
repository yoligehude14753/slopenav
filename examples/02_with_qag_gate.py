"""02_with_qag_gate.py — End-to-end OpenAI -> QAG-Gate -> SlopeNav loop.

Purpose:
    Wire up the full convergence loop: each round we ask OpenAI to refine
    an answer, score it with QAG-Gate, and let SlopeNav decide whether to
    keep iterating, pivot strategy, or deliver. Up to 5 rounds.

Run:
    pip install "qag-gate[openai]" slopenav
    OPENAI_API_KEY=sk-... python examples/02_with_qag_gate.py

Env vars:
    OPENAI_API_KEY  required
    OPENAI_BASE_URL optional
    QAG_MODEL       optional (default: gpt-4o-mini)
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter

from slopenav import SlopeNav


TASK = (
    "Write a 4-bullet pitch (each bullet ≤ 25 words) explaining what a "
    "vector database is, aimed at a senior product manager who is new to AI."
)


async def regenerate(client: AsyncOpenAI, model: str, prev: str, critique: str) -> str:
    user = (
        f"Task: {TASK}\n\nPrevious draft:\n{prev or '(none)'}\n\n"
        f"Address this critique in the next revision:\n{critique}\n\n"
        "Return only the 4 bullets."
    )
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": user}], temperature=0.3
    )
    return resp.choices[0].message.content or ""


async def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("QAG_MODEL", "gpt-4o-mini")

    gen = AsyncOpenAI(api_key=api_key, base_url=base_url)
    evaluator = QAGEvaluator(OpenAIAdapter(model=model, api_key=api_key, base_url=base_url))
    nav = SlopeNav(min_threshold=0.80, max_pivots=2)

    answer, critique = "", "Start with a first draft."
    last_i, last_score, last_decision = 0, 0.0, None

    for i in range(5):
        answer = await regenerate(gen, model, answer, critique)
        r = await evaluator.evaluate(
            task=TASK, content=answer,
            context={"iteration": i, "agent_state": "executing", "tools_used": []},
        )
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        last_i, last_score, last_decision = i, r.score, d
        print(f"iter={i} score={r.score:.3f} action={d.action} reason={d.reason}")
        if d.action == "deliver":
            break
        critique = "; ".join(v.reason for v in r.failed_verdicts[:3]) or "Tighten wording."

    print(f"\nfinal score: {last_score:.3f}, decision: {last_decision.action} at iter={last_i}")


if __name__ == "__main__":
    asyncio.run(main())
