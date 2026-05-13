# Integrations

SlopeNav plugs into any iterative agent loop that produces a per-iteration scalar score (and optionally a list of binary verdicts). It is **evaluator-agnostic** — QAG-Gate is the recommended pairing because its verdicts unlock the persistent-failure pivot signal, but G-Eval / RAGAS / unit-test pass-rate / a custom heuristic all work.

The pattern in every framework below is the same:

1. Score the latest output with whatever evaluator you already use.
2. Call `decision = nav.step(iteration=i, score=s, verdicts=v_or_None)`.
3. Map `decision.action` to the framework's control flow:
   - `continue` → run another iteration / step
   - `pivot` → run another iteration **but switch strategy** (new system prompt, new role, new tool set, …)
   - `deliver` → break out of the loop and return the best-seen output

All examples below assume:

```bash
pip install slopenav
# plus whatever evaluator + framework you want — see each section
```

> Framework APIs change. Where an integration depends on an API that has been moving fast in 2025–2026, we pin a version and a date at the top of that section, and we deliberately stick to the smallest stable surface.

---

## 1. Self-Refine (Madaan et al. 2023)

Self-Refine ships with a fixed iteration budget (typically 4). SlopeNav replaces that fixed budget with a budget-and-trajectory-aware decision so easy tasks stop early and hard tasks aren't cut off mid-improvement.

```python
import os
from openai import OpenAI
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
import asyncio
from slopenav import SlopeNav

TASK = "Write a haiku about distributed systems."

async def main() -> None:
    client = OpenAI()
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.80, max_pivots=2)
    answer, critique = "", "Start with a first draft."
    best = ("", 0.0)
    for i in range(8):
        resp = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{TASK}\n\nPrev:\n{answer}\n\nFix: {critique}"}])
        answer = resp.choices[0].message.content or ""
        r = await evaluator.evaluate(task=TASK, content=answer, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        if r.score > best[1]:
            best = (answer, r.score)
        print(f"iter={i} score={r.score:.3f} action={d.action} reason={d.reason}")
        if d.action == "deliver":
            break
        if d.action == "pivot":
            critique = "Tried this angle, didn't work. Restart from a completely different metaphor."
        else:
            critique = "; ".join(v.reason for v in r.failed_verdicts[:3]) or "Tighten wording."
    print(f"final best={best[1]:.3f}")

asyncio.run(main())
```

---

## 2. Reflexion (Shinn et al. 2023)

Reflexion adds a reflection memory across trials. SlopeNav controls *how many* trials happen and *when* to switch strategies. On `pivot`, write the persistent failures into long-term memory and reset short-term scratchpads.

```python
import asyncio, os
from openai import AsyncOpenAI
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

TASK = "Summarise a paper for a non-expert in 4 bullets, each <= 25 words."

async def main() -> None:
    client = AsyncOpenAI()
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=2)
    long_memory: list[str] = []; short_memory: list[str] = []
    answer, best = "", ("", 0.0)
    for trial in range(6):
        sys = "Avoid past failures:\n" + "\n".join(f"- {m}" for m in long_memory + short_memory)
        resp = await client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": f"{TASK}\n\nPrev:\n{answer}"}])
        answer = resp.choices[0].message.content or ""
        r = await evaluator.evaluate(task=TASK, content=answer, context={"iteration": trial, "agent_state": "executing"})
        d = nav.step(iteration=trial, score=r.score, verdicts=r.verdicts)
        if r.score > best[1]:
            best = (answer, r.score)
        print(f"trial={trial} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            break
        if d.action == "pivot":
            long_memory.extend(p["question"] for p in (d.verdict_progress.persistent_failures if d.verdict_progress else []))
            short_memory = []
        else:
            short_memory = [v.reason for v in r.failed_verdicts[:3]]

asyncio.run(main())
```

---

## 3. Claude Code SDK

*As of 2026-05 with `claude-code-sdk==0.0.x` (Anthropic Python SDK for Claude Code). The SDK does not expose a "stop early" hook; the supported pattern is to consume the message stream, score each assistant turn yourself, and break the consumer loop when SlopeNav says `deliver`.*

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

TASK = "Add type hints to all public functions in src/utils.py"

async def main() -> None:
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=1)
    opts = ClaudeCodeOptions(max_turns=8, allowed_tools=["Read", "Edit"])
    i = 0
    async for msg in query(prompt=TASK, options=opts):
        if getattr(msg, "type", "") != "assistant":
            continue
        text = getattr(msg, "text", "") or str(msg)
        r = await evaluator.evaluate(task=TASK, content=text, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        print(f"turn={i} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            print("SlopeNav: deliver — stop consuming further turns"); break
        if d.action == "pivot":
            print("SlopeNav: pivot — caller should restart with a different role/tools")
        i += 1

asyncio.run(main())
```

Claude Code doesn't natively support mid-run "restart with new system prompt" — when `pivot` fires, the practical move is to break the current `query()`, build a new `ClaudeCodeOptions` with a different system prompt or tool set, and call `query()` again.

---

## 4. Cursor Composer / OpenAI Codex CLI

These are CLI tools, so SlopeNav controls a *meta-loop* around repeated CLI invocations:

```python
import asyncio, subprocess
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

TASK = "Refactor src/utils.py to use dataclasses"

async def main() -> None:
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=1)
    prompt = TASK
    for i in range(6):
        out = subprocess.check_output(["codex", "exec", prompt], text=True)
        r = await evaluator.evaluate(task=TASK, content=out, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        print(f"iter={i} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            break
        if d.action == "pivot":
            prompt = f"Earlier attempts went sideways. Restart from scratch on a different design.\n\n{TASK}"
        else:
            prompt = f"{TASK}\n\nLast attempt scored {r.score:.2f}; fix: " + "; ".join(v.reason for v in r.failed_verdicts[:3])

asyncio.run(main())
```

The same shape works for `cursor-agent` or any subprocess-based agent.

---

## 5. Aider

*As of 2026-05 with `aider-chat>=0.60`. Aider exposes `Coder.run()` synchronously per turn — wrap it in a SlopeNav meta-loop.*

```python
import asyncio
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

TASK = "Add a 'reverse' method to src/utils/list_helpers.py with a unit test"

async def main() -> None:
    coder = Coder.create(main_model=Model("gpt-4o-mini"),
                         io=InputOutput(yes=True), fnames=["src/utils/list_helpers.py"])
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=1)
    next_msg = TASK
    for i in range(5):
        out = coder.run(with_message=next_msg) or ""
        r = await evaluator.evaluate(task=TASK, content=out, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        print(f"iter={i} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            break
        next_msg = ("Reset: pick a different approach. " if d.action == "pivot" else "Address: ") \
                   + "; ".join(v.reason for v in r.failed_verdicts[:3])

asyncio.run(main())
```

---

## 6. AutoGen

*As of 2026-05 with `autogen-agentchat>=0.4` (the redesigned, event-driven AutoGen). Run the agent for one turn, score the last assistant message, and let SlopeNav decide whether to call `agent.run()` again.*

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

TASK = "Draft a 3-paragraph release note for v0.2.0."

async def main() -> None:
    agent = AssistantAgent(name="writer",
                           model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"))
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=1)
    prompt = TASK
    for i in range(5):
        result = await agent.run(task=prompt)
        text = str(result.messages[-1].content) if result.messages else ""
        r = await evaluator.evaluate(task=TASK, content=text, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        print(f"iter={i} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            break
        prompt = (f"Earlier draft was off-track ({r.score:.2f}). Restart with a different tone.\n{TASK}"
                  if d.action == "pivot" else f"Revise:\n{text}\n\nFix: " +
                  "; ".join(v.reason for v in r.failed_verdicts[:3]))

asyncio.run(main())
```

---

## 7. CrewAI

*As of 2026-05 with `crewai>=0.50`. `Task` accepts `callback=`, but the callback is called once per task and cannot re-run the task. For SlopeNav-controlled retries, wrap `Crew.kickoff()` in a meta-loop.*

```python
import asyncio
from crewai import Agent, Task, Crew
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

TASK_DESC = "Write a one-paragraph release note for v0.2.0."

async def main() -> None:
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=1)
    extra = ""
    for i in range(5):
        writer = Agent(role="Writer", goal="Concise release notes", backstory="Senior tech writer.", llm="gpt-4o-mini")
        task = Task(description=TASK_DESC + extra, expected_output="One paragraph.", agent=writer)
        out = Crew(agents=[writer], tasks=[task]).kickoff()
        text = getattr(out, "raw", str(out))
        r = await evaluator.evaluate(task=TASK_DESC, content=text, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        print(f"iter={i} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            break
        extra = ("\n\nRestart with a different angle." if d.action == "pivot"
                 else "\n\nFix: " + "; ".join(v.reason for v in r.failed_verdicts[:3]))

asyncio.run(main())
```

---

## 8. Letta (formerly MemGPT)

*As of 2026-05 with `letta-client>=0.5`. Send one user message per iteration, score the assistant reply, and let SlopeNav drive the loop. On `pivot` you can either change `system` via `agents.modify` or just send a "restart from scratch" user message.*

```python
import asyncio, os
from letta_client import Letta
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

async def main() -> None:
    letta = Letta(base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"))
    agent = letta.agents.create(name="writer", model="openai/gpt-4o-mini",
                                embedding="openai/text-embedding-3-small")
    evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
    nav = SlopeNav(min_threshold=0.85, max_pivots=1)
    msg = "Draft a 4-bullet pitch for a vector DB aimed at a senior PM."
    for i in range(5):
        resp = letta.agents.messages.create(agent_id=agent.id, messages=[{"role": "user", "content": msg}])
        text = "\n".join(getattr(m, "content", "") for m in resp.messages
                         if getattr(m, "message_type", "") == "assistant_message")
        r = await evaluator.evaluate(task=msg, content=text, context={"iteration": i, "agent_state": "executing"})
        d = nav.step(iteration=i, score=r.score, verdicts=r.verdicts)
        print(f"iter={i} score={r.score:.3f} action={d.action}")
        if d.action == "deliver":
            break
        msg = ("Forget the previous angle. Try a different metaphor." if d.action == "pivot"
               else "Revise. Fix: " + "; ".join(v.reason for v in r.failed_verdicts[:3]))

asyncio.run(main())
```

---

## 9. LangGraph

*As of 2026-05 with `langgraph>=0.2`. SlopeNav fits naturally as a node that emits the next route via `add_conditional_edges`.*

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from qag_gate import QAGEvaluator
from qag_gate.infrastructure import OpenAIAdapter
from slopenav import SlopeNav

class S(TypedDict):
    task: str; draft: str; iteration: int; route: str; best: tuple[str, float]

client = AsyncOpenAI()
evaluator = QAGEvaluator(OpenAIAdapter(model="gpt-4o-mini"))
nav = SlopeNav(min_threshold=0.85, max_pivots=1)

async def write(s: S) -> S:
    extra = "Restart with a different angle. " if s["route"] == "pivot" else ""
    resp = await client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "user", "content": extra + s["task"] + "\nPrev:\n" + s["draft"]}])
    return {"draft": resp.choices[0].message.content or "", "iteration": s["iteration"] + 1}

async def navigate(s: S) -> S:
    r = await evaluator.evaluate(task=s["task"], content=s["draft"],
                                 context={"iteration": s["iteration"], "agent_state": "executing"})
    d = nav.step(iteration=s["iteration"], score=r.score, verdicts=r.verdicts)
    best = max(s["best"], (s["draft"], r.score), key=lambda x: x[1])
    print(f"iter={s['iteration']} score={r.score:.3f} action={d.action}")
    return {"route": d.action, "best": best}

g = StateGraph(S)
g.add_node("write", write); g.add_node("navigate", navigate)
g.set_entry_point("write"); g.add_edge("write", "navigate")
g.add_conditional_edges("navigate", lambda s: END if s["route"] == "deliver" or s["iteration"] >= 6 else "write")
asyncio.run(g.compile().ainvoke({"task": "Explain RAG in 3 bullets.", "draft": "",
                                  "iteration": 0, "route": "", "best": ("", 0.0)}))
```

---

## 10. YOLI

Built-in. YOLI's evaluation framework already wires SlopeNav as the default stopping criterion for long-running agent loops (paired with QAG-Gate as the scorer) — see `docs/EVAL.md` in the [`yoli`](https://github.com/yoligehude14753/yoli) repo.

---

## Evaluator-agnostic example: G-Eval instead of QAG-Gate

SlopeNav consumes any `(score, verdicts_or_None)`. Here is the same Self-Refine loop, but the score comes from a tiny G-Eval-style prompt instead of QAG-Gate. SlopeNav still drives the decision.

```python
import asyncio, re
from openai import AsyncOpenAI
from slopenav import SlopeNav

TASK = "Explain RAG in 3 bullets, each <= 20 words."

async def g_eval(client: AsyncOpenAI, task: str, content: str) -> float:
    prompt = (f"Task: {task}\n\nResponse:\n{content}\n\n"
              "Rate this response from 1 to 10 on usefulness, correctness, conciseness. "
              "Output ONLY a single number.")
    resp = await client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\d+(?:\.\d+)?", resp.choices[0].message.content or "")
    return float(m.group()) / 10.0 if m else 0.0

async def main() -> None:
    client = AsyncOpenAI()
    nav = SlopeNav(min_threshold=0.80, max_pivots=1)
    draft, critique = "", "Start with a draft."
    for i in range(6):
        resp = await client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{TASK}\nPrev: {draft}\nFix: {critique}"}])
        draft = resp.choices[0].message.content or ""
        score = await g_eval(client, TASK, draft)
        d = nav.step(iteration=i, score=score)  # no verdicts — verdict-level features degrade gracefully
        print(f"iter={i} score={score:.3f} action={d.action} reason={d.reason}")
        if d.action == "deliver":
            break
        critique = "Restart with a different structure." if d.action == "pivot" else "Tighten wording, add concrete example."

asyncio.run(main())
```

Without `verdicts=`, SlopeNav silently skips Rule 2 (verdict regression) and Rule 5 (persistent failures) but keeps Rules 1, 3, 4, 6–9 — score-only trajectories still get sensible decisions.

---

## Cookbook

- **Pair with QAG-Gate when verdicts matter.** Without verdicts, SlopeNav loses the persistent-failure pivot signal; with them, "Rule 5" (≥3 iterations of the same failure category) is the cleanest signal that the agent's current strategy is wrong, not just slow.
- **Tune `min_threshold`, `max_pivots`, `window`.** Defaults (`0.80 / 2 / 5`) target Self-Refine-style ≤ 8-iteration loops. For longer budgets, raise `window` to 8–10; for stricter delivery bars, raise `min_threshold` to 0.85+.
- **Use `Decision.reason`** as a label in your trace logs — every decision has a stable, machine-readable reason (`high_slope`, `persistent_failures`, `plateau_above_threshold`, …) that survives across refactors.
- **Always keep a `best` track.** SlopeNav decides *when* to stop; *what* to return is your responsibility. Track the best-seen `(content, score)` pair separately so a late `deliver` doesn't drop a stronger earlier candidate.
