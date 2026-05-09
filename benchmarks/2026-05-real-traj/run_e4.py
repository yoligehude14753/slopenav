"""
E4：Verdict-Level Pivot 有效性验证

在 E3 轨迹中，识别出 SlopeNav 触发 pivot 的轨迹，
对 pivot 决策点随机分配两种 reprompt 策略：
  A: Generic  → "Please improve the response."
  B: Targeted → "This response has persistently failed in: {persistent_failures}. Address these specifically."

指标：pivot 后下一轮 QAG 分数提升幅度 Δscore(B - A)

门控：Δscore(B - A) > 0.05 → PASS

用法：
  python run_e4.py
  python run_e4.py --dry-run
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
WORKSPACE = ROOT.parents[4]  # /Desktop/all
sys.path.insert(0, str(WORKSPACE / "openall/projects/slopenav/src"))
sys.path.insert(0, str(WORKSPACE / "openall/projects/qag-gate/src"))

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR   = ROOT / "cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

QAG_CONTEXT = {"agent_state": "delivering", "iteration": 3}

GENERIC_REPROMPT = """\
Task: {instruction}

Previous response:
{response}

Please improve the response."""

TARGETED_REPROMPT = """\
Task: {instruction}

Previous response:
{response}

This response has persistently failed in the following areas: {failures}.
Please specifically address these weaknesses in your improved response."""


def load_api_config():
    env_file = WORKSPACE / "easychat/backend/.env"
    config = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                config[k.strip()] = v.strip()
    return (config.get("YUNWU_GPT_KEY") or config.get("OPENAI_API_KEY", ""),
            config.get("YUNWU_BASE_URL", "https://yunwu.ai/v1"))


def find_pivot_trajectories(dry_run: bool) -> list[dict]:
    """在 E3 轨迹上重跑 SlopeNav，识别触发 pivot 的轨迹和 pivot 时刻。"""
    from slopenav import SlopeNav

    suffix   = "_dry" if dry_run else "_full"
    traj_file = DATA_DIR / f"trajectories{suffix}.jsonl"
    if not traj_file.exists():
        raise FileNotFoundError(
            f"Run generate_trajectories.py {'--dry-run' if dry_run else ''} first.")

    pivot_cases = []
    with open(traj_file) as f:
        for line in f:
            traj = json.loads(line)
            nav  = SlopeNav()
            steps = traj["steps"]

            for i, s in enumerate(steps):
                score = s["qag_score"]
                if score is None:
                    continue
                decision = nav.step(iteration=i, score=score)
                action   = str(decision.action) if decision else "continue"

                if action == "pivot":
                    # 找到 pivot 点
                    # 收集 persistent_failures
                    failures = nav.get_persistent_failures() or []
                    failure_str = ", ".join(str(f) for f in failures) if failures else \
                                  "completeness and accuracy"

                    # pivot 时的当前回答
                    pivot_response = s["response"]
                    if not pivot_response.strip():
                        break

                    pivot_cases.append({
                        "question_id": traj["question_id"],
                        "instruction": traj["instruction"],
                        "flask_avg":   traj["flask_avg"],
                        "quality_tier": traj["quality_tier"],
                        "pivot_iter":  i,
                        "pivot_score": score,
                        "pivot_response": pivot_response,
                        "persistent_failures": failure_str,
                    })
                    break  # 每条轨迹只取第一个 pivot

    print(f"找到 pivot 轨迹: {len(pivot_cases)} 条")
    return pivot_cases


async def generate_reprompt_response(
    llm, instruction: str, response: str,
    strategy: str, failures: str
) -> str:
    if strategy == "generic":
        user_prompt = GENERIC_REPROMPT.format(
            instruction=instruction, response=response)
    else:
        user_prompt = TARGETED_REPROMPT.format(
            instruction=instruction, response=response, failures=failures)
    try:
        return await llm.complete(
            "You are a helpful assistant improving a previous response.",
            user_prompt,
            temperature=0.7, max_tokens=600,
        )
    except Exception as e:
        return f"[ERROR: {e}]"


async def run_pivot_case(
    case: dict,
    llm_agent,
    evaluator,
    cache: dict,
    cache_fh,
    sem: asyncio.Semaphore,
) -> dict:
    qid = case["question_id"]
    key_a = f"{qid}_generic"
    key_b = f"{qid}_targeted"

    async with sem:
        await asyncio.sleep(0.5)

        async def _score_strategy(key, strategy):
            if key in cache:
                return cache[key]
            resp = await generate_reprompt_response(
                llm_agent,
                case["instruction"],
                case["pivot_response"],
                strategy,
                case["persistent_failures"],
            )
            if resp.startswith("[ERROR"):
                return None
            score = None
            try:
                r = await evaluator.evaluate(
                    content=resp, task=case["instruction"], context=QAG_CONTEXT)
                score = r.score
            except Exception as e:
                print(f"  QAG error {key}: {e}")
            cache[key] = {"score": score, "response": resp[:400]}
            cache_fh.write(json.dumps({key: cache[key]}) + "\n")
            cache_fh.flush()
            return cache[key]

        res_a, res_b = await asyncio.gather(
            _score_strategy(key_a, "generic"),
            _score_strategy(key_b, "targeted"),
        )

        score_a = res_a["score"] if isinstance(res_a, dict) else res_a
        score_b = res_b["score"] if isinstance(res_b, dict) else res_b

        delta = (score_b - score_a) if (score_a is not None and score_b is not None) else None
        sa_str = f"{score_a:.3f}" if isinstance(score_a, float) else "N/A"
        sb_str = f"{score_b:.3f}" if isinstance(score_b, float) else "N/A"
        d_str  = f"{delta:.3f}"  if isinstance(delta,   float) else "N/A"
        print(f"  qid={qid:4d}  pivot_score={case['pivot_score']:.3f}  "
              f"A(generic)={sa_str}  B(targeted)={sb_str}  Δ={d_str}", flush=True)

        return {
            **case,
            "score_a": score_a,
            "score_b": score_b,
            "delta_b_minus_a": delta,
        }


async def main(args):
    print("=== E4: Verdict-Level Pivot 验证 ===\n")
    api_key, base_url = load_api_config()

    from qag_gate import QAGEvaluator
    from qag_gate.infrastructure import OpenAIAdapter

    llm_agent = OpenAIAdapter(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
    llm_eval  = OpenAIAdapter(api_key=api_key, base_url=base_url, model="gpt-4o-mini")
    evaluator = QAGEvaluator(llm_client=llm_eval)

    # 识别 pivot 轨迹
    pivot_cases = find_pivot_trajectories(args.dry_run)
    if not pivot_cases:
        print("⚠️  没有找到 pivot 轨迹，请检查 E3 轨迹数据")
        return

    print(f"使用 {len(pivot_cases)} 条 pivot 轨迹\n")

    suffix    = "_dry" if args.dry_run else "_full"
    cache_file = CACHE_DIR / f"e4_scores{suffix}.jsonl"
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if line.strip():
                cache.update(json.loads(line))

    sem = asyncio.Semaphore(3)
    fh  = open(cache_file, "a")

    results = []
    for bs in range(0, len(pivot_cases), 6):
        batch = pivot_cases[bs:bs + 6]
        batch_r = await asyncio.gather(*[
            run_pivot_case(c, llm_agent, evaluator, cache, fh, sem)
            for c in batch
        ])
        results.extend(batch_r)

    fh.close()

    # 分析
    valid = [r for r in results if r["delta_b_minus_a"] is not None]
    if not valid:
        print("⚠️  没有有效的 Δ 结果")
        return

    avg_delta = sum(r["delta_b_minus_a"] for r in valid) / len(valid)
    pos_count = sum(1 for r in valid if r["delta_b_minus_a"] > 0)
    go        = avg_delta > 0.05

    print("\n" + "=" * 60)
    print("  E4 Results")
    print("=" * 60)
    print(f"  有效 pivot 对数: {len(valid)}")
    print(f"  平均 Δscore(B-A): {avg_delta:.4f}  (需 > 0.05)")
    print(f"  B > A 的比例:    {pos_count}/{len(valid)} = {pos_count/len(valid):.1%}")
    verdict = "✅  GO  → Targeted reprompt 有效，强化为主要贡献" if go else \
              "❌  NO-GO → 作为 Negative Finding 写入 Analysis 节（仍 publishable）"
    print(f"\n  门控结论: {verdict}")
    print("=" * 60)

    out = {
        "n_valid": len(valid),
        "avg_delta": avg_delta,
        "pos_rate": pos_count / len(valid),
        "go": go,
    }
    (RESULTS_DIR / f"e4_analysis{suffix}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    with open(RESULTS_DIR / f"e4_rows{suffix}.jsonl", "w") as f:
        for r in results:
            row = {k: v for k, v in r.items() if k not in ("pivot_response",)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[Results saved to {RESULTS_DIR}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
