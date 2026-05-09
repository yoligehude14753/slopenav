"""
E3 第一步：生成真实迭代轨迹

流程：
  1. 从 FLASK E1 样本中取 100 条 instruction-following 任务
  2. 对每条任务，用 GPT-3.5-turbo Self-Refine 迭代 5 轮：
     - iter 0: 直接生成初始回答
     - iter 1-4: 基于上一轮回答 + LLM 自我反馈生成新回答
  3. 每轮回答用 QAG-Gate 打分
  4. 保存轨迹：question_id, instruction, [(iter, response, qag_score), ...]

输出：
  data/trajectories.jsonl  （100 条轨迹）
  data/traj_stats.json

预算：
  - Agent:  100 × 5 × ¥0.05 ≈ ¥25
  - QAG:    500 × ¥0.03     ≈ ¥15
  总计 ≈ ¥40
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
WORKSPACE = ROOT.parents[4]  # /Desktop/all
sys.path.insert(0, str(WORKSPACE / "openall/projects/qag-gate/src"))

E1_DATA     = WORKSPACE / "openall/projects/qag-gate/benchmarks/2026-05-flask-eval/data/e1_samples.jsonl"
DATA_DIR    = ROOT / "data"
CACHE_DIR   = ROOT / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_TASKS     = 100
N_ITERS     = 5
QAG_CONTEXT = {"agent_state": "delivering", "iteration": 3}

SELF_REFINE_FEEDBACK_PROMPT = """\
You are a helpful critic. Review the following response to the given task and \
provide concise, actionable feedback on what to improve.

Task: {instruction}

Response:
{response}

Feedback (2-4 bullet points, specific and actionable):"""

SELF_REFINE_IMPROVE_PROMPT = """\
Task: {instruction}

Previous response:
{response}

Feedback to address:
{feedback}

Improved response (directly addressing the feedback):"""


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


def load_e1_samples(n: int) -> list[dict]:
    """取 E1 样本中质量均衡的 n 条（instruction-following 任务）。"""
    samples = []
    with open(E1_DATA) as f:
        for line in f:
            samples.append(json.loads(line))

    # 均衡采样：low/mid/high 各约 1/3
    from collections import defaultdict
    by_tier = defaultdict(list)
    for s in samples:
        by_tier[s["quality_tier"]].append(s)

    n_per_tier = n // 3
    selected = []
    for tier in ["low", "mid", "high"]:
        selected.extend(by_tier[tier][:n_per_tier])
    # 补足
    remaining = n - len(selected)
    all_remaining = [s for s in samples if s not in selected]
    selected.extend(all_remaining[:remaining])
    return selected[:n]


async def generate_initial_response(llm, instruction: str) -> str:
    """iter 0：直接生成初始回答。"""
    try:
        return await llm.complete(
            "You are a helpful assistant. Respond to the task clearly and completely.",
            instruction,
            temperature=0.7, max_tokens=600,
        )
    except Exception as e:
        return f"[ERROR: {e}]"


async def self_refine_step(llm, instruction: str, prev_response: str) -> str:
    """iter 1+：生成反馈并改进。"""
    try:
        feedback = await llm.complete(
            "You are a helpful critic reviewing a response.",
            SELF_REFINE_FEEDBACK_PROMPT.format(
                instruction=instruction, response=prev_response),
            temperature=0.3, max_tokens=300,
        )
        improved = await llm.complete(
            "You are a helpful assistant improving a previous response based on feedback.",
            SELF_REFINE_IMPROVE_PROMPT.format(
                instruction=instruction, response=prev_response, feedback=feedback),
            temperature=0.7, max_tokens=600,
        )
        return improved
    except Exception as e:
        return f"[ERROR: {e}]"


async def score_with_qag(evaluator, instruction: str, response: str) -> float | None:
    if not response or response.startswith("[ERROR"):
        return None
    try:
        r = await evaluator.evaluate(
            content=response, task=instruction, context=QAG_CONTEXT)
        return r.score
    except Exception as e:
        print(f"  QAG score error: {e}")
        return None


async def generate_trajectory(
    sample: dict,
    llm,
    evaluator,
    cache: dict,
    cache_fh,
    concurrency_sem: asyncio.Semaphore,
) -> dict:
    """为单个任务生成完整 5 轮轨迹。"""
    qid  = sample["question_id"]
    inst = sample["instruction"]

    # 检查缓存（按 question_id）
    if qid in cache:
        return cache[qid]

    async with concurrency_sem:
        steps = []
        prev_response = None

        for i in range(N_ITERS):
            if i == 0:
                resp = await generate_initial_response(llm, inst)
            else:
                resp = await self_refine_step(llm, inst, prev_response)

            score = await score_with_qag(evaluator, inst, resp)
            steps.append({
                "iteration": i,
                "response":  resp[:800],  # 截断节省存储
                "qag_score": score,
            })
            prev_response = resp
            print(f"    qid={qid:4d}  iter={i}  score={score:.3f}" if score else
                  f"    qid={qid:4d}  iter={i}  score=None", flush=True)

        result = {
            "question_id": qid,
            "instruction": inst,
            "quality_tier": sample["quality_tier"],
            "flask_avg": sample["flask_avg"],
            "steps": steps,
        }
        cache[qid] = result
        cache_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        cache_fh.flush()
        return result


async def main(args):
    print("=== E3: 生成 Self-Refine 迭代轨迹 ===\n")

    api_key, base_url = load_api_config()

    from qag_gate import QAGEvaluator
    from qag_gate.infrastructure import OpenAIAdapter

    # GPT-3.5 做 Self-Refine（便宜）
    llm_agent = OpenAIAdapter(
        api_key=api_key, base_url=base_url, model="gpt-4o-mini")
    # GPT-4o-mini 做 QAG 评分
    llm_eval  = OpenAIAdapter(
        api_key=api_key, base_url=base_url, model="gpt-4o-mini")
    evaluator = QAGEvaluator(llm_client=llm_eval)

    n_tasks = 10 if args.dry_run else N_TASKS
    samples = load_e1_samples(n_tasks)
    print(f"任务数: {len(samples)}  ({'DRY-RUN' if args.dry_run else '全量'})\n")

    suffix   = "_dry" if args.dry_run else "_full"
    traj_file = DATA_DIR / f"trajectories{suffix}.jsonl"

    # 加载已有缓存
    cache = {}
    if traj_file.exists():
        for line in traj_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["question_id"]] = r
    print(f"已有轨迹: {len(cache)}\n")

    todo = [s for s in samples if s["question_id"] not in cache]
    print(f"待生成: {len(todo)} 条\n")

    sem = asyncio.Semaphore(2)  # 同时跑 2 个任务（控制 API 速率）
    fh  = open(traj_file, "a")

    done  = 0
    tasks_coros = [
        generate_trajectory(s, llm_agent, evaluator, cache, fh, sem)
        for s in todo
    ]
    trajectories = list(cache.values())  # 已有缓存的

    for batch_start in range(0, len(tasks_coros), 4):
        batch = tasks_coros[batch_start:batch_start + 4]
        new_results = await asyncio.gather(*batch)
        trajectories.extend(new_results)
        done += len(batch)
        print(f"\n[进度] {done}/{len(todo)} 任务完成\n", flush=True)

    fh.close()

    # 统计
    valid = [t for t in trajectories if t.get("steps")]
    avg_iter_scores = {
        f"iter_{i}": sum(
            t["steps"][i]["qag_score"] for t in valid
            if i < len(t["steps"]) and t["steps"][i]["qag_score"] is not None
        ) / max(1, sum(
            1 for t in valid
            if i < len(t["steps"]) and t["steps"][i]["qag_score"] is not None
        ))
        for i in range(N_ITERS)
    }

    print("\n" + "=" * 55)
    print("  E3 轨迹生成结果")
    print("=" * 55)
    print(f"  生成轨迹总数: {len(valid)}")
    print(f"  每轮平均 QAG 分（应递增）:")
    for k, v in avg_iter_scores.items():
        print(f"    {k}: {v:.4f}")
    print("=" * 55)

    stats = {"n_tasks": len(valid), "avg_iter_scores": avg_iter_scores}
    (DATA_DIR / f"traj_stats{suffix}.json").write_text(
        json.dumps(stats, indent=2))
    print(f"\n[Trajectories saved to {traj_file}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
