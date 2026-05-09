"""
E3：SlopeNav vs Fixed-N 停止策略对比

在 generate_trajectories.py 生成的真实 Self-Refine 轨迹上，
比较 5 种停止策略的效率和准确率：
  1. SlopeNav（本方法）
  2. Fixed-3  （固定跑 3 轮）
  3. Fixed-5  （固定跑 5 轮，全量）
  4. Δ-threshold（Δscore < 0.03 时停止）
  5. Self-Refine-fixed-4（固定 4 轮，原论文配置）

指标：
  - efficiency       = success_rate / avg_iter
  - success_rate     = % 轨迹中 final_qag_score ≥ 0.70（高质量）
  - avg_iter         = 平均停止轮次
  - precision@oracle = % 轨迹内 ±1 轮停止（oracle 为首次达 0.70 轮次）

门控：SlopeNav efficiency ≥ fixed-5 efficiency × 1.20 → PASS

用法：
  python run_e3.py              # 全量
  python run_e3.py --dry-run    # 10 条轨迹
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
WORKSPACE = ROOT.parents[4]  # /Desktop/all
sys.path.insert(0, str(WORKSPACE / "openall/projects/slopenav/src"))

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUCCESS_THRESHOLD = 0.70  # QAG score ≥ 0.70 = 成功


def load_trajectories(dry_run: bool) -> list[dict]:
    suffix = "_dry" if dry_run else "_full"
    traj_file = DATA_DIR / f"trajectories{suffix}.jsonl"
    if not traj_file.exists():
        raise FileNotFoundError(
            f"未找到轨迹文件 {traj_file}\n"
            f"请先运行: python generate_trajectories.py {'--dry-run' if dry_run else ''}"
        )
    trajs = []
    with open(traj_file) as f:
        for line in f:
            trajs.append(json.loads(line))
    return trajs


# ── 停止策略 ────────────────────────────────────────────────────────────────

def apply_fixed_n(traj: dict, n: int) -> dict:
    """固定跑 n 轮（0-indexed: 实际步数 = n）。"""
    steps = traj["steps"]
    stop_iter = min(n - 1, len(steps) - 1)
    final_score = steps[stop_iter]["qag_score"]
    return {"stop_iter": stop_iter, "final_score": final_score, "iters_used": stop_iter + 1}


def apply_delta_threshold(traj: dict, delta: float = 0.03) -> dict:
    """Δscore < delta 时停止。"""
    steps = traj["steps"]
    prev_score = None
    for i, s in enumerate(steps):
        score = s["qag_score"]
        if score is None:
            continue
        if prev_score is not None and (score - prev_score) < delta:
            return {"stop_iter": i - 1, "final_score": prev_score, "iters_used": i}
        prev_score = score
    # 跑完仍未触发
    last = next((s["qag_score"] for s in reversed(steps) if s["qag_score"] is not None), None)
    return {"stop_iter": len(steps) - 1, "final_score": last, "iters_used": len(steps)}


def apply_slopenav(traj: dict) -> dict:
    """使用 SlopeNav 决定停止时机。"""
    from slopenav import SlopeNav

    nav   = SlopeNav()
    steps = traj["steps"]
    stop_iter   = len(steps) - 1
    final_score = None

    for i, s in enumerate(steps):
        score = s["qag_score"]
        if score is None:
            continue
        decision = nav.step(iteration=i, score=score)
        action   = str(decision.action) if decision else "continue"

        if action in ("deliver", "stop"):
            stop_iter   = i
            final_score = score
            break
        final_score = score

    if final_score is None:
        final_score = next(
            (s["qag_score"] for s in reversed(steps) if s["qag_score"] is not None), 0.0)

    return {
        "stop_iter":  stop_iter,
        "final_score": final_score,
        "iters_used": stop_iter + 1,
        "best_score": nav.get_best_score(),
    }


def oracle_stop(traj: dict) -> int | None:
    """首次达到 SUCCESS_THRESHOLD 的轮次，None 表示从未达到。"""
    for s in traj["steps"]:
        if s["qag_score"] is not None and s["qag_score"] >= SUCCESS_THRESHOLD:
            return s["iteration"]
    return None


# ── 评估指标 ────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    successes = [r for r in results if r["final_score"] is not None
                 and r["final_score"] >= SUCCESS_THRESHOLD]
    iters_used = [r["iters_used"] for r in results if r["iters_used"] is not None]
    success_rate = len(successes) / len(results) if results else 0.0
    avg_iter     = sum(iters_used) / len(iters_used) if iters_used else 0.0
    efficiency   = success_rate / avg_iter if avg_iter > 0 else 0.0

    # precision@oracle
    prec_oracle = sum(
        1 for r in results
        if r.get("oracle_iter") is not None
        and abs(r["stop_iter"] - r["oracle_iter"]) <= 1
    ) / max(1, sum(1 for r in results if r.get("oracle_iter") is not None))

    return {
        "n": len(results),
        "success_rate":   round(success_rate, 4),
        "avg_iter":       round(avg_iter, 3),
        "efficiency":     round(efficiency, 4),
        "precision_oracle": round(prec_oracle, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("=== E3: SlopeNav vs Fixed-N 对比 ===\n")

    trajs = load_trajectories(args.dry_run)
    print(f"轨迹数: {len(trajs)}\n")

    strategies = {
        "SlopeNav":      apply_slopenav,
        "Fixed-3":       lambda t: apply_fixed_n(t, 3),
        "Fixed-4 (SR)":  lambda t: apply_fixed_n(t, 4),
        "Fixed-5":       lambda t: apply_fixed_n(t, 5),
        "Δ-threshold":   apply_delta_threshold,
    }

    all_results = {}
    for name, fn in strategies.items():
        strategy_results = []
        for traj in trajs:
            r = fn(traj)
            r["question_id"] = traj["question_id"]
            r["oracle_iter"] = oracle_stop(traj)
            r["flask_avg"]   = traj["flask_avg"]
            r["quality_tier"] = traj["quality_tier"]
            strategy_results.append(r)
        all_results[name] = strategy_results

    # 计算指标
    metrics = {name: compute_metrics(results) for name, results in all_results.items()}

    # 门控
    sn_eff    = metrics["SlopeNav"]["efficiency"]
    f5_eff    = metrics["Fixed-5"]["efficiency"]
    go        = sn_eff >= f5_eff * 1.20

    # 打印报告
    print("=" * 70)
    print("  E3 Results")
    print("=" * 70)
    print(f"  {'Strategy':<18} {'SuccessR':>8} {'AvgIter':>8} {'Efficiency':>10} {'Prec@Oracle':>12}")
    print(f"  {'-' * 62}")
    for name, m in metrics.items():
        marker = " ←" if name == "SlopeNav" else ""
        print(f"  {name:<18} {m['success_rate']:>8.3f} {m['avg_iter']:>8.2f} "
              f"{m['efficiency']:>10.4f} {m['precision_oracle']:>12.3f}{marker}")

    print(f"\n  SlopeNav efficiency = {sn_eff:.4f}")
    print(f"  Fixed-5  efficiency = {f5_eff:.4f}")
    print(f"  比值 = {sn_eff/f5_eff:.2f}x  (需 ≥ 1.20x)")
    verdict = "✅  GO  → E4 Pivot 验证" if go else "❌  NO-GO → 分析轨迹模式"
    print(f"\n  门控结论: {verdict}")
    print("=" * 70)

    # 按 tier 拆分 SlopeNav
    print("\n  SlopeNav Per-tier efficiency:")
    for tier in ["low", "mid", "high"]:
        t_res = [r for r in all_results["SlopeNav"] if r["quality_tier"] == tier]
        t_m   = compute_metrics(t_res)
        print(f"    {tier:5s}: succ={t_m['success_rate']:.3f}  "
              f"avg_iter={t_m['avg_iter']:.2f}  eff={t_m['efficiency']:.4f}")

    # 保存
    suffix = "_dry" if args.dry_run else "_full"
    (RESULTS_DIR / f"e3_metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(RESULTS_DIR / f"e3_rows{suffix}.jsonl", "w") as f:
        for name, rows in all_results.items():
            for r in rows:
                f.write(json.dumps({"strategy": name, **r}, ensure_ascii=False) + "\n")
    print(f"\n[Results saved to {RESULTS_DIR}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args)
