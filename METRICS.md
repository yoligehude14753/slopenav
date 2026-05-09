# slopenav · METRICS（北极星 + OKR）

> 版本：v1.0 · 日期：2026-05-08

## 北极星指标

> **指标**：在固定 LLM 预算下，**与 oracle baseline 相比的任务完成率提升**  
> oracle baseline = 在 max_iter=10 下穷举所有 score 序列，事后挑最优停止点  
> 当前基线：固定 max-iter (=3) 策略  
> 论文目标：在保持 task success rate ≥ baseline 的前提下，**平均迭代次数减少 ≥ 30%**  
> 测量频率：每次主实验跑

**为什么是这个指标**：
- 直接回答"用 SlopeNav 比简单策略省多少算力"
- 双向限制：不能为了省算力牺牲成功率
- 与 oracle 对比，能看出 SlopeNav 离最优有多远（论文中的 "regret"）

## 输入指标

| 指标 | 阈值 | 监测频率 |
|------|------|---------|
| 单元测试覆盖率 | ≥ 90% | 每 PR |
| `decide()` 调用延迟 | P99 ≤ 1ms | 每 PR |
| Fitness Functions 通过率 | 100% | 每 PR |
| benchmark 跑通时长 | ≤ 30 分钟 | 每实验 |

## 护栏指标

| 指标 | 红线 |
|------|------|
| 决策一致性（同序列重跑） | < 100% 即视为 bug |
| 安装包大小 | ≤ 5MB（无 LLM 依赖应该很小） |
| 外部 import 数 | numpy + 标准库，其他不允许 |
| 任何 LLM 调用 | 0（零容忍） |

## 季度 OKR (2026 Q3)

**O1：SlopeNav 在合成 + 真实数据上证明价值**
- KR1：相对固定 max-iter=3 策略，迭代次数减少 ≥ 30%
- KR2：相对单阈值 (0.80) 策略，false-deliver 率减少 ≥ 50%
- KR3：相对 oracle baseline 的 regret ≤ 10%

**O2：开源仓库可被其他 agent 框架引用**
- KR1：发布 v1.0，README 含 Reflexion / Self-Refine 集成示例
- KR2：arXiv 发布
- KR3：至少 1 个外部项目 PR 集成 SlopeNav（如 LangChain agentevals）
