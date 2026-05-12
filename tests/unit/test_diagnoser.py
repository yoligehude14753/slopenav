"""单元测试 — StagnationDiagnoser 三种停滞原因覆盖。"""

from slopenav.diagnose.diagnoser import diagnose_stagnation


def _tool(success: bool, **kwargs) -> dict:
    return {"success": success, **kwargs}


class _ObjTool:
    """非 dict 的工具结果（测试 getattr 路径）。"""

    def __init__(self, success, path=None):
        self.success = success
        self.path = path


# ── 基础路径 ─────────────────────────────────────────────────────────────────


def test_short_history_returns_unclear():
    result = diagnose_stagnation([0.5, 0.4], [])
    assert result.cause == "unclear"
    assert "short" in result.detail


def test_no_signals_returns_unclear():
    result = diagnose_stagnation([0.5, 0.5, 0.5], [])
    assert result.cause == "unclear"


# ── eval_blind_spot ──────────────────────────────────────────────────────────


def test_eval_blind_spot_detected_when_file_produced_low_score():
    """有文件产出但分数卡在低位 → eval_blind_spot。"""
    tool_results = [
        _tool(True, path="/output/report.csv"),
        _tool(True, file_url="https://example.com/file.xlsx"),
    ]
    result = diagnose_stagnation([0.30, 0.32, 0.31], tool_results, min_threshold=0.80)
    assert result.cause == "eval_blind_spot"
    assert result.confidence >= 0.7
    assert "files=" in result.detail


def test_eval_blind_spot_uses_getattr_for_object_tool():
    """非 dict 工具结果走 getattr 路径。"""
    tool_results = [_ObjTool(success=True, path="/output/file.pdf")]
    result = diagnose_stagnation([0.28, 0.30, 0.29], tool_results, min_threshold=0.80)
    assert result.cause == "eval_blind_spot"


# ── capability_limit (tool failures) ─────────────────────────────────────────


def test_capability_limit_high_failure_rate():
    """工具失败率 >70% → capability_limit。"""
    tool_results = [_tool(False)] * 4 + [_tool(True)]  # 4/5 = 80% failure
    result = diagnose_stagnation([0.4, 0.38, 0.39], tool_results, min_threshold=0.80)
    assert result.cause == "capability_limit"
    assert result.confidence >= 0.8


def test_capability_limit_borderline_failure_does_not_trigger():
    """失败率 =60% (不超过70%) → 不触发。"""
    tool_results = [_tool(False)] * 3 + [_tool(True)] * 2  # 3/5 = 60%
    result = diagnose_stagnation([0.4, 0.38, 0.39], tool_results, min_threshold=0.80)
    assert result.cause != "capability_limit" or result.confidence < 0.8


# ── capability_limit (flat score) ────────────────────────────────────────────


def test_capability_limit_flat_score_no_tools():
    """分数极低且无成功工具 → capability_limit（flat 路径）。"""
    scores = [0.30, 0.31, 0.30, 0.30, 0.31]  # range < 0.05, avg < 0.56
    result = diagnose_stagnation(scores, [], min_threshold=0.80)
    assert result.cause == "capability_limit"
    assert "flat" in result.detail


def test_flat_score_does_not_trigger_when_above_threshold():
    """平坦分数但均值足够高 → 不判断为 capability_limit。"""
    scores = [0.75, 0.76, 0.75, 0.75, 0.76]
    result = diagnose_stagnation(scores, [], min_threshold=0.80)
    # avg_recent ≈ 0.75 > 0.80 * 0.7 = 0.56 → 不触发 flat 路径
    assert result.cause == "unclear"
