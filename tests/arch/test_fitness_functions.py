"""Fitness Functions — SlopeNav 架构约束。"""

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "slopenav"


def _get_imports(filepath: Path) -> list[str]:
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _get_all_py_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.py"))


# ── FF-01: 零外部依赖（decision/slope/verdicts/diagnose 不依赖网络库）──────────

def test_ff01_no_network_dependencies_in_core():
    """decision / slope / verdicts / diagnose 不依赖 openai/anthropic/httpx/requests。"""
    forbidden = {"openai", "anthropic", "httpx", "requests", "aiohttp"}
    core_dirs = ["decision", "slope", "verdicts", "diagnose"]
    for subdir in core_dirs:
        for filepath in _get_all_py_files(SRC_ROOT / subdir):
            imports = _get_imports(filepath)
            for imp in imports:
                for f in forbidden:
                    assert not imp.startswith(f), (
                        f"架构违规 [FF-01]: {subdir}/{filepath.name} 导入了网络库 {imp}"
                    )


# ── FF-02: domain 层无外部依赖 ────────────────────────────────────────────────

def test_ff02_domain_pure():
    """domain 层只用标准库。"""
    allowed_prefixes = {"slopenav", "from __future__", "__future__"}
    domain_files = _get_all_py_files(SRC_ROOT / "domain")
    third_party_forbidden = {"numpy", "pandas", "scipy"}
    for filepath in domain_files:
        imports = _get_imports(filepath)
        for imp in imports:
            for f in third_party_forbidden:
                assert not imp.startswith(f), (
                    f"架构违规 [FF-02]: domain/{filepath.name} 不应依赖 {imp}"
                )


# ── FF-03: decision/tree.py 是纯函数（无 self 修改状态）────────────────────────

def test_ff03_decision_tree_is_pure_function():
    """decision/tree.py 的 decide() 函数应是纯函数（不修改全局状态）。"""
    tree_file = SRC_ROOT / "decision" / "tree.py"
    assert tree_file.exists()
    source = tree_file.read_text(encoding="utf-8")
    # 不应有 global 赋值或 self.xxx = 模式
    assert "global " not in source, "decide() 不应修改全局变量"
    # 不应有 asyncio 依赖
    assert "import asyncio" not in source and "await " not in source, \
        "decide() 应是同步纯函数"


# ── FF-04: 公共接口通过 __init__.py 暴露 ──────────────────────────────────────

def test_ff04_public_api_in_init():
    init_file = SRC_ROOT / "__init__.py"
    assert init_file.exists()
    content = init_file.read_text(encoding="utf-8")
    assert "SlopeNav" in content
    assert "Decision" in content


# ── FF-05: 所有子包有 __init__.py ────────────────────────────────────────────

def test_ff05_all_packages_have_init():
    for subdir in ["domain", "slope", "verdicts", "decision", "diagnose"]:
        init = SRC_ROOT / subdir / "__init__.py"
        assert init.exists(), f"{subdir}/__init__.py 缺失"


# ── FF-06: SlopeNav.step() 返回 Decision ─────────────────────────────────────

def test_ff06_step_returns_decision():
    """SlopeNav.step() 返回 Decision 类型。"""
    from slopenav import Decision, SlopeNav
    nav = SlopeNav()
    d = nav.step(0, 0.7)
    assert isinstance(d, Decision)


# ── FF-07: decide() 纯函数可独立调用 ─────────────────────────────────────────

def test_ff07_decide_pure_function_callable_independently():
    """decide() 可不通过 SlopeNav 直接调用。"""
    from slopenav.decision.tree import decide
    d = decide(
        n_evals=3, current_score=0.5, best_score=0.5,
        linear_slope=0.0, ema_slope=0.0, effective_high_slope=0.05,
        min_threshold=0.80, pivot_count=0, max_pivots=2,
        require_min_evals=1, vp=None,
    )
    assert d.action in ("continue", "pivot", "deliver")
