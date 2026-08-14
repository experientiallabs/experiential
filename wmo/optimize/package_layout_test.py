"""Architecture tests for the optimization package boundaries."""

from pathlib import Path

OPTIMIZE_DIR = Path(__file__).resolve().parent
WMO_DIR = OPTIMIZE_DIR.parent

ROUTING_MODULES = {
    "cluster_labels.py",
    "compression.py",
    "deepswe.py",
    "embedding_cache.py",
    "knn.py",
    "outcomes.py",
    "pareto.py",
    "pipeline.py",
    "policy.py",
    "report.py",
    "routing.py",
    "scorecard.py",
    "sweep.py",
    "sweep_partial.py",
    "teacher.py",
}


def test_optimization_domains_are_nested() -> None:
    """Current router and model optimization ownership stays visible in the package tree."""
    expected = {"router", "model"}
    actual = {
        path.name for path in OPTIMIZE_DIR.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    assert actual == expected, (
        f"optimization packages are {sorted(actual)}, expected {sorted(expected)}"
    )

    flat_namespace = sorted(name for name in ("distill", "harness") if (WMO_DIR / name).exists())
    assert not flat_namespace, (
        f"optimization packages sit in the flat wmo namespace: {flat_namespace}"
    )

    assert not (OPTIMIZE_DIR / "harness").exists(), (
        "agent execution belongs under wmo/runtime/agents, not wmo/optimize/harness"
    )

    forbidden_modules = sorted(
        name for name in ("base.py", "gepa.py") if (OPTIMIZE_DIR / name).exists()
    )
    assert not forbidden_modules, f"forbidden optimizer modules present: {forbidden_modules}"

    flat_routing = sorted(name for name in ROUTING_MODULES if (OPTIMIZE_DIR / name).exists())
    assert not flat_routing, f"routing modules sit in the optimize package root: {flat_routing}"
