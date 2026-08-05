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
    """Routing, model, and harness ownership stays visible in the package tree."""
    expected = {"routing", "model", "harness"}
    missing = sorted(name for name in expected if not (OPTIMIZE_DIR / name).is_dir())
    assert not missing, f"optimization domain packages missing under wmo/optimize: {missing}"

    legacy = sorted(name for name in ("distill", "harness") if (WMO_DIR / name).exists())
    assert not legacy, f"optimization packages returned to the flat wmo namespace: {legacy}"

    flat_routing = sorted(name for name in ROUTING_MODULES if (OPTIMIZE_DIR / name).exists())
    assert not flat_routing, (
        f"routing modules returned to the optimize package root: {flat_routing}"
    )
