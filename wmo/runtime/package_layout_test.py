"""Architecture tests for the agent runtime package boundary."""

from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
WMO_DIR = RUNTIME_DIR.parent


def test_runtime_domains_are_nested() -> None:
    """Agent execution ownership stays visible in the package tree."""
    expected_dirs = {"agents", "environments", "models", "router"}
    missing_dirs = sorted(name for name in expected_dirs if not (RUNTIME_DIR / name).is_dir())
    assert not missing_dirs, f"runtime packages missing under wmo/runtime: {missing_dirs}"

    retired = sorted(
        name
        for name in ("__init__.py", "environment.py", "episode.py", "evaluation", "harness")
        if (RUNTIME_DIR / name).exists()
    )
    assert not retired, f"retired runtime owners returned under wmo/runtime: {retired}"

    legacy_dirs = sorted(
        name for name in ("agents", "platform", "runs") if (WMO_DIR / name).exists()
    )
    assert not legacy_dirs, f"runtime packages returned to the flat wmo namespace: {legacy_dirs}"

    assert not (WMO_DIR / "optimize" / "harness").exists(), "retired harness search returned"
    assert not (WMO_DIR / "evals" / "harbor").exists(), (
        "runtime evaluator returned to the simulation evaluation namespace"
    )
