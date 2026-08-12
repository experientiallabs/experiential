"""Architecture tests for the agent runtime package boundary."""

from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
WMO_DIR = RUNTIME_DIR.parent


def test_runtime_domains_are_nested() -> None:
    """Agent execution ownership stays visible in the package tree."""
    expected_dirs = {"agents", "evaluation", "harness"}
    missing_dirs = sorted(name for name in expected_dirs if not (RUNTIME_DIR / name).is_dir())
    assert not missing_dirs, f"runtime packages missing under wmo/runtime: {missing_dirs}"

    expected_modules = {"environment.py", "episode.py"}
    missing_modules = sorted(
        name for name in expected_modules if not (RUNTIME_DIR / name).is_file()
    )
    assert not missing_modules, f"runtime modules missing under wmo/runtime: {missing_modules}"

    legacy_dirs = sorted(
        name for name in ("agents", "platform", "runs") if (WMO_DIR / name).exists()
    )
    assert not legacy_dirs, f"runtime packages returned to the flat wmo namespace: {legacy_dirs}"

    assert not (WMO_DIR / "optimize" / "harness").exists(), (
        "runtime harness returned to the optimization namespace"
    )
    assert not (WMO_DIR / "evals" / "harbor").exists(), (
        "runtime evaluator returned to the simulation evaluation namespace"
    )
