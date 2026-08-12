"""Architecture tests for the world-model simulation package boundary."""

from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parent
WMO_DIR = SIMULATION_DIR.parent


def test_simulation_domains_are_nested() -> None:
    """Current ingestion and execution domains stay under the simulation package."""
    expected_dirs = {
        "ingest",
        "engines",
        "mining",
        "orchestration",
        "specs",
    }
    missing_dirs = sorted(name for name in expected_dirs if not (SIMULATION_DIR / name).is_dir())
    assert not missing_dirs, f"simulation packages missing under wmo/simulation: {missing_dirs}"

    expected_modules = {"build.py", "comparison.py"}
    missing_modules = sorted(
        name for name in expected_modules if not (SIMULATION_DIR / name).is_file()
    )
    assert not missing_modules, (
        f"simulation modules missing under wmo/simulation: {missing_modules}"
    )

    retired = sorted(
        name
        for name in ("evaluation", "model", "retrieval", "environment.py")
        if (SIMULATION_DIR / name).is_file() or any((SIMULATION_DIR / name).rglob("*.py"))
    )
    assert not retired, f"retired simulation owners returned: {retired}"

    legacy_dirs = sorted(
        name
        for name in (
            "connect",
            "engine",
            "env",
            "evals",
            "ingest",
            "retrieval",
            "scenarios",
            "serving",
        )
        if (WMO_DIR / name).exists()
    )
    assert not legacy_dirs, f"simulation packages returned to the flat wmo namespace: {legacy_dirs}"
