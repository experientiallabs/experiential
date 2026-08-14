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
    actual_dirs = {
        path.name
        for path in SIMULATION_DIR.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert actual_dirs == expected_dirs, (
        f"simulation packages are {sorted(actual_dirs)}, expected {sorted(expected_dirs)}"
    )

    expected_modules = {"build.py", "comparison.py"}
    actual_modules = {
        path.name for path in SIMULATION_DIR.glob("*.py") if not path.name.endswith("_test.py")
    }
    assert actual_modules == {"__init__.py", *expected_modules}, (
        f"simulation modules are {sorted(actual_modules)}, expected current build and comparison"
    )

    forbidden = sorted(
        name
        for name in ("evaluation", "model", "retrieval", "environment.py")
        if (SIMULATION_DIR / name).is_file() or any((SIMULATION_DIR / name).rglob("*.py"))
    )
    assert not forbidden, f"forbidden simulation owners present: {forbidden}"

    flat_namespace = sorted(
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
    assert not flat_namespace, (
        f"simulation packages sit in the flat wmo namespace: {flat_namespace}"
    )
