"""Architecture tests for the world-model simulation package boundary."""

from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parent
WMO_DIR = SIMULATION_DIR.parent


def test_simulation_domains_are_nested() -> None:
    """World-model construction, serving, and evaluation stay under one package."""
    expected_dirs = {
        "evaluation",
        "ingest",
        "model",
        "retrieval",
        "scenarios",
        "serving",
    }
    missing_dirs = sorted(name for name in expected_dirs if not (SIMULATION_DIR / name).is_dir())
    assert not missing_dirs, f"simulation packages missing under wmo/simulation: {missing_dirs}"

    expected_modules = {"environment.py", "hub.py"}
    missing_modules = sorted(
        name for name in expected_modules if not (SIMULATION_DIR / name).is_file()
    )
    assert not missing_modules, (
        f"simulation modules missing under wmo/simulation: {missing_modules}"
    )

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
    assert not (WMO_DIR / "hub.py").exists(), "simulation hub returned to the flat wmo namespace"
