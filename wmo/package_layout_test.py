"""Architecture tests for the top-level Python package."""

from pathlib import Path

WMO_DIR = Path(__file__).resolve().parent


def test_wmo_package_root_is_a_closed_allowlist() -> None:
    """Only the three product domains, shared code, and CLI live directly under ``wmo``."""
    actual_dirs = {
        path.name for path in WMO_DIR.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    expected_dirs = {"cli", "common", "optimize", "runtime", "simulation"}
    assert actual_dirs == expected_dirs, (
        f"wmo package directories are {sorted(actual_dirs)}, expected {sorted(expected_dirs)}"
    )

    allowed_modules = {"__init__.py", "__main__.py", "conftest.py"}
    flat_modules = sorted(
        path.name
        for path in WMO_DIR.glob("*.py")
        if path.name not in allowed_modules and not path.name.endswith("_test.py")
    )
    assert not flat_modules, (
        f"production modules returned to the flat wmo namespace: {flat_modules}"
    )
