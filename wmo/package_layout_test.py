"""Architecture tests for the top-level Python package."""

from pathlib import Path

import pytest

WMO_DIR = Path(__file__).resolve().parent

DOMAIN_SHAPES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "common": (
        frozenset(
            {
                "config",
                "core",
                "evaluations",
                "judging",
                "models",
                "observability",
                "project",
                "rollouts",
                "routing",
                "tasks",
                "traces",
            }
        ),
        frozenset({"__init__.py"}),
    ),
    "runtime": (
        frozenset({"agents", "environments", "models", "router"}),
        frozenset(),
    ),
    "simulation": (
        frozenset({"engines", "ingest", "mining", "orchestration", "retrieval", "specs"}),
        frozenset({"__init__.py", "build.py", "comparison.py"}),
    ),
    "optimize": (
        frozenset({"model", "router"}),
        frozenset({"__init__.py"}),
    ),
}


def test_wmo_package_root_is_a_closed_allowlist() -> None:
    """Only product domains, public workflow composition, shared code, and CLI live under wmo."""
    actual_dirs = {
        path.name for path in WMO_DIR.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    expected_dirs = {"cli", "common", "optimize", "runtime", "simulation", "workflow"}
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


@pytest.mark.parametrize(("domain", "shape"), sorted(DOMAIN_SHAPES.items()))
def test_domain_ownership_stays_visible_in_the_package_tree(
    domain: str, shape: tuple[frozenset[str], frozenset[str]]
) -> None:
    """Each product domain exposes exactly its current subpackages and root modules."""
    expected_packages, expected_modules = shape
    domain_dir = WMO_DIR / domain

    actual_packages = {
        path.name for path in domain_dir.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    assert actual_packages == expected_packages, (
        f"{domain} subpackages are {sorted(actual_packages)}, expected {sorted(expected_packages)}"
    )

    actual_modules = {
        path.name for path in domain_dir.glob("*.py") if not path.name.endswith("_test.py")
    }
    assert actual_modules == expected_modules, (
        f"{domain} root modules are {sorted(actual_modules)}, expected {sorted(expected_modules)}"
    )
