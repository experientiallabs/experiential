"""Ownership tests keeping `wmo.common.judging` the single owner of judge implementations."""

from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parents[2]
_FORBIDDEN_OWNER_FILES = (
    _PACKAGE_ROOT / "optimize" / "judge.py",
    _PACKAGE_ROOT / "optimize" / "reward.py",
    _PACKAGE_ROOT / "simulation" / "evaluation" / "gold.py",
    _PACKAGE_ROOT / "simulation" / "scenarios" / "verification" / "judge.py",
)
_FORBIDDEN_IMPORTS = (
    "wmo.optimize.judge",
    "wmo.optimize.reward",
    "wmo.simulation.evaluation.gold",
    "wmo.simulation.scenarios.verification.judge",
)


@pytest.mark.parametrize("forbidden_owner", _FORBIDDEN_OWNER_FILES)
def test_no_judging_owner_outside_common_judging(forbidden_owner: Path) -> None:
    """Only `wmo.common.judging` may hold a judge implementation."""
    assert not forbidden_owner.exists(), f"Judging owner outside common: {forbidden_owner}"


@pytest.mark.parametrize("forbidden_import", _FORBIDDEN_IMPORTS)
def test_production_code_imports_judging_only_from_common(forbidden_import: str) -> None:
    """Production callers reach judging through `wmo.common.judging` and no other module path."""
    offenders = [
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if not path.name.endswith("_test.py") and forbidden_import in path.read_text()
    ]
    assert not offenders, f"Production code imports judging from {forbidden_import}: {offenders}"
