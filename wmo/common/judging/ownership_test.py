"""Regression tests for the common judging clean-break migration."""

from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parents[2]
_LEGACY_OWNER_FILES = (
    _PACKAGE_ROOT / "optimize" / "judge.py",
    _PACKAGE_ROOT / "optimize" / "reward.py",
    _PACKAGE_ROOT / "simulation" / "evaluation" / "gold.py",
    _PACKAGE_ROOT / "simulation" / "scenarios" / "verification" / "judge.py",
)
_LEGACY_IMPORTS = (
    "wmo.optimize.judge",
    "wmo.optimize.reward",
    "wmo.simulation.evaluation.gold",
    "wmo.simulation.scenarios.verification.judge",
)


@pytest.mark.parametrize("legacy_owner", _LEGACY_OWNER_FILES)
def test_legacy_judging_owners_are_deleted(legacy_owner: Path) -> None:
    """Require common.judging to remain the only owner of migrated judge implementations."""
    assert not legacy_owner.exists(), f"Legacy judging owner still exists: {legacy_owner}"


@pytest.mark.parametrize("legacy_import", _LEGACY_IMPORTS)
def test_production_code_does_not_import_legacy_judging(legacy_import: str) -> None:
    """Prevent new production callers from crossing the clean-break boundary."""
    offenders = [
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if not path.name.endswith("_test.py") and legacy_import in path.read_text()
    ]
    assert not offenders, (
        f"Production code imports legacy judging owner {legacy_import}: {offenders}"
    )
