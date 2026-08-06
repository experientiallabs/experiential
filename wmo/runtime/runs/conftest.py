"""Shared fixtures for the run-telemetry tests.

The D-RUNS shared-truth artifacts live OUTSIDE this repo (they are real grid output,
co-owned with the platform side), so every test that reads them has to be skippable.
Exposing them as a pytest fixture rather than a module constant is deliberate: a test
declares the dependency in its signature, so it cannot read the directory without
having gone through the skip guard. Three tests previously reached past a
module-level guard and hard-failed on any checkout without the artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Where the shared artifacts live by default, and the override for a machine that
# keeps them elsewhere (CI, a second checkout, a copy under /tmp).
FIXTURES_ENV = "D_RUNS_FIXTURES"
DEFAULT_FIXTURES = Path.home() / "Desktop/Projects/wmh-plan/d-runs-fixtures"


def fixtures_dir() -> Path:
    """The shared-truth fixtures directory, honoring the env override."""
    override = os.environ.get(FIXTURES_ENV)
    return Path(override) if override else DEFAULT_FIXTURES


@pytest.fixture
def d_runs_fixtures() -> Path:
    """The shared-truth fixtures directory, skipping when it is not present.

    Returns:
        The directory holding `expected-events.jsonl` and `artifacts/`.
    """
    directory = fixtures_dir()
    if not (directory / "expected-events.jsonl").exists():
        pytest.skip(
            f"D-RUNS shared fixtures not present at {directory}; "
            f"set {FIXTURES_ENV} to point at a copy"
        )
    return directory


@pytest.fixture
def d_runs_artifacts(d_runs_fixtures: Path) -> Path:
    """The artifacts subdirectory the backfill mapping reads.

    Separate from `d_runs_fixtures` because the two are independently absent: a
    checkout can hold the expected stream without the raw artifacts.
    """
    artifacts = d_runs_fixtures / "artifacts"
    if not artifacts.exists():
        pytest.skip(f"D-RUNS artifacts not present at {artifacts}")
    return artifacts
