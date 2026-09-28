"""Finite learning-run configuration validation."""

import pytest
from pydantic import ValidationError

from exp.optimize.claas.service.configuration import RunConfiguration


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_run_deadline_must_be_finite_and_positive(seconds: float) -> None:
    """A run must never silently become unbounded."""
    with pytest.raises(ValidationError):
        RunConfiguration(maximum_run_seconds=seconds)
