"""Cumulative preparation cost admission never renews or expands authority."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.bootstrap_admission import validate_preparation_admission
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_plan import plan_bootstrap_preparation
from exp.optimize.evaluation.variance.bootstrap_resume import load_bootstrap_resume


def test_admission_rejects_full_cumulative_overrun_and_expired_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed pending costs and retained spend cannot be checked against a zeroed baseline."""
    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    reference = register_pilot(project, spec, root, pilot)
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="pilot-executor")
    plan = plan_bootstrap_preparation(
        project, spec, code_revision="pilot-executor", resume_binding=reference
    )
    over = plan.model_copy(update={"accounted_unpriced_attempts": 100001})
    with pytest.raises(ValueError, match="parent request cap"):
        validate_preparation_admission(over, binding, None, now=datetime.now(UTC))
    small = binding.model_copy(update={"preparation_unpriced_attempts": 0, "preparation_usd": 0})
    if plan.unpriced_attempts or plan.known_usd:
        with pytest.raises(ValueError, match="original preparation authority"):
            validate_preparation_admission(plan, small, None, now=datetime.now(UTC))
    expired = binding.model_copy(update={"execution_deadline": binding.deadline.replace(year=2000)})
    with pytest.raises(ValueError, match="deadline expired"):
        validate_preparation_admission(plan, expired, None, now=datetime.now(UTC))
