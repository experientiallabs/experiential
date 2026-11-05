"""Runtime construction honors native scoped authority before importing trusted factories."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes
from exp.optimize.evaluation.continuations.job import load_continuation_runtime
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.runtime.models.scoped_spending import open_spend_envelope, spend_envelope_scope
from exp.runtime.models.scoped_spending_test import plan, setup_policy


def test_scoped_continuation_refuses_missing_native_world_authority_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing actor scope alone cannot grant an unscoped simulator fallback."""
    project, job, transport = job_fixture(tmp_path / "job", monkeypatch)
    source = tmp_path / "must-not-import.py"
    source.write_text('raise AssertionError("factory executed without council world authority")')
    job = job.model_copy(
        update={"runtime_file": source, "runtime_sha256": sha256_bytes(source.read_bytes())}
    )
    ceiling, policy = setup_policy(tmp_path / "budget")
    envelope = open_spend_envelope(ceiling, plan(policy, "actors", 1))
    with spend_envelope_scope(ceiling, envelope):
        with pytest.raises(ValueError, match="no native per-job provider authority"):
            load_continuation_runtime(project, job)
    assert not transport.requests and not ceiling.summary().roles


def test_legacy_unscoped_continuation_keeps_actual_factory_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing offline native factories still restore their original exact checkpoint."""
    project, job, transport = job_fixture(tmp_path / "job", monkeypatch)
    with load_continuation_runtime(project, job) as runtime:
        assert runtime.probe_sha256 == job.study.probe_sha256
    assert not transport.requests
