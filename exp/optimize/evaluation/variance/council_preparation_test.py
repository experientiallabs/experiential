"""Actual native preparation qualification, independent of final judge or study roles."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.council_preparation import (
    CouncilPreparationContract,
    qualify_prepared_council,
)
from exp.optimize.evaluation.variance.council_qualification_test import _fixture


def test_native_preparation_proof_does_not_require_a_rubric_or_actor_critic_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read committed native QA using only preparation and selected-task inputs."""
    project, prepared, resume, study, wire = _fixture(tmp_path, monkeypatch)
    contract = CouncilPreparationContract(
        preparation_spec=study.preparation_spec,
        bootstrap=study.bootstrap,
        exposure=study.exposure,
        family_simulator=study.family_simulator,
    )
    calls = len(wire.requests)
    proof = qualify_prepared_council(
        project,
        prepared,
        resume,
        contract,
        code_revision="corrected",
    )
    assert proof.start.validation.passed
    assert proof.start.job.study.checkpoint == proof.start.checkpoint
    assert proof.contract == contract and proof.producer_revision == "corrected"
    assert len(wire.requests) == calls
