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
        preparation_producer=study.preparation_producer,
        preparation_source_sha256=study.preparation_source_sha256,
        preparation_provenance=study.preparation_provenance,
        preparation_provenance_type=study.preparation_provenance_type,
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


def test_historical_preparation_and_new_executor_remain_distinct_without_new_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new council reads exact saved source/QA without relabeling the preparation producer."""
    project, prepared, resume, study, wire = _fixture(tmp_path, monkeypatch)
    contract = CouncilPreparationContract(
        preparation_spec=study.preparation_spec,
        preparation_producer=study.preparation_producer,
        preparation_source_sha256=study.preparation_source_sha256,
        preparation_provenance=study.preparation_provenance,
        preparation_provenance_type=study.preparation_provenance_type,
        bootstrap=study.bootstrap,
        exposure=study.exposure,
        family_simulator=study.family_simulator,
    )
    before = (len(wire.requests), project.artifacts.list_ids())
    proof = qualify_prepared_council(
        project, prepared, resume, contract, code_revision="new-council-executor"
    )
    assert proof.producer_revision == "new-council-executor"
    assert project.artifacts.read(prepared.artifact_id).manifest.code_revision == "corrected"
    assert contract.preparation_producer == "corrected"
    for field, changed in (
        ("preparation_producer", "unrelated-producer"),
        ("preparation_source_sha256", "0" * 64),
        ("preparation_provenance", prepared),
        ("preparation_provenance_type", "bootstrap-protocol-revision"),
    ):
        with pytest.raises(ValueError, match="declared preparation provenance"):
            qualify_prepared_council(
                project,
                prepared,
                resume,
                contract.model_copy(update={field: changed}),
                code_revision="new-council-executor",
            )
    assert (len(wire.requests), project.artifacts.list_ids()) == before
