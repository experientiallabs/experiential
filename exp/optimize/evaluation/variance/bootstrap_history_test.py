"""Native historical source proofs stay read-only and never authorize current execution."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_history import (
    read_bootstrap_preparation_provenance,
)
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_protocols import (
    current_bootstrap_source,
    propose_bootstrap_protocol_revision,
    register_bootstrap_protocol_revision,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    load_bootstrap_resume,
    register_bootstrap_resume,
)


def test_historical_reader_pins_real_source_record_without_granting_current_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intrinsic original source remains readable after the consuming source has changed."""
    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, root, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    proposal = propose_bootstrap_protocol_revision(
        project, previous_binding=previous, code_revision="saved-source-executor"
    )
    control = register_bootstrap_protocol_revision(
        project, proposal, code_revision="saved-source-executor"
    )
    reference = register_bootstrap_resume(
        project,
        original_inputs=parent.original_inputs,
        code_revision="saved-source-executor",
        permitted_changes=(*parent.permitted_changes, "preparation_protocol_revision"),
        previous_binding=previous,
        review_revision=parent.review_revision,
        pilot_revision=parent.pilot_revision,
        execution_deadline=parent.execution_deadline,
        execution_window=parent.execution_window,
        protocol_revision=control,
    )
    ceiling = operation_spend_ceiling(project, parent.original_operation)
    before = (ceiling.summary(), project.artifacts.list_ids())
    source = current_bootstrap_source()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_protocols.current_bootstrap_source",
        lambda: source.model_copy(update={"protocols": {**source.protocols, "consumer": "new"}}),
    )
    with pytest.raises(ValueError, match="differs from reviewed source"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="saved-source-executor")
    proof = read_bootstrap_preparation_provenance(project, reference, spec=spec)
    assert proof.binding.original_operation == parent.original_operation
    assert proof.producer_revision == "saved-source-executor"
    assert proof.source_ref == control
    assert proof.source_type == "bootstrap-protocol-revision"
    assert proof.source_sha256 == sha256_json(proposal.source)
    assert proof.execution_spec.task_ids == (pilot.task_id,)
    assert (ceiling.summary(), project.artifacts.list_ids()) == before
    with pytest.raises(ValueError, match="reference differs"):
        read_bootstrap_preparation_provenance(
            project, reference.model_copy(update={"sha256": "0" * 64}), spec=spec
        )


def test_legacy_resume_without_native_source_proof_is_correctly_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer label and a completed build never substitute for recorded source bytes."""
    project, spec, reference, _ = pilot_fixture(tmp_path, monkeypatch)
    before = project.artifacts.list_ids()
    with pytest.raises(ValueError, match="no recorded source provenance"):
        read_bootstrap_preparation_provenance(project, reference, spec=spec)
    assert project.artifacts.list_ids() == before
