"""Exact source-only successors retain historical builds, settings and cumulative authority."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.rules.direct_research import direct_research_protocol_sha256
from exp.optimize.evaluation.rules.rule_tags import rule_tagging_protocol_sha256
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import bootstrap_plan as preparation
from exp.optimize.evaluation.variance import bootstrap_protocols as protocol_owner
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_protocols import (
    current_bootstrap_source,
    load_bootstrap_protocol_revision,
    propose_bootstrap_protocol_revision,
    register_bootstrap_protocol_revision,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    bootstrap_execution_spec,
    load_bootstrap_resume,
    register_bootstrap_resume,
)
from exp.simulation.mining.environment_candidates import candidate_protocol_sha256
from exp.simulation.mining.environment_review import review_protocol_sha256


def test_source_successor_preserves_native_plan_claims_and_original_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual native continuation keeps the old build and ledger without charging either again."""
    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, root, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    ceiling = operation_spend_ceiling(project, parent.original_operation)
    reservation = ceiling.reserve(alias="unpriced", url="https://fixture.invalid", payload={})
    ceiling.mark_dispatched(reservation, "historical")
    ceiling.retain_unknown(reservation)
    before = ceiling.summary()
    proposal = propose_bootstrap_protocol_revision(
        project, previous_binding=previous, code_revision="source-executor"
    )
    control = register_bootstrap_protocol_revision(
        project, proposal, code_revision="source-executor"
    )
    assert (
        register_bootstrap_protocol_revision(project, proposal, code_revision="source-executor")
        == control
    )
    reference = register_bootstrap_resume(
        project,
        original_inputs=parent.original_inputs,
        code_revision="source-executor",
        permitted_changes=(*parent.permitted_changes, "preparation_protocol_revision"),
        previous_binding=previous,
        review_revision=parent.review_revision,
        pilot_revision=parent.pilot_revision,
        execution_deadline=parent.execution_deadline,
        execution_window=parent.execution_window,
        protocol_revision=control,
    )
    current = load_bootstrap_resume(project, reference, spec=spec, code_revision="source-executor")
    assert bootstrap_execution_spec(project, current, spec) == bootstrap_execution_spec(
        project, parent, spec
    )
    assert current.original_operation == parent.original_operation
    assert current.grants == parent.grants and current.stop == parent.stop
    assert current.build_revision == pilot.build_producer_revision
    assert current.build_identity + "-result" == pilot.saved_build.artifact_id
    plan = preparation.plan_bootstrap_preparation(
        project, spec, code_revision="source-executor", resume_binding=reference
    )
    assert plan.identity == parent.original_operation and plan.build_saved
    assert plan.accounted_unpriced_attempts == 1 and plan.build_identity == parent.build_identity
    assert all(
        "tagging" not in cost.name and "review compatibility" not in cost.name
        for cost in plan.costs
    )
    assert ceiling.summary() == before
    # The actual execution entrypoint uses this same provider-free admission API.
    admitted = preparation.admit_bootstrap_preparation(
        project, spec, code_revision="source-executor", resume_binding=reference
    )
    assert admitted == plan and ceiling.summary() == before
    assert current.producer_revision != current.build_revision
    changed_spec = spec.model_copy(
        update={
            "bootstrap": spec.bootstrap.model_copy(
                update={
                    "author": spec.bootstrap.author.model_copy(
                        update={"maximum_input_bytes": 256000}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="original specification"):
        preparation.plan_bootstrap_preparation(
            project, changed_spec, code_revision="source-executor", resume_binding=reference
        )
    changed = proposal.source.model_copy(
        update={"files": {**proposal.source.files, "unreviewed.py": "d" * 64}}
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(protocol_owner, "current_bootstrap_source", lambda: changed)
        with pytest.raises(ValueError, match="reviewed source"):
            load_bootstrap_resume(project, reference, spec=spec, code_revision="source-executor")

    def stop_at_admission(*args: object, **kwargs: object) -> None:
        """Prove execution calls the shared cost gate before client/runtime composition."""
        raise RuntimeError("offline admission sentinel")

    monkeypatch.setattr(preparation, "admit_bootstrap_preparation", stop_at_admission)
    with pytest.raises(RuntimeError, match="offline admission sentinel"):
        preparation.run_bootstrap_preparation(
            project, spec, code_revision="source-executor", resume_binding=reference
        )


def test_source_successor_refuses_unreviewed_values_refs_and_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fingerprint, provenance, saved-stage and execution-setting tampering cannot register."""
    project, spec, root, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, root, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    before = operation_spend_ceiling(project, parent.original_operation).summary()
    proposal = propose_bootstrap_protocol_revision(
        project, previous_binding=previous, code_revision="source-executor"
    )
    forged_source = proposal.source.model_copy(
        update={"protocols": {**proposal.source.protocols, "skip_controls": "true"}}
    )
    for delta in [
        {"producer_revision": "unreviewed"},
        {"previous_producer_revision": "unreviewed"},
        {"execution_spec_sha256": "a" * 64},
        {"build_producer_revision": "source-executor"},
        {"saved_build": ArtifactInput(artifact_id=pilot.saved_build.artifact_id, sha256="b" * 64)},
        {"source": forged_source},
        {"compatible_stages": ("rules",)},
    ]:
        with pytest.raises(ValueError):
            register_bootstrap_protocol_revision(
                project, proposal.model_copy(update=delta), code_revision="source-executor"
            )
    control = register_bootstrap_protocol_revision(
        project, proposal, code_revision="source-executor"
    )
    with pytest.raises(ValueError):
        load_bootstrap_protocol_revision(
            project,
            ArtifactInput(artifact_id=control.artifact_id, sha256="c" * 64),
            code_revision="source-executor",
        )
    with pytest.raises(ValueError):
        register_bootstrap_resume(
            project,
            original_inputs=parent.original_inputs,
            code_revision="source-executor",
            permitted_changes=(*parent.permitted_changes, "preparation_protocol_revision"),
            previous_binding=previous,
            pilot_revision=parent.pilot_revision,
            protocol_revision=control,
            execution_deadline=datetime.now(UTC) + timedelta(minutes=30),
        )
    with pytest.raises(ValueError):
        register_bootstrap_resume(
            project,
            original_inputs=parent.original_inputs,
            code_revision="source-executor",
            permitted_changes=parent.permitted_changes,
            previous_binding=previous,
            pilot_revision=parent.pilot_revision,
            protocol_revision=control,
        )
    assert operation_spend_ceiling(project, parent.original_operation).summary() == before
    assert current_bootstrap_source() == proposal.source


def test_source_snapshot_binds_direct_research_and_rule_questions() -> None:
    """Source continuations pin extraction implementation and transferable question semantics."""
    snapshot = current_bootstrap_source()
    assert "exp/optimize/evaluation/rules/direct_research.py" in snapshot.files
    assert snapshot.protocols["direct_research"] == direct_research_protocol_sha256()
    assert snapshot.protocols["rule_questions"] == rule_tagging_protocol_sha256()


def test_source_snapshot_binds_complete_bounded_review_implementation() -> None:
    """Source successors fingerprint partition, native review, family and quote owners."""
    snapshot = current_bootstrap_source()
    assert {
        "exp/optimize/evaluation/variance/build.py",
        "exp/optimize/evaluation/variance/build_clients.py",
        "exp/optimize/evaluation/variance/baseline_execution.py",
        "exp/optimize/evaluation/variance/bootstrap_history.py",
        "exp/optimize/evaluation/variance/pipeline_review.py",
        "exp/simulation/mining/environment_candidates.py",
        "exp/simulation/mining/environment_families.py",
        "exp/simulation/mining/environment_review.py",
        "exp/simulation/rules/contracts.py",
    } <= snapshot.files.keys()
    assert snapshot.protocols["native_successor"] == "bootstrap-source-revision-v11"
    assert snapshot.protocols["environment_candidates"] == candidate_protocol_sha256()
    assert snapshot.protocols["environment_review"] == review_protocol_sha256()


def test_source_snapshot_binds_native_placement_and_schema_coverage() -> None:
    """A reviewed source successor cannot omit placement semantics or shared declarations."""
    from exp.optimize.evaluation.variance.family_scopes import probe_placement_protocol_sha256

    snapshot = current_bootstrap_source()
    assert {
        "exp/optimize/evaluation/variance/family_scopes.py",
        "exp/optimize/evaluation/variance/build_records.py",
        "exp/optimize/evaluation/variance/bootstrap_pilot.py",
        "exp/simulation/rules/contracts.py",
    } <= snapshot.files.keys()
    assert snapshot.protocols["probe_placement"] == probe_placement_protocol_sha256()
