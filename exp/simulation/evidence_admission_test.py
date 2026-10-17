"""Witnessed labels are admitted by re-derivation, and every refusal carries a named reason."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.simulation.evaluation_tasks import (
    OutcomeRule,
    TaskVerifier,
    Verdict,
    VerifierChallenge,
    label_admitted,
)
from exp.simulation.evidence_admission import (
    EvaluationSpecification,
    EvidenceAdmission,
    EvidenceAdmissionError,
    adequacy_refusals,
    persist_specification,
    record_label_witness,
    require_admitted,
    verify_label_witness,
)

NOW = datetime(2026, 10, 17, tzinfo=UTC)
REFERENCE = TaskVerifier(
    description="Reference completion specification.",
    outcome_schema={"type": "object"},
    rules=(OutcomeRule(path="/final_state/status", operator="equals", value="done"),),
)


def outcome(status: str) -> JsonObject:
    """A complete normalized outcome with one declared final status."""
    return {
        "initial_state": {},
        "final_state": {"status": status},
        "final_answer": status,
        "steps": [],
        "stop_reason": "completed",
    }


def saved(
    store: ProjectStore, identity: str, kind: str, inputs: tuple[ArtifactInput, ...] = ()
) -> ArtifactInput:
    """Write one opaque artifact of a named type with explicit provenance."""
    return artifact_input(
        store.artifacts.write_json(
            artifact_id=identity,
            artifact_type=kind,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=NOW, inputs=inputs, code_revision="test"
            ),
            files={"value.json": {"identity": identity}},
        )
    )


def specification(
    store: ProjectStore,
    *,
    predicate: TaskVerifier = REFERENCE,
    sources: tuple[ArtifactInput, ...] = (),
) -> ArtifactInput:
    """Save one outcome specification with declared sources."""
    return persist_specification(
        store.artifacts,
        EvaluationSpecification(
            description="An order is complete only when its final status is done.",
            predicate=predicate,
            subject="outcome",
        ),
        sources=sources,
        created_at=NOW,
        code_revision="test",
    )


def witnessed(
    store: ProjectStore, spec: ArtifactInput, status: str, expected: str = ""
) -> VerifierChallenge:
    """A synthetic-specification label whose witness is a saved specification check."""
    return VerifierChallenge.model_validate(
        {
            "id": status,
            "outcome": outcome(status),
            "expected": expected or ("pass" if status == "done" else "fail"),
            "authority": "synthetic_specification",
            "rationale": "The reference specification decides completion.",
            "witness": record_label_witness(
                store.artifacts,
                specification=spec,
                outcome=outcome(status),
                created_at=NOW,
                code_revision="test",
            ).model_dump(mode="json"),
        }
    )


def test_specification_witness_admits_a_label_without_any_approval(tmp_path: Path) -> None:
    """A re-derived specification check is the whole admission path: no approval flag."""
    store = ProjectStore(tmp_path, "admission")
    label = witnessed(store, specification(store), "done")
    assert label.approved is False
    assert verify_label_witness(store.artifacts, label, label="pass", compared=()) is None
    assert label_admitted(label, {"done"})
    assert not label_admitted(label, set())


def test_witnessed_authorities_cannot_be_admitted_by_assertion(tmp_path: Path) -> None:
    """Approval cannot stand in for a witness, and a native label always names one."""
    store = ProjectStore(tmp_path, "admission")
    label = witnessed(store, specification(store), "done")
    data = label.model_dump(mode="json")
    with pytest.raises(ValidationError, match="re-verifying"):
        VerifierChallenge.model_validate(
            {**data, "approved": True, "source_refs": [data["witness"]]}
        )
    with pytest.raises(ValidationError, match="native label witness"):
        VerifierChallenge.model_validate({**data, "authority": "native_execution", "witness": None})
    with pytest.raises(ValidationError, match="carry a label witness"):
        VerifierChallenge.model_validate({**data, "authority": "human"})
    bare = VerifierChallenge.model_validate({**data, "witness": None})
    assert not label_admitted(bare, {"done"})
    refusal = verify_label_witness(store.artifacts, bare, label="pass", compared=())
    assert refusal is not None and refusal.reason == "unverifiable_witness"


def test_witness_that_does_not_reproduce_is_refused(tmp_path: Path) -> None:
    """Another outcome, another verdict, a wrong digest or a missing file are all refused."""
    store = ProjectStore(tmp_path, "admission")
    spec = specification(store)
    done = witnessed(store, spec, "done")
    assert done.witness is not None
    moved = done.model_copy(update={"outcome": outcome("pending")})
    flipped = done.model_copy(update={"expected": "fail"})
    forged = done.model_copy(
        update={"witness": done.witness.model_copy(update={"sha256": "0" * 64})}
    )
    missing = done.model_copy(
        update={"witness": ArtifactInput(artifact_id="no-such-witness", sha256="0" * 64)}
    )
    for label, verdict, detail in (
        (moved, "pass", "different outcome"),
        (flipped, "fail", "label declares fail"),
        (forged, "pass", "cannot be re-read"),
        (missing, "pass", "cannot be re-read"),
    ):
        refusal = verify_label_witness(store.artifacts, label, label=verdict, compared=())
        assert refusal is not None
        assert refusal.reason == "unverifiable_witness"
        assert detail in refusal.detail


def test_the_predicate_under_test_cannot_witness_its_own_labels(tmp_path: Path) -> None:
    """A specification equal to the candidate (or a generated proposal) is self-approval."""
    store = ProjectStore(tmp_path, "admission")
    candidate = REFERENCE.model_copy(update={"description": "Candidate verifier."})
    label = witnessed(store, specification(store, predicate=candidate), "done")
    refusal = verify_label_witness(store.artifacts, label, label="pass", compared=(candidate,))
    assert refusal is not None and refusal.reason == "self_approval"
    independent = witnessed(store, specification(store), "done")
    assert (
        verify_label_witness(store.artifacts, independent, label="pass", compared=(candidate,))
        is None
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_generated_provenance_anywhere_in_the_chain_is_self_approval(
    tmp_path: Path, depth: int
) -> None:
    """A specification derived, at any depth, from model output cannot label anything."""
    store = ProjectStore(tmp_path, "admission")
    source = saved(store, "proposal", "task-refinement-proposal")
    for index in range(depth - 1):
        source = saved(store, f"derived-{index}", "curated-notes", (source,))
    label = witnessed(store, specification(store, sources=(source,)), "done")
    refusal = verify_label_witness(store.artifacts, label, label="pass", compared=())
    assert refusal is not None
    assert refusal.reason == "self_approval"
    assert "task-refinement-proposal" in refusal.detail
    recorded = saved(store, "recorded", "trace-dataset")
    clean = witnessed(store, specification(store, sources=(recorded,)), "done")
    assert verify_label_witness(store.artifacts, clean, label="pass", compared=()) is None


def test_every_partition_verdict_and_axis_needs_enough_admitted_labels() -> None:
    """One admitted case per side, or an unadmitted second case, is inadequate evidence."""
    axes = frozenset({"outcome"})
    rows: list[tuple[str, Verdict, bool, frozenset[str]]] = [
        ("development", "pass", True, axes),
        ("development", "pass", True, axes),
        ("development", "fail", True, axes),
        ("development", "fail", False, axes),
        ("confirmation", "pass", True, axes),
        ("confirmation", "pass", True, frozenset[str]()),
        ("confirmation", "fail", True, axes),
        ("confirmation", "fail", True, axes),
    ]
    refusals = adequacy_refusals(
        rows, partitions=("development", "confirmation"), axes=("outcome",)
    )
    assert {(row.reason, row.subject) for row in refusals} == {
        ("inadequate_evidence", "development"),
        ("inadequate_evidence", "development on axis outcome"),
        ("inadequate_evidence", "confirmation on axis outcome"),
    }
    complete = [(partition, verdict, True, axes) for partition, verdict, _, _ in rows]
    assert adequacy_refusals(complete, partitions=("development", "confirmation")) == ()


def test_refusals_stop_paid_work_with_their_names() -> None:
    """The raised error names every reason and subject."""
    admission = EvidenceAdmission.model_validate(
        {
            "refusals": [
                {"reason": "contaminated_lineage", "subject": "lineage-a", "detail": "exposed"},
                {"reason": "self_approval", "subject": "case-1", "detail": "generated"},
            ]
        }
    )
    with pytest.raises(EvidenceAdmissionError) as error:
        require_admitted(admission)
    assert "[contaminated_lineage] lineage-a" in str(error.value)
    assert "[self_approval] case-1" in str(error.value)
    require_admitted(EvidenceAdmission())


def test_admission_binds_the_exact_challenge_not_its_identifier(tmp_path: Path) -> None:
    """An edited label reusing a verified ID is not admitted."""
    store = ProjectStore(tmp_path, "admission")
    label = witnessed(store, specification(store), "done")
    admission = EvidenceAdmission.model_validate({"verified": {label.id: sha256_json(label)}})
    assert admission.verified_ids((label,)) == {"done"}
    edited = label.model_copy(update={"rationale": "Changed after verification."})
    assert admission.verified_ids((edited,)) == frozenset()
