"""Witnessed labels are admitted by re-derivation, and every refusal carries a named reason."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    EvidenceAdmission,
    EvidenceAdmissionError,
    LabelWitness,
    adequacy_refusals,
    persist_specification,
    record_label_witness,
    require_admitted,
    verify_label_witness,
)
from exp.simulation.evaluation.tasks import (
    OutcomeRule,
    TaskVerifier,
    Verdict,
    VerifierChallenge,
    label_admitted,
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
    renamed = witnessed(store, specification(store), "done")
    refusal = verify_label_witness(store.artifacts, renamed, label="pass", compared=(candidate,))
    assert refusal is not None and refusal.reason == "self_approval"


@pytest.mark.parametrize(
    "change", ["description", "annotations", "conjunction_order", "ignored_vocabulary"]
)
def test_cosmetic_predicate_changes_cannot_manufacture_independence(
    tmp_path: Path, change: str
) -> None:
    """Titles, descriptions and conjunction ordering do not create a second source of truth."""
    store = ProjectStore(tmp_path, "admission")
    candidate = TaskVerifier(
        description="Candidate requirement.",
        outcome_schema={"type": "object", "properties": {"final_state": {"type": "object"}}},
        required_paths=("/final_state", "/final_answer"),
        rules=(
            OutcomeRule(path="/final_state/status", operator="equals", value="done"),
            OutcomeRule(path="/final_answer", operator="type", value="string"),
        ),
    )
    updates: JsonObject = {}
    if change == "description":
        updates["description"] = "Claims to be independent."
    elif change == "annotations":
        updates["outcome_schema"] = {
            "title": "Independent oracle",
            "type": "object",
            "properties": {
                "final_state": {"type": "object", "description": "Copied constraint"},
            },
        }
    elif change == "ignored_vocabulary":
        updates["outcome_schema"] = {
            **candidate.outcome_schema,
            "x-claim": "independent",
            "format": "anything",
        }
    else:
        updates["required_paths"] = list(reversed(candidate.required_paths))
        updates["rules"] = [rule.model_dump(mode="json") for rule in reversed(candidate.rules)]
    copied = TaskVerifier.model_validate({**candidate.model_dump(mode="json"), **updates})
    label = witnessed(store, specification(store, predicate=copied), "done")
    refusal = verify_label_witness(store.artifacts, label, label="pass", compared=(candidate,))
    assert refusal is not None and refusal.reason == "self_approval"


def test_schema_property_named_description_remains_an_assertion(tmp_path: Path) -> None:
    """Removing schema annotations must not remove data properties with the same spelling."""
    store = ProjectStore(tmp_path, "admission")
    first = TaskVerifier(
        description="Field requirement.",
        outcome_schema={
            "properties": {
                "description": {"const": "first"},
            }
        },
    )
    different = first.model_copy(
        update={"outcome_schema": {"properties": {"description": {"const": "second"}}}}
    )
    label = witnessed(store, specification(store, predicate=first), "done")
    assert verify_label_witness(store.artifacts, label, label="pass", compared=(different,)) is None


def test_prefix_item_order_remains_semantic_when_detecting_copies(tmp_path: Path) -> None:
    """Ordered tuple constraints cannot be normalized like an unordered conjunction."""
    store = ProjectStore(tmp_path, "admission")
    first = TaskVerifier(
        description="Ordered prefix",
        outcome_schema={
            "properties": {"steps": {"prefixItems": [{"const": 1}, {"const": 2}]}},
        },
    )
    other = first.model_copy(
        update={
            "outcome_schema": {
                "properties": {"steps": {"prefixItems": [{"const": 2}, {"const": 1}]}},
            }
        }
    )
    label = witnessed(store, specification(store, predicate=first), "done")
    assert verify_label_witness(store.artifacts, label, label="pass", compared=(other,)) is None


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize(
    "source_type",
    [
        "task-refinement-proposal",
        "continuation-study",
        "continuation-checkpoint",
        "executable-world-image",
        "investigation-evidence",
        "task-factory-registration",
        "task-factory-cell",
        "task-factory-result",
        "author-intent",
        "author-response",
        "world-binding-intent",
        "world-proposal-cell",
        "world-proposals",
    ],
)
def test_generated_provenance_anywhere_in_the_chain_is_self_approval(
    tmp_path: Path, depth: int, source_type: str
) -> None:
    """A specification derived, at any depth, from model output cannot label anything."""
    store = ProjectStore(tmp_path, "admission")
    source = saved(store, "proposal", source_type)
    for index in range(depth - 1):
        source = saved(store, f"derived-{index}", "authored-requirement", (source,))
    label = witnessed(store, specification(store, sources=(source,)), "done")
    refusal = verify_label_witness(store.artifacts, label, label="pass", compared=())
    assert refusal is not None
    assert refusal.reason == "self_approval"
    assert source_type in refusal.detail
    recorded = saved(store, "recorded", "trace-dataset")
    clean = witnessed(store, specification(store, sources=(recorded,)), "done")
    assert verify_label_witness(store.artifacts, clean, label="pass", compared=()) is None


@pytest.mark.parametrize("depth", [1, 3])
def test_unclassified_provenance_type_is_refused(tmp_path: Path, depth: int) -> None:
    """A type nobody classified as recorded evidence fails closed, at any depth."""
    store = ProjectStore(tmp_path, "admission")
    source = saved(store, "unclassified", "newly-generated-proposal")
    for index in range(depth - 1):
        source = saved(store, f"derived-{index}", "authored-requirement", (source,))
    label = witnessed(store, specification(store, sources=(source,)), "done")
    refusal = verify_label_witness(store.artifacts, label, label="pass", compared=())
    assert refusal is not None
    assert refusal.reason == "self_approval"
    assert "unclassified (newly-generated-proposal)" in refusal.detail


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


def test_a_hand_written_witness_verdict_is_regraded(tmp_path: Path) -> None:
    """A saved witness claiming a verdict its specification does not produce is refused."""
    store = ProjectStore(tmp_path, "admission")
    spec = specification(store)
    forged = LabelWitness(
        kind="specification_check",
        specification=spec,
        outcome_sha256=sha256_json(outcome("pending")),
        observation=outcome("pending"),
        verdict="pass",
        reason="Asserted, not graded.",
    )
    witness = artifact_input(
        store.artifacts.write_json(
            artifact_id="forged-witness",
            artifact_type="label-witness",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=NOW, inputs=(spec,), code_revision="test"
            ),
            files={"witness.json": forged.model_dump(mode="json")},
        )
    )
    label = VerifierChallenge(
        id="pending",
        outcome=outcome("pending"),
        expected="pass",
        authority="synthetic_specification",
        rationale="Claimed by a forged witness.",
        witness=witness,
    )
    refusal = verify_label_witness(store.artifacts, label, label="pass", compared=())
    assert refusal is not None
    assert refusal.reason == "unverifiable_witness"
    assert "re-grades to another verdict" in refusal.detail


def test_reordered_table_relations_cannot_certify_themselves(tmp_path: Path) -> None:
    """Composite-key order, annotations and duplicate clauses do not create independent checks."""
    from exp.simulation.evaluation.table_relations import TableRelation

    project = ProjectStore(tmp_path, "tables")
    relation = TableRelation(
        source="/initial_state/rows",
        target="/final_state/rows",
        operation="last_per_key",
        key_columns=(0, 1),
    )
    candidate = TaskVerifier(description="Original", outcome_schema={}, table_relations=(relation,))
    disguised = candidate.model_copy(
        update={
            "description": "Independent",
            "required_paths": (relation.source, relation.target),
            "table_relations": (relation.model_copy(update={"key_columns": (1, 0)}), relation),
        }
    )
    ref = specification(project, predicate=disguised)
    value: JsonObject = {
        "initial_state": {"rows": [[1, 2]]},
        "final_state": {"rows": [[1, 2]]},
        "steps": [],
        "final_answer": "done",
    }
    witness = record_label_witness(
        project.artifacts, specification=ref, outcome=value, created_at=NOW, code_revision="test"
    )
    challenge = VerifierChallenge(
        id="rows",
        outcome=value,
        expected="pass",
        authority="synthetic_specification",
        witness=witness,
        rationale="Check table semantics.",
    )
    refusal = verify_label_witness(
        project.artifacts, challenge, label="pass", compared=(candidate,)
    )
    assert refusal is not None and refusal.reason == "self_approval"


def test_argument_schema_fingerprint_is_canonical_and_opt_in() -> None:
    """Annotations and local refs cannot manufacture a different policy oracle."""
    from exp.simulation.evaluation.admission import _predicate_fingerprint

    legacy = TaskVerifier(description="Legacy predicate.", outcome_schema={"type": "object"})
    assert (
        _predicate_fingerprint(legacy)
        == "cad1acfee27b1afa4886fcba759ae364621ffa2a601cfdd32ec54b2685504a39"
    )
    first = legacy.model_copy(
        update={"argument_schemas": {"edit": {"properties": {"updates": {"maxItems": 0}}}}}
    )
    equivalent = legacy.model_copy(
        update={
            "argument_schemas": {
                "edit": {
                    "$defs": {"empty": {"maxItems": 0, "description": "annotation"}},
                    "properties": {"updates": {"$ref": "#/$defs/empty"}},
                }
            }
        }
    )
    different = legacy.model_copy(
        update={"argument_schemas": {"edit": {"properties": {"updates": {"maxItems": 1}}}}}
    )
    assert _predicate_fingerprint(first) == _predicate_fingerprint(equivalent)
    assert _predicate_fingerprint(first) != _predicate_fingerprint(different)
    assert _predicate_fingerprint(first) != _predicate_fingerprint(legacy)
