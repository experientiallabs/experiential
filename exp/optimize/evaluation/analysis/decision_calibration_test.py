"""Negative controls keep candidate failure, teacher labels and missing calls visible."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonValue, sha256_json
from exp.common.project import ArtifactStore, ProjectConfig, ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.decision_calibration import (
    DecisionCalibration,
    DecisionCase,
    DecisionDefinition,
    DecisionLabel,
    DecisionPrediction,
    calibrate_decisions,
)
from exp.optimize.evaluation.analysis.memory_evidence import EvidenceLocator

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def save(
    store: ArtifactStore,
    name: str,
    value: JsonValue,
    *,
    inputs: tuple[ArtifactInput, ...] = (),
    offset: int = 1,
    file: str = "value.json",
) -> EvidenceLocator:
    """Save actual immutable source evidence instead of supplying unchecked labels."""
    manifest = store.write_json(
        artifact_id=name,
        artifact_type="decision-fixture",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=NOW + timedelta(seconds=offset),
            inputs=inputs,
            code_revision="fixture",
        ),
        files={file: value},
    )
    return EvidenceLocator(
        source=artifact_input(manifest), file=file, path=(), sha256=sha256_json(value)
    )


def study(
    tmp_path: Path, *, shared_source: bool = False
) -> tuple[ArtifactStore, DecisionCalibration]:
    """Keep one confident correct, one missing-candidate and one failed confirmation case.

    With ``shared_source`` every case declares the same leakage group.
    """
    project = ProjectStore(tmp_path, "decisions")
    project.initialize(ProjectConfig(project_id="decisions"))
    store = project.artifacts
    definition = DecisionDefinition(
        student={"model": "small"},
        teacher={"model": "large"},
        fallback={"model": "large"},
        question_sha256="a" * 64,
        renderer_sha256="b" * 64,
        candidate_builder_sha256="c" * 64,
        tools_sha256="d" * 64,
        knowledge_sha256="e" * 64,
        policy_sha256="f" * 64,
    )

    def group(index: int) -> str:
        """The declared leakage group of one case."""
        return "same-source" if shared_source else f"source-{index}"

    config = {
        "definition": definition.model_dump(mode="json"),
        "threshold": 0.8,
        "minimum_support": 2,
        "maximum_error": 0.2,
    }
    inventory = [
        {"id": f"case-{index}", "partition": "confirmation", "leakage_groups": [group(index)]}
        for index in range(3)
    ]
    # Registered first, on a host whose clock runs a day ahead of the questions' host.
    registration = save(
        store,
        "registration",
        {**config, "inventory": inventory},
        file="registration.json",
        offset=86_400,
    )
    cases = []
    for index, target in enumerate(("a", "missing", "a")):
        question = save(
            store,
            f"question-{index}",
            {
                "question": "Which action?",
                "permitted_state": {},
                "candidates": ["a", "b"],
                "definition_sha256": sha256_json(definition),
                "hard_allowed": True,
            },
            inputs=(registration.source,),
        )
        prediction = {"selected": "a", "probabilities": {"a": 0.9, "b": 0.05, "OTHER": 0.05}}
        source = save(store, f"answer-{index}", prediction, offset=2)
        label = save(store, f"label-{index}", target, offset=3)
        cases.append(
            DecisionCase(
                id=f"case-{index}",
                partition="confirmation",
                leakage_groups=(group(index),),
                question=question,
                candidates=("a", "b"),
                hard_allowed=True,
                student=DecisionPrediction(source=source, **prediction) if index < 2 else None,
                label=DecisionLabel(source=label, target=target, authority="native_verifier"),
            )
        )
    return store, DecisionCalibration(
        definition=definition,
        threshold=0.8,
        minimum_support=2,
        maximum_error=0.2,
        cases=tuple(cases),
        registration=registration.source,
    )


def test_missing_candidates_and_failures_do_not_disappear(tmp_path: Path) -> None:
    """Candidate generation failure is distinct from classification and all slots remain."""
    store, spec = study(tmp_path)
    before = store.list_ids()
    report = calibrate_decisions(store, spec)
    row = report.partitions[-1]
    assert (row.planned, row.predicted, row.labelled, row.candidate_available) == (3, 2, 3, 2)
    assert (row.accepted, row.accepted_errors, row.correct) == (2, 1, 1)
    assert row.brier_mean == pytest.approx(0.865)
    assert not report.confirmation_supported
    assert "confirmation has missing or failed student predictions" in report.blockers
    assert store.list_ids() == before


@pytest.mark.parametrize("change", ["threshold", "inventory", "definition"])
def test_posthoc_policy_or_denominator_change_is_refused(tmp_path: Path, change: str) -> None:
    """Neither a better cutoff nor dropping a failed case can reuse a registered study."""
    store, spec = study(tmp_path)
    if change == "threshold":
        spec = spec.model_copy(update={"threshold": 0.95})
    elif change == "inventory":
        spec = spec.model_copy(update={"cases": spec.cases[:-1]})
    else:
        spec = spec.model_copy(
            update={
                "definition": spec.definition.model_copy(update={"fallback": {"model": "changed"}})
            }
        )
    with pytest.raises(ValueError, match="changed after registration"):
        calibrate_decisions(store, spec)


def test_teacher_answer_cannot_become_independent_label_through_a_child(tmp_path: Path) -> None:
    """Relabelling a teacher-derived artifact is caught through its dependency closure."""
    store, spec = study(tmp_path)
    first = spec.cases[0]
    assert first.student
    derived = save(store, "derived-label", "a", inputs=(first.student.source.source,))
    case = first.model_copy(
        update={"label": DecisionLabel(source=derived, target="a", authority="native_verifier")}
    )
    with pytest.raises(ValueError, match="depends on a teacher or student"):
        calibrate_decisions(store, spec.model_copy(update={"cases": (case, *spec.cases[1:])}))


def test_claimed_policy_allowance_must_match_saved_question(tmp_path: Path) -> None:
    """A high confidence action cannot silently overwrite the recorded hard restriction."""
    store, spec = study(tmp_path)
    case = spec.cases[0].model_copy(update={"hard_allowed": False})
    with pytest.raises(ValueError, match="hard-policy decision differs"):
        calibrate_decisions(store, spec.model_copy(update={"cases": (case, *spec.cases[1:])}))


def test_confirmation_before_registration_is_not_prospective(tmp_path: Path) -> None:
    """A correctly hashed registration still cannot legitimize earlier confirmation evidence,
    even when its clock claims it was saved before the questions."""
    import json

    store, spec = study(tmp_path)
    contents = json.loads(store.read_bytes(spec.registration.artifact_id, "registration.json"))
    late = save(store, "late-registration", contents, file="registration.json", offset=-86_400)
    question = store.read(spec.cases[0].question.source.artifact_id).manifest
    assert store.read(late.source.artifact_id).manifest.created_at < question.created_at
    with pytest.raises(ValueError, match="does not cite the frozen registration"):
        calibrate_decisions(store, spec.model_copy(update={"registration": late.source}))


def test_confirmation_citing_the_registration_follows_it_whatever_the_clocks_say(
    tmp_path: Path,
) -> None:
    """Questions citing the registration are prospective although their clock claims earlier."""
    store, spec = study(tmp_path)
    question = store.read(spec.cases[0].question.source.artifact_id).manifest
    assert question.created_at < store.read(spec.registration.artifact_id).manifest.created_at
    assert calibrate_decisions(store, spec).partitions[-1].planned == 3


def test_probability_scores_are_not_ordinal_relevance(tmp_path: Path) -> None:
    """JEV's 0..2 relevance scores cannot be mistaken for correctness probabilities."""
    _, spec = study(tmp_path)
    first = spec.cases[0].student
    assert first
    with pytest.raises(ValueError, match="sum to one"):
        DecisionPrediction(
            source=first.source, selected="a", probabilities={"a": 2, "b": 1, "OTHER": 0}
        )


def test_correlated_rows_do_not_inflate_confirmation_support(tmp_path: Path) -> None:
    """Two accepted answers from the same source count as one risk-estimation unit."""
    store, spec = study(tmp_path, shared_source=True)
    result = calibrate_decisions(store, spec)
    row = result.partitions[-1]
    assert row.accepted_labelled == 2 and row.independent_units == 1
    assert row.error_upper_95 == pytest.approx(1)
    assert "insufficient independent confirmation support" in result.blockers


def test_shared_lineage_cannot_cross_partition_even_with_different_questions(
    tmp_path: Path,
) -> None:
    """Splitting related questions into different rows does not make confirmation independent."""
    store, spec = study(tmp_path)
    cases = (
        spec.cases[0].model_copy(update={"partition": "train"}),
        spec.cases[1].model_copy(update={"leakage_groups": spec.cases[0].leakage_groups}),
        spec.cases[2],
    )
    registration = spec.model_dump(mode="json", exclude={"cases", "registration"})
    registration["inventory"] = [
        case.model_dump(mode="json", include={"id", "partition", "leakage_groups"})
        for case in cases
    ]
    saved = save(store, "leaky-registration", registration, file="registration.json", offset=0)
    with pytest.raises(ValueError, match="partitions share"):
        calibrate_decisions(
            store, spec.model_copy(update={"cases": cases, "registration": saved.source})
        )
