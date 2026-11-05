"""Review-only fixtures exercise stored bytes, human histories and paired regressions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonValue, sha256_json
from exp.common.judging.labels import HumanScore, ReviewerJudgment
from exp.common.models import BillingSource, ModelSnapshot
from exp.common.project import ArtifactStore, ProjectConfig, ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.memory_evidence import EvidenceLocator
from exp.optimize.evaluation.analysis.source_access import declare_source_access
from exp.optimize.evaluation.variance.review_cases import (
    JudgeAnswer,
    JudgeRun,
    ReviewBank,
    ReviewCase,
    ReviewSnapshot,
    case_key,
    disagreement_drafts,
    export_gold,
    measure_reviews,
    paired_retest,
    review_coordinates,
    save_review_snapshot,
)
from exp.simulation.mining.questions import Question, QuestionBank

NOW = datetime(2026, 1, 1, tzinfo=UTC)
MODEL = ModelSnapshot(
    provider="fixture",
    model_id="fixture-judge",
    revision="frozen",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)


def save(
    store: ArtifactStore, name: str, value: JsonValue, *, inputs: tuple[ArtifactInput, ...] = ()
) -> EvidenceLocator:
    """Persist actual immutable fixture bytes with a verified locator."""
    manifest = store.write_json(
        artifact_id=name,
        artifact_type="review-fixture",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=NOW, code_revision="fixture", inputs=inputs
        ),
        files={"value.json": value},
    )
    declare_source_access(
        store,
        artifact_input(manifest),
        scope="development",
        rationale="Test fixture diagnostics only.",
        code_revision="fixture",
    )
    return EvidenceLocator(
        source=artifact_input(manifest), file="value.json", path=(), sha256=sha256_json(value)
    )


def study(tmp_path: Path) -> tuple[ArtifactStore, ReviewSnapshot]:
    """Build three predicates on one real coordinate plus a synthetic pending proposal."""
    project = ProjectStore(tmp_path, "review")
    project.initialize(ProjectConfig(project_id="review"))
    store = project.artifacts
    questions = QuestionBank(
        bank_id="fixture-bank",
        version=1,
        questions=(
            Question(
                question_id="bad", family="quality", text="Does the recorded action violate scope?"
            ),
            Question(
                question_id="good", family="quality", text="Was the recorded request fulfilled?"
            ),
            Question(
                question_id="description", family="quality", text="Did the actor inspect the file?"
            ),
        ),
    )
    question_source = save(store, "question-definition", questions.model_dump(mode="json"))
    rubric = save(
        store, "rubric-definition", {"version": 1, "instructions": "Use recorded effects."}
    )
    real = save(store, "real-trace", {"session": "recorded", "steps": [{"tool": "inspect"}]})
    synthetic = save(
        store,
        "synthetic-trace",
        {"provenance": "synthetic", "expected_label": "AUTHOR_PROPOSED", "steps": []},
    )
    cases = tuple(
        ReviewCase(
            subject=real,
            question_id=question,
            family_id="family-a",
            task_id="task-a",
            lineages=("session-a",),
            partition="development",
            provenance="real",
        )
        for question in questions.question_ids()
    ) + (
        ReviewCase(
            subject=synthetic,
            question_id="bad",
            family_id="family-b",
            task_id="task-b",
            lineages=("synthetic-session",),
            partition="test",
            provenance="synthetic",
        ),
    )
    bank = ReviewBank(
        questions=questions,
        questions_source=question_source,
        target_rubric=rubric,
        semantics={"bad": "failure_yes", "good": "success_yes", "description": "descriptive"},
        cases=cases,
    )
    return store, ReviewSnapshot(bank=bank)


def label(
    store: ArtifactStore,
    snapshot: ReviewSnapshot,
    index: int,
    score: int,
    *,
    suffix: str = "first",
    supersedes: str | None = None,
    adjudicates: tuple[str, ...] = (),
    authority: str = "human",
) -> ReviewSnapshot:
    """Record an explicit fixture human label, never convert an expected/model answer."""
    case = snapshot.bank.cases[index]
    record = HumanScore(
        **review_coordinates(snapshot.bank, case).model_dump(),
        label_id=f"label-{index}-{suffix}",
        score=score,
        created_at=NOW,
        supersedes_label_id=supersedes,
        adjudicates_review_ids=adjudicates,
    )
    history = snapshot.history.correct(record) if supersedes else snapshot.history.append(record)
    source = save(
        store,
        record.label_id,
        {
            "authority": authority,
            "record": record.model_dump(mode="json"),
            "comment": "Fixture only.",
            "evidence": [case.subject.model_dump(mode="json")],
        },
    )
    return ReviewSnapshot(
        bank=snapshot.bank,
        history=history,
        human_sources={**snapshot.human_sources, record.label_id: source},
    )


def review(
    store: ArtifactStore,
    snapshot: ReviewSnapshot,
    index: int,
    score: int,
    reviewer: str,
) -> ReviewSnapshot:
    """Record an independent reviewer, retaining disagreement until explicit adjudication."""
    record = ReviewerJudgment(
        **review_coordinates(snapshot.bank, snapshot.bank.cases[index]).model_dump(),
        review_id=f"review-{index}-{reviewer}",
        reviewer_id=reviewer,
        score=score,
        created_at=NOW,
        critique="Fixture evidence review, not an actual human annotation.",
    )
    source = save(
        store,
        record.review_id,
        {
            "authority": "human",
            "record": record.model_dump(mode="json"),
            "comment": "Fixture only.",
            "evidence": [snapshot.bank.cases[index].subject.model_dump(mode="json")],
        },
    )
    return ReviewSnapshot(
        bank=snapshot.bank,
        history=snapshot.history.review(record),
        human_sources={**snapshot.human_sources, record.review_id: source},
    )


def predictions(
    store: ArtifactStore, snapshot: ReviewSnapshot, answers: tuple[int, ...], version: str
) -> JudgeRun:
    """Save exact model-version records without constructing or calling a provider."""
    grader_rubric = save(
        store,
        f"grader-rubric-{version}",
        {"version": version, "instructions": "Fixture grader only."},
    )
    rows = []
    for index, answer in enumerate(answers):
        identity = case_key(snapshot.bank, snapshot.bank.cases[index])
        payload = {
            "case_id": identity,
            "answer": answer,
            "probability": 0.75,
            "model": MODEL.model_dump(mode="json"),
            "version": version,
            "grader_rubric": grader_rubric.model_dump(mode="json"),
        }
        source = save(store, f"judge-{version}-{index}", payload)
        rows.append(
            JudgeAnswer.model_validate(
                {"case_id": identity, "answer": answer, "probability": 0.75, "source": source}
            )
        )
    return JudgeRun(
        bank_sha256=sha256_json(snapshot.bank),
        model=MODEL,
        version=version,
        grader_rubric=grader_rubric,
        answers=tuple(rows),
    )


def test_zero_human_and_author_model_proposals_never_become_metrics(tmp_path: Path) -> None:
    """All pending cases remain unknown, even with confident grader answers."""
    store, snapshot = study(tmp_path)
    run = predictions(store, snapshot, (1, 1, 1, 1), "v1")
    report = measure_reviews(store, snapshot, run)
    assert report["calibration"] == "not_estimated"
    assert report["verified_human_records"] == 0
    measurements = report["measurements"]
    assert isinstance(measurements, list)
    assert all(isinstance(row, dict) and row["human_labelled"] == 0 for row in measurements)
    assert all(
        isinstance(row, dict)
        and isinstance(row["agreement"], dict)
        and row["agreement"]["estimate"] is None
        for row in measurements
    )
    assert export_gold(store, snapshot)["populations"] == {"real": [], "synthetic": []}
    assert disagreement_drafts(store, snapshot, run) == ()
    for index, authority in enumerate(("model", "author_proposed")):
        snapshot = label(store, snapshot, index, 1, authority=authority)
    assert measure_reviews(store, snapshot, run)["verified_human_records"] == 0


def test_dedup_keeps_distinct_predicates_and_separates_synthetic_gold(tmp_path: Path) -> None:
    """One source can carry several predicates; exact repeats cannot inflate support."""
    store, snapshot = study(tmp_path)
    bank = ReviewBank.model_validate(
        {**snapshot.bank.model_dump(), "cases": (*snapshot.bank.cases, snapshot.bank.cases[0])}
    )
    snapshot = ReviewSnapshot(bank=bank)
    for index in range(4):
        snapshot = label(store, snapshot, index, 1)
    run = predictions(store, snapshot, (1, 0, 0, 1), "v1")
    report = measure_reviews(store, snapshot, run)
    gold = export_gold(store, snapshot)
    populations = gold["populations"]
    assert isinstance(populations, dict)
    real, synthetic = populations["real"], populations["synthetic"]
    assert isinstance(real, list) and len(real) == 3
    assert isinstance(synthetic, list) and len(synthetic) == 1
    rows = report["measurements"]
    assert isinstance(rows, list) and len(rows) == 4
    bad, good, description = rows[0], rows[2], rows[3]
    assert isinstance(bad, dict) and bad["planned"] == 1
    assert isinstance(bad["agreement"], dict) and bad["agreement"]["independent_units"] == 1
    assert isinstance(good, dict) and isinstance(good["false_fail_rate"], dict)
    assert good["false_fail_rate"]["estimate"] == 1
    assert isinstance(description, dict) and description["catch_rate"] is None
    assert description["false_fail_rate"] is None


def test_conflicting_reviews_need_adjudication_and_corrections_stay_linear(tmp_path: Path) -> None:
    """Shared history retains dissent and refuses competing active labels/correction forks."""
    store, snapshot = study(tmp_path)
    snapshot = review(store, snapshot, 0, 1, "alice")
    snapshot = review(store, snapshot, 0, 0, "bob")
    assert export_gold(store, snapshot)["populations"] == {"real": [], "synthetic": []}
    with pytest.raises(ValueError, match="every independent review"):
        label(store, snapshot, 0, 1, adjudicates=("review-0-alice",))
    snapshot = label(store, snapshot, 0, 1, adjudicates=("review-0-alice", "review-0-bob"))
    with pytest.raises(ValueError, match="correct an existing"):
        label(store, snapshot, 0, 0, suffix="conflict")
    snapshot = label(
        store,
        snapshot,
        0,
        0,
        suffix="corrected",
        supersedes="label-0-first",
        adjudicates=("review-0-alice", "review-0-bob"),
    )
    with pytest.raises(ValueError, match="only one direct correction"):
        label(store, snapshot, 0, 1, suffix="fork", supersedes="label-0-first")
    assert len(snapshot.history.reviews) == 2 and len(snapshot.history.scores) == 2
    gold = export_gold(store, snapshot)["populations"]
    assert isinstance(gold, dict) and isinstance(gold["real"], list)
    assert gold["real"][0] == {
        "case_id": case_key(snapshot.bank, snapshot.bank.cases[0]),
        "case": snapshot.bank.cases[0].model_dump(mode="json"),
        "answer": 0,
        "human_label": snapshot.history.active_scores()[0].model_dump(mode="json"),
        "human_source": snapshot.human_sources["label-0-corrected"].model_dump(mode="json"),
    }


def test_missing_evidence_and_unknown_labels_remain_unassessable(tmp_path: Path) -> None:
    """A claimed review with an unresolved witness cannot contribute a human metric."""
    store, snapshot = study(tmp_path)
    snapshot = label(store, snapshot, 0, 1)
    snapshot = label(store, snapshot, 1, -1)
    source = snapshot.human_sources["label-0-first"]
    missing = source.model_copy(update={"sha256": "f" * 64})
    snapshot = snapshot.model_copy(
        update={"human_sources": {**snapshot.human_sources, "label-0-first": missing}}
    )
    run = predictions(store, snapshot, (1, 1, 1, 1), "v1")
    report = measure_reviews(store, snapshot, run)
    assert report["calibration"] == "not_estimated"
    issues = report["human_evidence_issues"]
    assert isinstance(issues, dict) and "label-0-first" in issues
    assert export_gold(store, snapshot)["populations"] == {"real": [], "synthetic": []}


def test_question_rubric_and_prediction_version_drift_refused(tmp_path: Path) -> None:
    """Persisted values, not caller declarations, bind all versioned evidence."""
    store, snapshot = study(tmp_path)
    run = predictions(store, snapshot, (1,), "v1")
    drifted = snapshot.bank.questions.model_copy(update={"version": 2})
    changed = snapshot.model_copy(
        update={"bank": snapshot.bank.model_copy(update={"questions": drifted})}
    )
    assert case_key(changed.bank, changed.bank.cases[0]) != case_key(
        snapshot.bank, snapshot.bank.cases[0]
    )
    with pytest.raises(ValueError, match="question definition differs"):
        measure_reviews(store, changed, run)
    rubric = snapshot.bank.target_rubric.model_copy(update={"sha256": "f" * 64})
    changed = snapshot.model_copy(
        update={"bank": snapshot.bank.model_copy(update={"target_rubric": rubric})}
    )
    with pytest.raises(ValueError, match="digest differs"):
        measure_reviews(store, changed, run)
    with pytest.raises(ValueError, match="model/version/prediction"):
        measure_reviews(store, snapshot, run.model_copy(update={"version": "v2"}))
    with pytest.raises(ValueError, match="another frozen bank"):
        measure_reviews(store, snapshot, run.model_copy(update={"bank_sha256": "f" * 64}))


def test_split_lineage_and_selection_exposure_are_frozen(tmp_path: Path) -> None:
    """Shared task/lineage/source groups may never straddle development and test."""
    _, snapshot = study(tmp_path)
    real = snapshot.bank.cases[0]
    for changed in (
        real.model_copy(update={"partition": "test", "question_id": "good"}),
        snapshot.bank.cases[3].model_copy(update={"lineages": real.lineages}),
        snapshot.bank.cases[3].model_copy(update={"task_id": real.task_id}),
    ):
        with pytest.raises(ValueError, match="share evidence|conflicting metadata"):
            ReviewBank.model_validate(
                {**snapshot.bank.model_dump(), "cases": (*snapshot.bank.cases, changed)}
            )
    with pytest.raises(ValueError, match="exposed to selection"):
        ReviewBank.model_validate(
            {**snapshot.bank.model_dump(), "selection_lineages": ("synthetic-session",)}
        )


def test_paired_regression_and_development_only_existing_curation_export(tmp_path: Path) -> None:
    """Paired counts use one gold revision and feedback never exposes held-out records."""
    store, snapshot = study(tmp_path)
    for index in range(4):
        snapshot = label(store, snapshot, index, 1)
    before = predictions(store, snapshot, (0, 1, -1, 1), "before")
    after = predictions(store, snapshot, (1, 0, 1, 0), "after")
    assert before.grader_rubric != after.grader_rubric
    frozen_gold = snapshot.gold_version
    result = paired_retest(store, snapshot, before, after)
    assert result["gold_version"] == frozen_gold == snapshot.gold_version
    strata = result["strata"]
    assert isinstance(strata, dict)
    assert isinstance(strata["bad/development/real"], dict)
    assert strata["bad/development/real"]["better"] == 1
    assert isinstance(strata["good/development/real"], dict)
    assert strata["good/development/real"]["worse"] == 1
    assert isinstance(strata["description/development/real"], dict)
    assert strata["description/development/real"]["unassessable"] == 1
    assert result["detection_floor"] == "not_estimated"
    drafts = disagreement_drafts(store, snapshot, after)
    assert len(drafts) == 1
    assert drafts[0].kind == "rubric_problem" and drafts[0].hint["status"] == "draft_advisory_only"
    assert drafts[0].hint["human_sources"] == {
        "label-1-first": snapshot.human_sources["label-1-first"].model_dump(mode="json")
    }
    assert drafts[0].hint["target_rubric"] == snapshot.bank.target_rubric.model_dump(mode="json")
    reference = save_review_snapshot(store, snapshot, code_revision="fixture")
    assert save_review_snapshot(store, snapshot, code_revision="fixture") == reference
    loaded = ReviewSnapshot.model_validate_json(
        store.read_bytes(reference.artifact_id, "review.json")
    )
    assert loaded == snapshot
    assert paired_retest(store, loaded, before, after) == result


def test_related_step_questions_use_one_lineage_unit(tmp_path: Path) -> None:
    """Two answers from one task are two slots and one independent uncertainty unit."""
    store, snapshot = study(tmp_path)
    original = snapshot.bank.cases[0]
    step = original.subject.model_copy(
        update={"path": ("steps", 0), "sha256": sha256_json({"tool": "inspect"})}
    )
    additional = original.model_copy(update={"subject": step})
    bank = ReviewBank.model_validate(
        {**snapshot.bank.model_dump(), "cases": (*snapshot.bank.cases, additional)}
    )
    snapshot = ReviewSnapshot(bank=bank)
    snapshot = label(store, snapshot, 0, 1)
    snapshot = label(store, snapshot, 4, 1)
    before = predictions(store, snapshot, (1, 1, 1, 1, 1), "before")
    after = predictions(store, snapshot, (1, 1, 1, 1, 0), "after")
    report = measure_reviews(store, snapshot, after)
    measurements = report["measurements"]
    assert isinstance(measurements, list) and isinstance(measurements[0], dict)
    measured = measurements[0]
    assert measured["planned"] == 2 and measured["distinct_tasks"] == 1
    agreement = measured["agreement"]
    assert isinstance(agreement, dict)
    assert agreement["estimate"] == 0.5 and agreement["independent_units"] == 1
    paired = paired_retest(store, snapshot, before, after)["strata"]
    assert isinstance(paired, dict) and isinstance(paired["bad/development/real"], dict)
    row = paired["bad/development/real"]
    assert row["unchanged"] == 1 and row["worse"] == 1
    interval = row["after_minus_before"]
    assert isinstance(interval, dict) and interval["independent_units"] == 1


def test_valid_new_definition_requires_fresh_human_review(tmp_path: Path) -> None:
    """A valid new bank cannot silently reuse labels from the preceding proposition version."""
    store, snapshot = study(tmp_path)
    snapshot = label(store, snapshot, 0, 1)
    questions = snapshot.bank.questions.model_copy(update={"version": 2})
    source = save(store, "question-definition-v2", questions.model_dump(mode="json"))
    bank = ReviewBank.model_validate(
        {**snapshot.bank.model_dump(), "questions": questions, "questions_source": source}
    )
    changed = ReviewSnapshot(
        bank=bank, history=snapshot.history, human_sources=snapshot.human_sources
    )
    with pytest.raises(ValueError, match="human target differs"):
        export_gold(store, changed)


def test_protected_source_cannot_enter_development_feedback(tmp_path: Path) -> None:
    """Existing source-access protection overrides an earlier diagnostic declaration."""
    store, snapshot = study(tmp_path)
    snapshot = label(store, snapshot, 0, 1)
    run = predictions(store, snapshot, (0,), "v1")
    declare_source_access(
        store,
        snapshot.bank.cases[0].subject.source,
        scope="protected",
        rationale="Protected evaluation evidence.",
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="protected diagnostic source"):
        disagreement_drafts(store, snapshot, run)
