"""Offline question-case review over the existing human history and artifact owners.

A history target's rollout_id is a review coordinate, not a RolloutArtifact claim. Scores map
explicitly to -1 (unknown), 0 (no), and 1 (yes). Human provenance is a recorded declaration;
verifying its saved bytes does not authenticate a person. Nothing here fits or publishes a judge.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.uncertainty import unit_difference_summary, unit_mean_summary
from exp.common.judging.labels import HumanScore, HumanScoreHistory, ReviewerJudgment
from exp.common.models import ModelSnapshot
from exp.common.project import ArtifactStore, ArtifactStoreError, artifact_input
from exp.optimize.evaluation.analysis.memory_evidence import EvidenceLocator, located_value
from exp.optimize.evaluation.analysis.source_access import require_development_sources
from exp.optimize.evaluation.variance.behavior_profiles import CurationEvidence
from exp.simulation.mining.questions import QuestionBank


class ReviewCase(ContractModel):
    """One trace, step window or session crossed with one exact proposition.

    Attributes:
        subject: Exact retained JSON evidence; its path identifies the trace or step window.
        question_id: Proposition in the bank, never a task-level failure label.
        family_id: Existing authoring family receiving development-only advisory feedback.
        task_id: Original task identity, grouped across all its predicates and steps.
        lineages: Complete declared task, session and dependency groups for split/uncertainty.
        partition: Development may inform drafts; test never enters feedback exports.
        provenance: Real traffic and synthetic diagnostic evidence remain separate populations.
    """

    subject: EvidenceLocator
    question_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    lineages: tuple[str, ...] = Field(min_length=1)
    partition: Literal["development", "test"]
    provenance: Literal["real", "synthetic"]


class ReviewBank(ContractModel):
    """Frozen proposition definitions, evidence coordinates and declared split inventory.

    Attributes:
        questions: Existing typed question bank, including its explicit version and exact text.
        questions_source: Saved value that must exactly equal questions, checked on every read.
        target_rubric: Frozen human correctness standard, independent of grader implementation.
            Changing this standard requires fresh human review; grader edits do not.
        semantics: Meaning of yes for each question; descriptive yes never means task failure.
        cases: Complete inventory, including unanswered and unreviewed slots.
        selection_lineages: Sources exposed to judge selection/training, forbidden in test.
    """

    questions: QuestionBank
    questions_source: EvidenceLocator
    target_rubric: EvidenceLocator
    semantics: dict[str, Literal["failure_yes", "success_yes", "descriptive"]]
    cases: tuple[ReviewCase, ...]
    selection_lineages: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent_inventory(self) -> "ReviewBank":
        """Reject unknown propositions and source/lineage crossover before review."""
        ids = self.questions.question_ids()
        if len(set(ids)) != len(ids) or set(self.semantics) != set(ids):
            raise ValueError("review semantics must name each unique question exactly once")
        roles: dict[str, str] = {}
        seen: dict[str, ReviewCase] = {}
        for case in self.cases:
            if case.question_id not in ids:
                raise ValueError("review case names an unknown question; freeze a matching bank")
            key = case_key(self, case)
            if key in seen and seen[key] != case:
                raise ValueError(
                    "same case/question has conflicting metadata; resolve before freezing"
                )
            seen[key] = case
            nodes = (
                *case.lineages,
                "task:" + case.task_id,
                "artifact:" + case.subject.source.artifact_id,
            )
            for node in nodes:
                if node in roles and roles[node] != case.partition:
                    raise ValueError("development and test share evidence or declared lineage")
                roles[node] = case.partition
            if case.partition == "test" and set(case.lineages) & set(self.selection_lineages):
                raise ValueError(
                    "test lineage was exposed to selection; choose a separate case bank"
                )
        return self

    def unique_cases(self) -> tuple[ReviewCase, ...]:
        """Deduplicate exact case/proposition versions, retaining separate predicates."""
        return tuple({case_key(self, case): case for case in self.cases}.values())


class ReviewCoordinates(ContractModel):
    """Explicit adapter mapping into the shared append-only human history.

    Attributes:
        rubric_id: Original saved rubric artifact, never a manufactured rollout.
        rollout_id: Stable review case/proposition/rubric coordinate, not runtime provenance.
        lineage_id: Digest of the case's complete declared lineage groups.
        dimension_id: Exact question text, bank identity/version and yes semantics.
    """

    rubric_id: str
    rollout_id: str
    lineage_id: str
    dimension_id: str


def question_key(bank: ReviewBank, question_id: str) -> str:
    """Bind a proposition to its wording, bank version and explicit label semantics."""
    question = next(row for row in bank.questions.questions if row.question_id == question_id)
    return stable_id(
        "review-question",
        {
            "question": question.model_dump(mode="json"),
            "bank_id": bank.questions.bank_id,
            "version": bank.questions.version,
            "semantics": bank.semantics[question_id],
        },
    )


def case_key(bank: ReviewBank, case: ReviewCase) -> str:
    """Identify the exact evidence coordinate crossed with predicate and rubric versions."""
    return stable_id(
        "review-case",
        {
            "subject": case.subject.model_dump(mode="json"),
            "question": question_key(bank, case.question_id),
            "target_rubric": bank.target_rubric.model_dump(mode="json"),
        },
    )


def review_coordinates(bank: ReviewBank, case: ReviewCase) -> ReviewCoordinates:
    """Return the only valid shared-history target for this frozen case."""
    return ReviewCoordinates(
        rubric_id=bank.target_rubric.source.artifact_id,
        rollout_id=case_key(bank, case),
        lineage_id=stable_id("review-lineage", {"groups": sorted(set(case.lineages))}),
        dimension_id=question_key(bank, case.question_id),
    )


class HumanReviewEvidence(ContractModel):
    """Explicit human submission wrapper, separate from model/author proposals.

    Attributes:
        authority: Explicit human declaration, never inferred from an expected answer.
        record: Shared review or score record retaining its existing correction semantics.
        comment: Human explanation of the recorded label or review.
        evidence: Exact checked evidence, including the reviewed subject coordinate.
    """

    authority: Literal["human"]
    record: HumanScore | ReviewerJudgment
    comment: str = Field(min_length=1)
    evidence: tuple[EvidenceLocator, ...] = Field(min_length=1)


class ReviewSnapshot(ContractModel):
    """Versioned review import/export using shared correction and adjudication rules.

    Attributes:
        bank: Exact question, rubric, evidence and split inventory.
        history: Independent human reviews plus explicit active labels and their corrections.
        human_sources: Exact saved records keyed by label/review ID; each record is a
            HumanReviewEvidence binding its label, comment and checked subject evidence. Missing or
            unverifiable witnesses remain unknown, never gold. Author/model proposals belong
            outside history and are not inferred from synthetic expected labels.
    """

    bank: ReviewBank
    history: HumanScoreHistory = Field(default_factory=HumanScoreHistory)
    human_sources: dict[str, EvidenceLocator] = Field(default_factory=dict)

    @property
    def gold_version(self) -> Sha256:
        """Bind comparisons to all frozen labels, corrections, reviews and their witnesses."""
        return sha256_json(self)


class JudgeAnswer(ContractModel):
    """A saved grader prediction, kept apart from human label authority.

    Attributes:
        case_id: Exact frozen case/proposition coordinate.
        answer: -1 unknown, 0 no, 1 yes, with no implicit yes-means-failure policy.
        probability: Retained proposition probability, absent when not recorded.
        source: Saved record with case_id, answer, probability, model, version and grader_rubric.
    """

    case_id: str
    answer: Literal[-1, 0, 1]
    probability: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    source: EvidenceLocator


class JudgeRun(ContractModel):
    """One fixed saved judging version with no provider execution or threshold fitting.

    Attributes:
        bank_sha256: Complete frozen inventory to which every prediction applies.
        model: Exact judge model snapshot, including its revision metadata.
        version: Caller-frozen prompt/policy implementation version.
        grader_rubric: Exact grader instructions, allowed to change while target gold stays fixed.
        answers: Saved predictions; failed or absent judgments remain unassessable.
    """

    bank_sha256: Sha256
    model: ModelSnapshot
    version: str = Field(min_length=1)
    grader_rubric: EvidenceLocator
    answers: tuple[JudgeAnswer, ...] = ()


def _verify_bank(store: ArtifactStore, bank: ReviewBank) -> None:
    """Resolve exact source bytes, refusing question/rubric/evidence drift."""
    if located_value(store, bank.questions_source) != bank.questions.model_dump(mode="json"):
        raise ValueError("saved question definition differs; freeze a fresh bank version")
    located_value(store, bank.target_rubric)
    for case in bank.unique_cases():
        located_value(store, case.subject)


def _human_labels(
    store: ArtifactStore, snapshot: ReviewSnapshot
) -> tuple[dict[str, int], dict[str, str], set[str]]:
    """Verify explicit human records and exclude unresolved or unadjudicated targets."""
    cases = {case_key(snapshot.bank, case): case for case in snapshot.bank.unique_cases()}
    valid: set[str] = set()
    issues: dict[str, str] = {}
    records = (*snapshot.history.reviews, *snapshot.history.scores)
    for record in records:
        identity = record.review_id if isinstance(record, ReviewerJudgment) else record.label_id
        case = cases.get(record.rollout_id)
        if case is None or any(
            getattr(record, field) != value
            for field, value in review_coordinates(snapshot.bank, case).model_dump().items()
        ):
            raise ValueError("human target differs from frozen case/question/rubric coordinates")
        if record.score not in {-1, 0, 1}:
            raise ValueError("review adapter scores are -1 unknown, 0 no, or 1 yes")
        source = snapshot.human_sources.get(identity)
        try:
            if source is None:
                issues[identity] = "human witness missing"
                continue
            witness = HumanReviewEvidence.model_validate(located_value(store, source))
            if witness.record != record or case.subject not in witness.evidence:
                issues[identity] = "human witness record or reviewed coordinate differs"
                continue
            for location in witness.evidence:
                located_value(store, location)
            valid.add(identity)
        except (ArtifactStoreError, OSError, ValueError):
            issues[identity] = "human witness unavailable or unverifiable"
    resolved = {
        target.active_label_id
        for target in snapshot.history.reviewed_targets()
        if target.adjudicated and all(row.review_id in valid for row in target.reviews)
    }
    reviewed_targets = {row.rollout_id for row in snapshot.history.reviews}
    labels = {}
    for label in snapshot.history.active_scores():
        if (
            label.label_id in valid
            and label.score != -1
            and (label.rollout_id not in reviewed_targets or label.label_id in resolved)
        ):
            labels[label.rollout_id] = label.score
    return labels, issues, valid


def _predictions(store: ArtifactStore, bank: ReviewBank, run: JudgeRun) -> dict[str, JudgeAnswer]:
    """Verify exact retained predictions without granting model outputs label authority."""
    if run.bank_sha256 != sha256_json(bank):
        raise ValueError("judge run belongs to another frozen bank; retest identical case IDs")
    located_value(store, run.grader_rubric)
    known = {case_key(bank, case) for case in bank.unique_cases()}
    answers = {}
    for row in run.answers:
        if row.case_id not in known or row.case_id in answers:
            raise ValueError("judge answers repeat a case or name an unknown case")
        expected = row.model_dump(mode="json", exclude={"source"})
        expected.update(
            model=run.model.model_dump(mode="json"),
            version=run.version,
            grader_rubric=run.grader_rubric.model_dump(mode="json"),
        )
        if located_value(store, row.source) != expected:
            raise ValueError("saved judge answer differs from declared model/version/prediction")
        answers[row.case_id] = row
    return answers


def _components(cases: tuple[ReviewCase, ...]) -> dict[str, str]:
    """Join shared source/lineage groups transitively before computing uncertainty."""
    groups: list[set[str]] = []
    for case in cases:
        nodes = set(case.lineages) | {
            "task:" + case.task_id,
            "artifact:" + case.subject.source.artifact_id,
        }
        merged = [group for group in groups if group & nodes]
        for group in merged:
            nodes |= group
            groups.remove(group)
        groups.append(nodes)
    return {node: min(group) for group in groups for node in group}


def measure_reviews(store: ArtifactStore, snapshot: ReviewSnapshot, run: JudgeRun) -> JsonObject:
    """Report per-predicate populations with explicit missingness and lineage-unit intervals."""
    _verify_bank(store, snapshot.bank)
    labels, issues, valid = _human_labels(store, snapshot)
    predictions = _predictions(store, snapshot.bank, run)
    cases = snapshot.bank.unique_cases()
    components = _components(cases)
    measurements: list[JsonObject] = []
    for question in snapshot.bank.questions.question_ids():
        for partition in ("development", "test"):
            for origin in ("real", "synthetic"):
                rows = [
                    case
                    for case in cases
                    if (case.question_id, case.partition, case.provenance)
                    == (question, partition, origin)
                ]
                if not rows:
                    continue
                confusion = {"yes_yes": 0, "yes_no": 0, "no_yes": 0, "no_no": 0}
                correct: dict[str, list[float]] = {}
                caught: dict[str, list[float]] = {}
                false_fail: dict[str, list[float]] = {}
                labelled = assessed = 0
                semantics = snapshot.bank.semantics[question]
                for case in rows:
                    identity = case_key(snapshot.bank, case)
                    target, prediction = labels.get(identity), predictions.get(identity)
                    labelled += target is not None
                    if target is None or prediction is None or prediction.answer == -1:
                        continue
                    assessed += 1
                    actual, answer = int(target), prediction.answer
                    confusion[("yes" if actual else "no") + "_" + ("yes" if answer else "no")] += 1
                    unit = components[case.lineages[0]]
                    correct.setdefault(unit, []).append(float(actual == answer))
                    if semantics != "descriptive":
                        failure = 1 if semantics == "failure_yes" else 0
                        bucket = caught if actual == failure else false_fail
                        bucket.setdefault(unit, []).append(float(answer == failure))
                measurements.append(
                    {
                        "question_id": question,
                        "question_version": question_key(snapshot.bank, question),
                        "partition": partition,
                        "provenance": origin,
                        "planned": len(rows),
                        "human_labelled": labelled,
                        "predicted": sum(
                            case_key(snapshot.bank, case) in predictions
                            and predictions[case_key(snapshot.bank, case)].answer != -1
                            for case in rows
                        ),
                        "assessed": assessed,
                        "unassessable": len(rows) - assessed,
                        "distinct_families": len({case.family_id for case in rows}),
                        "distinct_tasks": len({case.task_id for case in rows}),
                        "declared_lineage_units": len(
                            {components[case.lineages[0]] for case in rows}
                        ),
                        "confusion_human_then_judge": confusion,
                        "agreement": unit_mean_summary(correct),
                        "catch_rate": None
                        if semantics == "descriptive"
                        else unit_mean_summary(caught),
                        "false_fail_rate": None
                        if semantics == "descriptive"
                        else unit_mean_summary(false_fail),
                    }
                )
    return {
        "gold_version": snapshot.gold_version,
        "judge": run.model_dump(mode="json", exclude={"answers"}),
        "verified_human_records": len(valid),
        "human_evidence_issues": issues,
        "measurements": measurements,
        "calibration": "not_estimated" if not labels else "agreement_only_no_score_map_fit",
        "detection_floor": "not_estimated",
        "scope": (
            "declared provenance/lineages; no person authentication, selection, "
            "training or release authority"
        ),
    }


def paired_retest(
    store: ArtifactStore, snapshot: ReviewSnapshot, before: JudgeRun, after: JudgeRun
) -> JsonObject:
    """Compare saved judge versions on one identical bank and human-label revision."""
    _verify_bank(store, snapshot.bank)
    labels, issues, _ = _human_labels(store, snapshot)
    left = _predictions(store, snapshot.bank, before)
    right = _predictions(store, snapshot.bank, after)
    cases = snapshot.bank.unique_cases()
    components = _components(cases)
    strata: dict[str, JsonObject] = {}
    for case in cases:
        identity = case_key(snapshot.bank, case)
        key = f"{case.question_id}/{case.partition}/{case.provenance}"
        row = strata.setdefault(key, {"better": 0, "worse": 0, "unchanged": 0, "unassessable": 0})
        target, first, second = labels.get(identity), left.get(identity), right.get(identity)
        disposition = "unassessable"
        if target is not None and first and second and first.answer != -1 and second.answer != -1:
            delta = int(second.answer == target) - int(first.answer == target)
            disposition = "better" if delta > 0 else "worse" if delta < 0 else "unchanged"
        count = row[disposition]
        assert isinstance(count, int)
        row[disposition] = count + 1
    for key, row in strata.items():
        first_units: dict[str, list[float]] = {}
        second_units: dict[str, list[float]] = {}
        for case in cases:
            if key != f"{case.question_id}/{case.partition}/{case.provenance}":
                continue
            identity = case_key(snapshot.bank, case)
            target, first, second = labels.get(identity), left.get(identity), right.get(identity)
            if target is None or not first or not second or -1 in {first.answer, second.answer}:
                continue
            unit = components[case.lineages[0]]
            first_units.setdefault(unit, []).append(float(first.answer == target))
            second_units.setdefault(unit, []).append(float(second.answer == target))
        row["after_minus_before"] = unit_difference_summary(second_units, first_units)
    return {
        "gold_version": snapshot.gold_version,
        "before": before.model_dump(mode="json", exclude={"answers"}),
        "after": after.model_dump(mode="json", exclude={"answers"}),
        "strata": strata,
        "human_evidence_issues": issues,
        "detection_floor": "not_estimated",
        "winner_confirmation": "not_performed",
    }


def export_gold(store: ArtifactStore, snapshot: ReviewSnapshot) -> JsonObject:
    """Export deduplicated human-labeled cases, separating real and synthetic evidence."""
    _verify_bank(store, snapshot.bank)
    labels, issues, _ = _human_labels(store, snapshot)
    populations: dict[str, list[JsonObject]] = {"real": [], "synthetic": []}
    active = {row.rollout_id: row for row in snapshot.history.active_scores()}
    for case in snapshot.bank.unique_cases():
        identity = case_key(snapshot.bank, case)
        if identity in labels:
            populations[case.provenance].append(
                {
                    "case_id": identity,
                    "case": case.model_dump(mode="json"),
                    "answer": labels[identity],
                    "human_label": active[identity].model_dump(mode="json"),
                    "human_source": snapshot.human_sources[active[identity].label_id].model_dump(
                        mode="json"
                    ),
                }
            )
    return {
        "gold_version": snapshot.gold_version,
        "populations": populations,
        "questions_source": snapshot.bank.questions_source.model_dump(mode="json"),
        "target_rubric": snapshot.bank.target_rubric.model_dump(mode="json"),
        "human_evidence_issues": issues,
    }


def disagreement_drafts(
    store: ArtifactStore, snapshot: ReviewSnapshot, run: JudgeRun
) -> tuple[CurationEvidence, ...]:
    """Export development disagreements to the existing curator as draft advisory evidence."""
    _verify_bank(store, snapshot.bank)
    labels, _, _ = _human_labels(store, snapshot)
    predictions = _predictions(store, snapshot.bank, run)
    drafts = []
    for case in snapshot.bank.unique_cases():
        identity = case_key(snapshot.bank, case)
        target, answer = labels.get(identity), predictions.get(identity)
        if (
            case.partition != "development"
            or target is None
            or not answer
            or answer.answer in {-1, target}
        ):
            continue
        record_ids = {
            record.review_id if isinstance(record, ReviewerJudgment) else record.label_id
            for record in (*snapshot.history.reviews, *snapshot.history.scores)
            if record.rollout_id == identity
        }
        sources = {
            case.subject.source,
            snapshot.bank.questions_source.source,
            snapshot.bank.target_rubric.source,
            run.grader_rubric.source,
            answer.source.source,
        }
        for record_id in record_ids:
            location = snapshot.human_sources.get(record_id)
            if location is not None:
                sources.add(location.source)
                witness = HumanReviewEvidence.model_validate(located_value(store, location))
                sources.update(item.source for item in witness.evidence)
        require_development_sources(store, tuple(sources), maximum_artifacts=100_000)
        drafts.append(
            CurationEvidence(
                evidence_id=stable_id(
                    "review-disagreement",
                    {"gold": snapshot.gold_version, "run": sha256_json(run), "case": identity},
                ),
                kind="rubric_problem",
                family_id=case.family_id,
                summary=(
                    f"Human and saved judge disagree on question {case.question_id}; "
                    "review the definition and cited evidence."
                ),
                hint={
                    "status": "draft_advisory_only",
                    "case_id": identity,
                    "gold_version": snapshot.gold_version,
                    "human_answer": target,
                    "judge": run.model_dump(mode="json", exclude={"answers"}),
                    "prediction": answer.model_dump(mode="json"),
                    "target_rubric": snapshot.bank.target_rubric.model_dump(mode="json"),
                    "question_version": question_key(snapshot.bank, case.question_id),
                    "human_sources": {
                        key: value.model_dump(mode="json")
                        for key, value in snapshot.human_sources.items()
                        if key in record_ids
                    },
                },
            )
        )
    return tuple(drafts)


def save_review_snapshot(
    store: ArtifactStore, snapshot: ReviewSnapshot, *, code_revision: str
) -> ArtifactInput:
    """Persist a review-only JSON bank using the existing immutable artifact store."""
    _verify_bank(store, snapshot.bank)
    _human_labels(store, snapshot)
    identity = stable_id("question-case-review", {"snapshot": snapshot.model_dump(mode="json")})
    if store.exists(identity):
        saved = store.read(identity)
        if ReviewSnapshot.model_validate_json(saved.payloads["review.json"]) != snapshot:
            raise ValueError("saved review snapshot differs; import a fresh immutable revision")
        return artifact_input(saved.manifest)
    sources = {snapshot.bank.questions_source.source, snapshot.bank.target_rubric.source}
    sources.update(case.subject.source for case in snapshot.bank.cases)
    # Unresolved records stay in JSON for review, but cannot become verified manifest inputs.
    for location in snapshot.human_sources.values():
        try:
            located_value(store, location)
        except (ArtifactStoreError, OSError, ValueError):
            continue
        sources.add(location.source)
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type="question-case-review",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            inputs=tuple(sorted(sources, key=lambda source: source.artifact_id)),
            code_revision=code_revision,
        ),
        files={"review.json": snapshot},
    )
    return artifact_input(manifest)
