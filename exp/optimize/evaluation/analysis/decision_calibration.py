"""Offline fixed-question decision measurements, never a specialist activation gate.

Teacher answers are retained separately from independently supplied labels. No threshold is
searched on confirmation data, no missing slot disappears, and no model is trained here.
"""

import json
import math
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.evaluations.uncertainty import Z95_ONE_SIDED, wilson_interval
from exp.common.project import ArtifactStore, artifact_input
from exp.common.project.causality import cites
from exp.optimize.evaluation.analysis.memory_evidence import EvidenceLocator, located_value
from exp.runtime.agents.cascade import delegates


class DecisionDefinition(ContractModel):
    """Exact behavior to which a calibration measurement applies.

    Attributes:
        student: Exact model/readout configuration, including any adapter.
        teacher: Teacher model profile, independent of the evaluation label authority.
        fallback: Exact fallback policy and profile.
        question_sha256: Decision-question definition.
        renderer_sha256: Prompt and option rendering definition.
        candidate_builder_sha256: Candidate generation and ordering definition.
        tools_sha256: Tool schemas and capability restrictions.
        knowledge_sha256: Frozen memory, retrieval and knowledge configuration.
        policy_sha256: Hard safety and eligibility restrictions, applied before deferral.
    """

    student: JsonObject
    teacher: JsonObject
    fallback: JsonObject
    question_sha256: Sha256
    renderer_sha256: Sha256
    candidate_builder_sha256: Sha256
    tools_sha256: Sha256
    knowledge_sha256: Sha256
    policy_sha256: Sha256


class DecisionPrediction(ContractModel):
    """One independently saved model answer, not a ground-truth label.

    Attributes:
        source: Exact saved answer value.
        selected: Chosen candidate identity or OTHER.
        probabilities: Normalized distribution over the ordered candidates plus OTHER.
    """

    source: EvidenceLocator
    selected: str
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def valid_distribution(self) -> "DecisionPrediction":
        """Reject ordinal scores, NaNs, invalid confidence and inconsistent selections."""
        if (
            not self.probabilities
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.probabilities.values())
            or not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-6)
        ):
            raise ValueError("decision probabilities must be finite and sum to one")
        if self.selected not in self.probabilities:
            raise ValueError("selected answer is absent from the distribution")
        return self


class DecisionLabel(ContractModel):
    """An external label, distinct from teacher output and student predictions.

    Attributes:
        source: Exact independently supplied label, verified against its saved value.
        target: Correct candidate identity; may be absent from the candidate pool.
        authority: Declared independent label method, not inferred from model agreement.
    """

    source: EvidenceLocator
    target: str = Field(min_length=1)
    authority: Literal["native_verifier", "expert_annotation"]


class DecisionCase(ContractModel):
    """One planned shadow or offline slot, including unavailable predictions and labels.

    Attributes:
        id: Unique planned case coordinate.
        partition: Train, threshold-calibration or protected confirmation role.
        leakage_groups: Complete declared source/task/entity/dependency groups for this case.
        question: Exact saved decision question, permitted state and ordered candidates.
        candidates: Ordered candidate identities without a label-informed insertion.
        student: Retained student answer, absent on failed or missing inference.
        teacher: Retained teacher supervision, never used as an independent correctness label.
        label: Independent correctness evidence, absent when not established.
        hard_allowed: Whether the frozen hard policy permits this student action.
    """

    id: str = Field(min_length=1)
    partition: Literal["train", "calibration", "confirmation"]
    leakage_groups: tuple[str, ...] = Field(min_length=1)
    question: EvidenceLocator
    candidates: tuple[str, ...] = Field(min_length=1)
    student: DecisionPrediction | None = None
    teacher: DecisionPrediction | None = None
    label: DecisionLabel | None = None
    hard_allowed: bool

    @model_validator(mode="after")
    def exact_options(self) -> "DecisionCase":
        """Bind both distributions to the original options, keeping OTHER explicit."""
        if len(set(self.candidates)) != len(self.candidates) or "OTHER" in self.candidates:
            raise ValueError("candidates must be unique and reserve OTHER")
        expected = {*self.candidates, "OTHER"}
        for prediction in (self.student, self.teacher):
            if prediction and set(prediction.probabilities) != expected:
                raise ValueError("prediction distribution differs from candidate inventory")
        return self


class DecisionCalibration(ContractModel):
    """Frozen fixed-question study, with authored policy evaluated without tuning it.

    Attributes:
        definition: Complete policy identity; any change requires another study.
        threshold: Preselected acceptance confidence, never optimized on these results.
        minimum_support: Minimum distinct declared lineage components supporting confirmation.
        maximum_error: Maximum one-sided 95 percent Wilson upper error bound.
        cases: Complete planned inventory, including missing outcomes.
        registration: Prospective immutable definition and threshold; every confirmation
            question cites it in its manifest inputs, which proves the question came after.
    """

    definition: DecisionDefinition
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    minimum_support: int = Field(gt=0)
    maximum_error: float = Field(ge=0, lt=1, allow_inf_nan=False)
    cases: tuple[DecisionCase, ...] = Field(min_length=1)
    registration: ArtifactInput


class PartitionMeasurement(ContractModel):
    """Denominators and diagnostics for one source-disjoint partition.

    Attributes:
        partition: Declared split role.
        planned: All scheduled cases, including failures.
        predicted: Cases with retained student predictions.
        labelled: Cases with independent labels.
        candidate_available: Labels present in the original candidate pool.
        answered_labelled: Labelled cases with saved student answers.
        correct: Correct classifications, counting OTHER when the target is outside the pool.
        brier_mean: Multiclass Brier score on answered labelled cases, or unknown.
        accepted: Predicted, policy-allowed, above-threshold non-OTHER answers.
        accepted_labelled: Accepted cases with an independent label.
        accepted_errors: Incorrect accepted labelled cases.
        independent_units: Connected declared lineage components with accepted labelled answers.
        error_upper_95: Wilson bound over components, counting any accepted error as failure.
    """

    partition: str
    planned: int
    predicted: int
    labelled: int
    candidate_available: int
    answered_labelled: int
    correct: int
    brier_mean: float | None
    accepted: int
    accepted_labelled: int
    accepted_errors: int
    independent_units: int
    error_upper_95: float | None


class DecisionCalibrationReport(ContractModel):
    """Measurements only; closed-loop confirmation and serving remain separate work.

    Attributes:
        study_sha256: Identity of every supplied configuration, case, prediction and label.
        partitions: All split measurements with their complete denominators.
        confirmation_supported: Whether this fixed-question confirmation meets declared bounds.
        blockers: Explicit limitations preventing a supported fixed-question claim.
        scope: No runtime activation, carried-workflow success, or training-lift claim.
    """

    study_sha256: Sha256
    partitions: tuple[PartitionMeasurement, ...]
    confirmation_supported: bool
    blockers: tuple[str, ...]
    scope: str = (
        "fixed-question, declared-lineage-disjoint evidence only; label authority and complete "
        "lineage declarations require independent audit; no carried-workflow qualification, "
        "runtime activation, probability recalibration or training-lift claim"
    )


def _verify_case(store: ArtifactStore, case: DecisionCase, definition: DecisionDefinition) -> None:
    """Verify question and answers against exact saved values and separate label ancestry."""
    question = located_value(store, case.question)
    if not isinstance(question, dict) or question.get("candidates") != list(case.candidates):
        raise ValueError("saved question differs from ordered candidates")
    if not isinstance(question.get("question"), str) or "permitted_state" not in question:
        raise ValueError("saved decision requires question and permitted state")
    if question.get("definition_sha256") != sha256_json(definition):
        raise ValueError("saved question belongs to a different frozen decision definition")
    if question.get("hard_allowed") is not case.hard_allowed:
        raise ValueError("saved hard-policy decision differs")
    forbidden: set[ArtifactInput] = set()
    for prediction in (case.student, case.teacher):
        if prediction is None:
            continue
        value = located_value(store, prediction.source)
        if value != {"selected": prediction.selected, "probabilities": prediction.probabilities}:
            raise ValueError("saved model answer differs from declared prediction")
        forbidden.add(prediction.source.source)
    if case.label is not None:
        if located_value(store, case.label.source) != case.label.target:
            raise ValueError("saved independent label differs")
        pending = [case.label.source.source]
        seen: set[ArtifactInput] = set()
        while pending:
            source = pending.pop()
            if source in forbidden:
                raise ValueError("independent label depends on a teacher or student answer")
            if source in seen:
                continue
            seen.add(source)
            manifest = store.read(source.artifact_id).manifest
            if artifact_input(manifest) != source:
                raise ValueError("label dependency identity differs")
            pending.extend(manifest.inputs)


def _measure(
    cases: tuple[DecisionCase, ...], partition: str, threshold: float
) -> PartitionMeasurement:
    """Compute fixed-policy measurements while preserving unlabelled and failed slots."""
    rows = [case for case in cases if case.partition == partition]
    labelled = available = answered = correct = accepted = supported = errors = 0
    briers = []
    components: dict[str, set[str]] = {}
    for case in rows:
        nodes = set(case.leakage_groups) | {"question:" + case.question.source.artifact_id}
        for group in tuple(components.values()):
            if nodes & group:
                nodes |= group
        for node in nodes:
            components[node] = nodes
    supported_units: set[str] = set()
    error_units: set[str] = set()
    for case in rows:
        prediction, label = case.student, case.label
        target = (
            None if label is None else label.target if label.target in case.candidates else "OTHER"
        )
        labelled += label is not None
        available += label is not None and label.target in case.candidates
        if prediction and target is not None:
            answered += 1
            correct += prediction.selected == target
            briers.append(
                sum((p - int(key == target)) ** 2 for key, p in prediction.probabilities.items())
            )
        if prediction and delegates(
            prediction.selected,
            prediction.probabilities[prediction.selected],
            threshold=threshold,
            allowed=case.candidates if case.hard_allowed else (),
        ):
            accepted += 1
            if target is not None:
                supported += 1
                errors += prediction.selected != target
                unit = min(components[case.leakage_groups[0]])
                supported_units.add(unit)
                if prediction.selected != target:
                    error_units.add(unit)
    upper = None
    count = len(supported_units)
    if count:
        upper = wilson_interval(len(error_units) / count, count, z=Z95_ONE_SIDED)[1]
    return PartitionMeasurement(
        partition=partition,
        planned=len(rows),
        predicted=sum(c.student is not None for c in rows),
        labelled=labelled,
        candidate_available=available,
        answered_labelled=answered,
        correct=correct,
        brier_mean=sum(briers) / len(briers) if briers else None,
        accepted=accepted,
        accepted_labelled=supported,
        accepted_errors=errors,
        independent_units=count,
        error_upper_95=upper,
    )


def calibrate_decisions(
    store: ArtifactStore, spec: DecisionCalibration
) -> DecisionCalibrationReport:
    """Measure a fixed definition without training, changing thresholds or making calls."""
    if len({case.id for case in spec.cases}) != len(spec.cases):
        raise ValueError("decision inventory repeats a case")
    registered = store.read(spec.registration.artifact_id)
    if artifact_input(registered.manifest) != spec.registration:
        raise ValueError("decision registration identity differs")
    registration = json.loads(store.read_bytes(spec.registration.artifact_id, "registration.json"))
    expected = spec.model_dump(mode="json", exclude={"cases", "registration"})
    expected["inventory"] = [
        case.model_dump(mode="json", include={"id", "partition", "leakage_groups"})
        for case in spec.cases
    ]
    if registration != expected:
        raise ValueError("decision definition or fixed threshold changed after registration")
    groups: dict[str, str] = {}
    for case in spec.cases:
        for group in (*case.leakage_groups, case.question.source.artifact_id):
            if group in groups and groups[group] != case.partition:
                raise ValueError("decision partitions share source or dependency lineage")
            groups[group] = case.partition
        _verify_case(store, case, spec.definition)
        if case.partition == "confirmation" and not cites(
            store, case.question.source.artifact_id, spec.registration
        ):
            # Citing the registration's manifest digest is only possible after it was saved,
            # so it proves order where comparing the two clocks could not.
            raise ValueError(
                f"confirmation question {case.question.source.artifact_id} does not cite the "
                "frozen registration, so nothing shows it was made after it (it may predate "
                "frozen registration); save confirmation questions with the registration among "
                "their inputs"
            )
    partitions = tuple(
        _measure(spec.cases, name, spec.threshold)
        for name in ("train", "calibration", "confirmation")
    )
    confirmation = partitions[-1]
    blockers = []
    if confirmation.accepted != confirmation.accepted_labelled:
        blockers.append("accepted confirmation decisions lack independent labels")
    if confirmation.independent_units < spec.minimum_support:
        blockers.append("insufficient independent confirmation support")
    if confirmation.error_upper_95 is None or confirmation.error_upper_95 > spec.maximum_error:
        blockers.append("confirmation error upper bound exceeds declared risk")
    if confirmation.predicted != confirmation.planned:
        blockers.append("confirmation has missing or failed student predictions")
    return DecisionCalibrationReport(
        study_sha256=sha256_json(spec),
        partitions=partitions,
        confirmation_supported=not blockers,
        blockers=tuple(blockers),
    )
