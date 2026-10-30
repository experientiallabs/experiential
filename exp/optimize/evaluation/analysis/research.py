"""Counterexample checks over cited evidence, without learning from misassigned failures.

The checker executes finite data predicates, never generated programs. Authored translations of
task intent and supplied witness outcomes remain visible assumptions. A constructed witness can
refute a proposed algorithmic rule without proving the historical actor actually executed it.
"""

import json
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonValue,
    stable_id,
    validate_artifact_file_path,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.evaluation.tasks import TaskVerifier, VerifierGrade, grade
from exp.simulation.reconstruction.models import path_parts, resolve

RepairDestination = Literal[
    "interface", "retrieval", "simulator", "actor", "verifier", "execution_recovery", "unresolved"
]


class ResearchEvidence(ContractModel):
    """Exact locator in immutable evidence; never a model-generated paraphrase.

    Attributes:
        source: Artifact manifest digest.
        file: JSON artifact file containing the evidence.
        pointer: JSON pointer to the original observation or outcome.
        quote: Optional exact substring required in a text observation.
    """

    source: ArtifactInput
    file: str
    pointer: str
    quote: str | None = Field(default=None, min_length=1, max_length=20000)

    @model_validator(mode="after")
    def locator(self) -> Self:
        """Reject unsafe paths and malformed pointers before reading evidence."""
        validate_artifact_file_path(self.file)
        path_parts(self.pointer)
        return self

    def read(self, store: ArtifactStore) -> JsonValue:
        """Verify the exact source and quote, without interpreting its semantic authority."""
        if artifact_input(store.read(self.source.artifact_id).manifest) != self.source:
            raise ValueError("research source digest changed; select immutable original evidence")
        value = resolve(
            json.loads(store.read_bytes(self.source.artifact_id, self.file)), self.pointer
        )
        if self.quote is not None and (not isinstance(value, str) or self.quote not in value):
            raise ValueError("research quote absent from original evidence; correct its locator")
        return value


class FailureHypothesis(ContractModel):
    """Competing explanation and proposed repair owner, never adjudicated causal blame.

    Attributes:
        id: Local identity used by counterexample proposals.
        explanation: Bounded candidate explanation.
        destination: Subsystem that would change if evidence supports this explanation.
        falsifier: Observation that would undermine the explanation.
    """

    id: str = Field(min_length=1)
    explanation: str = Field(min_length=1, max_length=4000)
    destination: RepairDestination
    falsifier: str = Field(min_length=1, max_length=4000)


class ResearchFailure(ContractModel):
    """Source-bound symptom kept separate from competing explanations.

    Attributes:
        symptom: Observed discrepancy, not an inferred cause.
        observations: Exact original observations supporting the symptom.
        hypotheses: Alternatives retained together, without choosing a convenient cause.
    """

    symptom: str = Field(min_length=1, max_length=4000)
    observations: tuple[ResearchEvidence, ...] = Field(min_length=1, max_length=32)
    hypotheses: tuple[FailureHypothesis, ...] = Field(min_length=1, max_length=16)


class ResearchRequirement(ContractModel):
    """Explicit task semantics or an unresolved requirement blocking assessment.

    Attributes:
        id: Local requirement identity.
        statement: Authored semantic reading, not automatically entailed by a matching quote.
        evidence: Original task wording, with an exact quote for declared requirements.
        status: Declared or unresolved; missing prohibitions must stay unresolved.
        verifier: Existing finite predicate; absent for unresolved requirements.
    """

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=4000)
    evidence: ResearchEvidence
    status: Literal["declared", "unresolved"]
    verifier: TaskVerifier | None = None

    @model_validator(mode="after")
    def authority(self) -> Self:
        """Do not manufacture a check when task semantics are missing."""
        if (self.status == "declared") != (self.verifier is not None):
            raise ValueError("declared requirements need a verifier; unresolved ones must omit it")
        if self.status == "declared" and self.evidence.quote is None:
            raise ValueError("declared requirements need an exact source quotation")
        return self


class CounterexampleProposal(ContractModel):
    """A retained falsification attempt and its evidence, with no generated code execution.

    Attributes:
        id: Distinct attempt identity, including controls.
        hypothesis: Candidate explanation being probed.
        requirement: Frozen requirement used to grade this outcome.
        purpose: Why this observation discriminates between explanations.
        kind: Candidate witness or explicit positive, negative or missing-evidence control.
        basis: Supplied historical outcome, constructed witness or unexecuted proposal.
        outcome: Exact normalized outcome locator; None means not executed.
        construction: How the outcome was obtained and the relation to the original program.
            A transcription is not an execution of that program.
    """

    id: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    purpose: str = Field(min_length=1, max_length=4000)
    kind: Literal["candidate", "positive_control", "negative_control", "missing_evidence_control"]
    basis: Literal["supplied_observation", "constructed_witness", "unexecuted_proposal"]
    outcome: ResearchEvidence | None = None
    construction: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def evidence(self) -> Self:
        """Keep unexecuted proposals separate from saved observations."""
        if (self.basis == "unexecuted_proposal") != (self.outcome is None):
            raise ValueError(
                "unexecuted proposals omit outcomes; other bases require saved outcomes"
            )
        return self


class ProvisionalRepair(ContractModel):
    """Proposed research revision, deliberately outside active predictor memory.

    Attributes:
        hypothesis: Which candidate explanation motivates this revision.
        destination: Must agree with that explanation's repair owner.
        change: Scoped proposed revision to investigate next.
        remaining_uncertainty: What this research does not establish.
    """

    hypothesis: str
    destination: RepairDestination
    change: str = Field(min_length=1, max_length=4000)
    remaining_uncertainty: str = Field(min_length=1, max_length=4000)


class FailureResearchSpec(ContractModel):
    """Finite source-bound research pass; no model, training or memory-write authority.

    Attributes:
        failure: Symptom and competing explanations.
        requirements: Frozen interpretations retained separately from outcome observations.
        counterexamples: Every proposed attempt, including controls and missing results.
        revision: Proposed change, held as provisional research only.
        parent: Optional prior research revision, preserved as a dependency rather than replaced.
    """

    failure: ResearchFailure
    requirements: tuple[ResearchRequirement, ...] = Field(min_length=1, max_length=32)
    counterexamples: tuple[CounterexampleProposal, ...] = Field(min_length=1, max_length=128)
    revision: ProvisionalRepair
    parent: ArtifactInput | None = None

    @model_validator(mode="after")
    def linked(self) -> Self:
        """Require complete unique inventories and keep actor repairs out of simulator routing."""
        for rows in (self.failure.hypotheses, self.requirements, self.counterexamples):
            if len({row.id for row in rows}) != len(rows):
                raise ValueError("research IDs must be unique within each inventory")
        hypotheses = {row.id: row for row in self.failure.hypotheses}
        requirements = {row.id for row in self.requirements}
        if any(
            row.hypothesis not in hypotheses or row.requirement not in requirements
            for row in self.counterexamples
        ):
            raise ValueError("counterexample names an absent hypothesis or requirement")
        if (
            self.revision.hypothesis not in hypotheses
            or hypotheses[self.revision.hypothesis].destination != self.revision.destination
        ):
            raise ValueError("provisional repair must retain its hypothesis destination")
        return self


class CounterexampleCheck(ContractModel):
    """One computed check, preserving evidence absence rather than guessing success.

    Attributes:
        id: Proposal identity in original order.
        assessment: Existing goal/policy check, absent if intent or execution is missing.
        result: Bounded verdict about this supplied observation, not historical actor blame.
        reason: Explicit missing requirement/evidence or the finite predicate result.
    """

    id: str
    assessment: VerifierGrade | None
    result: Literal["violates_declared_requirement", "not_falsified", "unassessable"]
    reason: str


class FailureResearchReport(ContractModel):
    """All attempts and a provisional research revision, never a learning approval.

    Attributes:
        spec: Complete frozen proposal and requirement inventory.
        checks: All counterexamples and controls, in original order.
        controls: Per-requirement control result; these are not protected independent labels.
        revision_status: Provisional only with a violating candidate and passing controls.
        memory_published: Always false; this operation cannot write active simulator memory.
        attribution: Causal blame remains unresolved even when a witness fails a predicate.
        scope: Fixed limitations of deterministic checking over authored interpretations.
    """

    spec: FailureResearchSpec
    checks: tuple[CounterexampleCheck, ...]
    controls: dict[str, Literal["passed", "failed", "incomplete"]]
    revision_status: Literal["provisional", "blocked"]
    memory_published: Literal[False] = False
    attribution: Literal["unresolved"] = "unresolved"
    scope: str = (
        "Development research; deterministic checks of supplied outcomes against authored "
        "task interpretations. Quotation matching is not semantic validation. Constructed "
        "witnesses do not prove original program execution, native truth or causal utility. "
        "No protected-label qualification, training or predictor memory publication."
    )


def check_failure_research(
    store: ArtifactStore, spec: FailureResearchSpec
) -> FailureResearchReport:
    """Recompute every check from frozen sources with the maintained predicate interpreter."""
    for evidence in (*spec.failure.observations, *(row.evidence for row in spec.requirements)):
        evidence.read(store)
    if spec.parent is not None:
        manifest = store.read(spec.parent.artifact_id).manifest
        if artifact_input(manifest) != spec.parent or manifest.artifact_type != "failure-research":
            raise ValueError(
                "research parent differs; select an immutable failure-research revision"
            )
    requirements = {row.id: row for row in spec.requirements}
    checks: list[CounterexampleCheck] = []
    for proposal in spec.counterexamples:
        requirement = requirements[proposal.requirement]
        outcome = None if proposal.outcome is None else proposal.outcome.read(store)
        if outcome is not None and not isinstance(outcome, dict):
            raise ValueError("research outcome must be a normalized object; fix its source locator")
        assessment = (
            grade(requirement.verifier, outcome)
            if requirement.verifier is not None and isinstance(outcome, dict)
            else None
        )
        status = "unassessable" if assessment is None else assessment.status
        checks.append(
            CounterexampleCheck(
                id=proposal.id,
                assessment=assessment,
                result=(
                    "violates_declared_requirement"
                    if status == "fail"
                    else "not_falsified"
                    if status == "pass"
                    else "unassessable"
                ),
                reason=(
                    assessment.reason
                    if assessment is not None
                    else "requirement unresolved or execution unavailable"
                ),
            )
        )
    controls: dict[str, Literal["passed", "failed", "incomplete"]] = {}
    expected = {
        "positive_control": "pass",
        "negative_control": "fail",
        "missing_evidence_control": "unassessable",
    }
    for requirement in spec.requirements:
        selected = [
            (proposal, check)
            for proposal, check in zip(spec.counterexamples, checks, strict=True)
            if proposal.requirement == requirement.id and proposal.kind != "candidate"
        ]
        if {proposal.kind for proposal, _ in selected} != set(expected):
            controls[requirement.id] = "incomplete"
        elif any(
            check.assessment is None or check.assessment.status != expected[proposal.kind]
            for proposal, check in selected
        ):
            controls[requirement.id] = "failed"
        else:
            controls[requirement.id] = "passed"
    candidates = [
        (proposal, check)
        for proposal, check in zip(spec.counterexamples, checks, strict=True)
        if proposal.kind == "candidate" and proposal.hypothesis == spec.revision.hypothesis
    ]
    supported = (
        bool(candidates)
        and all(
            controls[proposal.requirement] == "passed" and check.result != "unassessable"
            for proposal, check in candidates
        )
        and any(check.result == "violates_declared_requirement" for _, check in candidates)
    )
    return FailureResearchReport(
        spec=spec,
        checks=tuple(checks),
        controls=controls,
        revision_status="provisional" if supported else "blocked",
    )


def persist_failure_research(
    store: ArtifactStore, spec: FailureResearchSpec, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Keep proposals, rejections and revisions immutable; replay always recomputes the checks."""
    report = check_failure_research(store, spec)
    locators = (
        *spec.failure.observations,
        *(row.evidence for row in spec.requirements),
        *(row.outcome for row in spec.counterexamples if row.outcome is not None),
    )
    sources = {row.source for row in locators}
    if spec.parent is not None:
        sources.add(spec.parent)
    inputs = tuple(sorted(sources, key=lambda ref: ref.artifact_id))
    identity = stable_id(
        "failure-research", {"spec": spec.model_dump(mode="json"), "code": code_revision}
    )
    if store.exists(identity):
        if (
            FailureResearchReport.model_validate_json(store.read_bytes(identity, "report.json"))
            != report
            or store.read(identity).manifest.inputs != inputs
        ):
            raise ValueError("saved research differs from recomputed evidence; inspect its sources")
        return artifact_input(store.read(identity).manifest)
    return artifact_input(
        store.write_json(
            artifact_id=identity,
            artifact_type="failure-research",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision, inputs=inputs
            ),
            files={"report.json": report},
        )
    )
