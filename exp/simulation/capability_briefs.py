"""Versioned future-user needs, assumptions and evidence for the existing task factory."""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase


class CapabilityAssumption(ContractModel):
    """One explicit source of confidence or uncertainty in a proposed capability.

    Attributes:
        statement: Behavioral or product assumption being made.
        authority: Observation, authored contract or explicitly unresolved speculation.
        sources: Exact evidence, required for observed and authored claims.
    """

    statement: str = Field(min_length=1)
    authority: Literal["recorded_observation", "authored_contract", "speculation"]
    sources: tuple[ArtifactInput, ...] = ()

    @model_validator(mode="after")
    def evidenced(self) -> Self:
        """Do not let a source label substitute for actual recorded evidence."""
        if self.authority != "speculation" and not self.sources:
            raise ValueError("observed and authored assumptions require immutable sources")
        return self


class RequiredInformation(ContractModel):
    """One necessary fact and the permitted observation that can supply it.

    Attributes:
        fact: Information necessary to complete the intended task.
        obtain_via: Permitted tool or public context route that supplies it.
    """

    fact: str = Field(min_length=1)
    obtain_via: str = Field(min_length=1)


class CapabilityBrief(ContractModel):
    """A future-user hypothesis that supplies intent without dictating an actor's solution.

    Attributes:
        category: Regression, frontier or explicitly exploratory research.
        novelty: Existing workflow, novel composition, new situation or authored interface.
        user: Intended beneficiary.
        outcome: Useful result and the task's immutable intent.
        capability: Skill being tested.
        gap: What existing coverage does not establish.
        required_tools: Exact tools that must be available.
        required_information: Each needed fact mapped to its permitted acquisition route.
        allowed_changes: State and effects the actor may change.
        preserved: State and effects that must remain unchanged.
        unavailable_behavior: Correct clarification, refusal or escalation when requirements fail.
        success_evidence: Observable distinctions between genuine success and an attractive miss.
        assumptions: Authority and uncertainty behind the proposed behavior.
    """

    category: Literal["regression", "frontier", "exploratory"]
    novelty: Literal[
        "existing_workflow", "novel_composition", "new_situation", "authored_interface"
    ]
    user: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    gap: str = Field(min_length=1)
    required_tools: tuple[str, ...]
    required_information: tuple[RequiredInformation, ...]
    allowed_changes: tuple[str, ...]
    preserved: tuple[str, ...]
    unavailable_behavior: str = Field(min_length=1)
    success_evidence: tuple[str, ...] = Field(min_length=1)
    assumptions: tuple[CapabilityAssumption, ...] = Field(min_length=1)

    def blockers(self) -> tuple[str, ...]:
        """Unresolved assumptions prevent qualification regardless of actor success rate."""
        return (
            *(
                f"unresolved assumption: {row.statement}"
                for row in self.assumptions
                if row.authority == "speculation"
            ),
            *(
                ("exploratory brief has not established a qualified capability scope",)
                if self.category == "exploratory"
                else ()
            ),
        )

    def require_task(self, task: TaskCase, intent: str) -> None:
        """Refuse intent drift and missing interfaces before task authoring dispatch."""
        if intent != self.outcome:
            raise ValueError("task intent differs from its capability brief")
        missing = set(self.required_tools) - {tool.name for tool in task.tools}
        if missing:
            raise ValueError(
                f"capability requires unavailable tools {sorted(missing)}; "
                "supply them or revise the brief"
            )


def persist_capability_brief(
    store: ArtifactStore, brief: CapabilityBrief, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Persist intent and exact sources without upgrading speculative assumptions."""
    sources = validate_brief_sources(store, brief)
    identity = stable_id("capability-brief", {"brief": sha256_json(brief)})
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type="capability-brief",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=created_at, code_revision=code_revision, inputs=sources
        ),
        files={"brief.json": brief},
    )
    return artifact_input(manifest)


def validate_brief_sources(
    store: ArtifactStore, brief: CapabilityBrief
) -> tuple[ArtifactInput, ...]:
    """Validate original references for publication or credential-free CLI preflight."""
    sources = tuple(
        sorted(
            {source for row in brief.assumptions for source in row.sources},
            key=lambda ref: ref.artifact_id,
        )
    )
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("capability source differs from its saved digest")
    _require_recorded_observations(store, brief)
    return sources


def load_capability_brief(store: ArtifactStore, reference: ArtifactInput) -> CapabilityBrief:
    """Verify a brief and all declared source identities before using it as author context."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "capability-brief":
        raise ValueError("capability brief differs from its saved artifact")
    brief = CapabilityBrief.model_validate_json(
        store.read_bytes(reference.artifact_id, "brief.json")
    )
    sources = {source for row in brief.assumptions for source in row.sources}
    if sources != set(manifest.inputs):
        raise ValueError("capability brief lost its declared source dependencies")
    for source in sources:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("capability source differs from its saved digest")
    _require_recorded_observations(store, brief)
    return brief


def _require_recorded_observations(store: ArtifactStore, brief: CapabilityBrief) -> None:
    """Refuse a recorded observation backed by simulated, generated or unclassified output.

    Raises:
        ValueError: A ``recorded_observation`` source's provenance leaves recorded evidence.
    """
    # Admission imports evaluation tasks, which import this module, so import it here.
    from exp.simulation.evaluation.admission import generated_ancestor

    for row in brief.assumptions:
        if row.authority != "recorded_observation":
            continue
        for source in row.sources:
            if (generated := generated_ancestor(store, source)) is not None:
                raise ValueError(
                    f"recorded observation '{row.statement}' derives from {generated}, which "
                    "is not recorded evidence; cite recorded artifacts or mark it speculation"
                )
