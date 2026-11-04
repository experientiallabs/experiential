"""The one semantic success authority for decision-variance tasks: a frozen LLM episode judge.

Every success label on this path (continuation rates, action-contrast cells, critic targets,
group advantages, variant confirmation, eligibility, training exports and Harbor rewards) comes
from one saved :class:`EpisodeAssessment` made by one frozen :class:`JudgeDefinition` over one
complete :class:`EpisodeEvidence`. Code validates shapes, identities and citations; it never
decides on its own whether a task succeeded. Rule checks keep the simulated world valid; they do
not grade the actor.

- A :class:`JudgeProfile` fixes the judging model, prompt, output schema and sampling before any
  rubric exists, so the rubric author never needs its own output to configure the judge.
- A :class:`JudgeDefinition` binds that profile to one authored rubric, the event contract, the
  task and the world. Changing any of them is a new definition and a new experiment; stale
  labels are never reused under an alias.
- :class:`EpisodeEvidence` is the harness-owned, ordered, append-only record of an episode. Each
  event says who owns it (the actor, the user or simulator, the world, the harness, or an
  experiment that imposed it) and whether it is scored or context only. Inherited source-prefix
  mistakes are context; an imposed contrast action is scored but is not actor-sampled.
- :class:`EpisodeAssessment` is the judge's saved answer: criterion findings with cited event
  ids, an overall verdict, decisive references and limitations, or an explicit operational
  status when the response was malformed or unavailable.

:func:`episode_reward` is the only encoding of a verdict into a number: pass is 1, fail is 0,
and anything else is unscored (None), never 0.5 and never silently dropped from coverage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonValue,
    Sha256,
    sha256_json,
)
from exp.common.models import ModelSnapshot

EVENT_CONTRACT_VERSION = "episode-events-v1"
"""Version of the episode event contract a judge definition is bound to."""

Verdict = Literal["pass", "fail", "unassessable"]
"""The judge's overall semantic verdict."""

CriterionStatus = Literal["met", "not_met", "insufficient_evidence", "not_applicable"]
"""A criterion's finding on one episode."""

EventKind = Literal[
    "user_message",
    "simulator_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "tool_error",
    "receipt",
    "state_snapshot",
    "exogenous_event",
    "final_answer",
    "harness_note",
]
"""What one recorded event is."""

EventOwner = Literal["actor", "user", "simulator", "world", "harness", "experiment"]
"""Who produced an event. ``experiment`` marks an imposed action that no actor sampled."""

EventScope = Literal["context", "scored"]
"""Whether the scored actor is responsible for an event or it only supplies context."""

JUDGE_INPUT_BYTES = 128000
"""Default judge request ceiling; equal to the shared author-request default."""


class JudgeProfile(ContractModel):
    """The judging model and its frozen request settings, fixed before any rubric exists.

    Attributes:
        model_alias: Catalog alias of the judge model.
        model: Exact judge model snapshot.
        prompt_version: Version label of the judge prompt.
        prompt_sha256: Digest of the exact judge prompt text.
        output_schema_sha256: Digest of the judge's structured answer schema.
        temperature: Sampling temperature.
        maximum_output_tokens: Output allowance per judgment.
        reasoning_effort: Requested reasoning effort, or None for the model default.
        evidence_tool_budget: Bounded read-only evidence-tool calls the judge may make before
            answering; 0 means the whole evidence is rendered into one request.
        maximum_input_bytes: Whole serialized request ceiling per judge request. The default
            equals the shared author-request default; changing it is a new profile.
    """

    model_alias: str = Field(min_length=1)
    model: ModelSnapshot
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    maximum_output_tokens: int = Field(default=4096, ge=1, le=65536)
    reasoning_effort: str | None = None
    evidence_tool_budget: int = Field(default=0, ge=0, le=64)
    maximum_input_bytes: int = Field(default=JUDGE_INPUT_BYTES, ge=512, le=512000)

    @property
    def identity(self) -> str:
        """Digest of the complete profile."""
        return sha256_json(self)


class JudgeDefinition(ContractModel):
    """A judge profile bound to one authored rubric, event contract, task and world.

    Attributes:
        profile: The frozen judging profile.
        rubric: Saved rubric artifact the judge applies.
        rubric_sha256: Digest of the rubric file content (not the artifact manifest digest).
        event_contract: Version of the episode event contract.
        task_sha256: Digest of the exact task (instruction and initialization) being judged.
        world_sha256: Digest of the exact frozen world definition.
    """

    profile: JudgeProfile
    rubric: ArtifactInput
    rubric_sha256: Sha256
    event_contract: str = EVENT_CONTRACT_VERSION
    task_sha256: Sha256
    world_sha256: Sha256

    @model_validator(mode="after")
    def _rubric_content_digest(self) -> JudgeDefinition:
        """Refuse a rubric digest that is really the rubric artifact's manifest digest."""
        if self.rubric_sha256 == self.rubric.sha256:
            raise ValueError(
                "rubric_sha256 must digest the rubric file content, not the rubric artifact"
            )
        return self

    @property
    def identity(self) -> str:
        """Digest of the complete definition; any change is a new experiment."""
        return sha256_json(self)


class EpisodeEvent(ContractModel):
    """One harness-recorded event of an episode, in order.

    Attributes:
        event_id: Stable identity within the episode, cited by judge findings.
        sequence: Position in the append-only log, starting at 0.
        kind: What the event is.
        owner: Who produced it.
        scope: Whether the scored actor is responsible for it.
        content: The event's exact recorded content (text, call, reply or state).
        source: Locator of the record the event was taken from.
    """

    event_id: str = Field(min_length=1, max_length=256)
    sequence: int = Field(ge=0)
    kind: EventKind
    owner: EventOwner
    scope: EventScope
    content: JsonValue
    source: str = Field(min_length=1)


class EpisodeEvidence(ContractModel):
    """The complete, ordered, harness-owned record of one episode.

    Attributes:
        episode_id: Stable episode identity.
        task_sha256: Digest of the task the episode attempted.
        world_sha256: Digest of the world it ran in.
        events: Every event in sequence order; context events precede scored ones.
        complete: Whether the harness recorded the whole episode.
        missing: What the harness knows it could not record, when incomplete.
    """

    episode_id: str = Field(min_length=1)
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    events: tuple[EpisodeEvent, ...]
    complete: bool
    missing: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> EpisodeEvidence:
        """Require contiguous sequence numbers, unique ids and context before scored events."""
        if [event.sequence for event in self.events] != list(range(len(self.events))):
            raise ValueError("episode events must be numbered 0..n-1 in order")
        identities = [event.event_id for event in self.events]
        if len(set(identities)) != len(identities):
            raise ValueError("episode event ids must be unique")
        scopes = [event.scope for event in self.events]
        if "scored" in scopes and "context" in scopes[scopes.index("scored") :]:
            raise ValueError("context events must precede every scored event")
        if self.complete and self.missing:
            raise ValueError("a complete episode cannot list missing records")
        return self

    @property
    def identity(self) -> str:
        """Digest of the exact evidence a judgment is about."""
        return sha256_json(self)

    def event_ids(self) -> frozenset[str]:
        """Every citable event identity."""
        return frozenset(event.event_id for event in self.events)


class CriterionFinding(ContractModel):
    """The judge's finding for one rubric criterion on one episode.

    Attributes:
        criterion_id: Rubric criterion identity.
        status: Met, not met, insufficient evidence or not applicable.
        evidence_refs: Event ids supporting the finding.
        rationale: Why, tied to the cited events.
    """

    criterion_id: str = Field(min_length=1)
    status: CriterionStatus
    evidence_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=4000)


class EpisodeAssessment(ContractModel):
    """One saved judgment of one episode by one frozen judge definition.

    Attributes:
        episode_id: The judged episode.
        evidence_sha256: Digest of the exact evidence judged.
        judge: Identity of the judge definition.
        status: ``judged`` when the model returned a valid assessment, ``malformed`` when its
            response failed validation, ``unavailable`` when no response was obtained.
        verdict: The overall verdict; ``unassessable`` whenever status is not ``judged``.
        findings: Criterion findings, empty unless judged.
        decisive_refs: Event ids the verdict rests on.
        limitations: Evidence the judge could not inspect or that was missing.
        response: Saved raw model response, when one exists.
        failure: Why the assessment is not judged, when it is not.
    """

    episode_id: str = Field(min_length=1)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["judged", "malformed", "unavailable"]
    verdict: Verdict
    findings: tuple[CriterionFinding, ...] = ()
    decisive_refs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    response: ArtifactInput | None = None
    failure: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _consistent(self) -> EpisodeAssessment:
        """Keep operational failures unscored and judged answers fully formed."""
        if self.status != "judged":
            if self.verdict != "unassessable" or self.findings or self.failure is None:
                raise ValueError(
                    "an unjudged assessment is unassessable, with a failure and no findings"
                )
        elif self.failure is not None:
            raise ValueError("a judged assessment carries no failure")
        identities = [finding.criterion_id for finding in self.findings]
        if len(set(identities)) != len(identities):
            raise ValueError("each criterion may be found at most once")
        return self

    def check_citations(self, evidence: EpisodeEvidence) -> None:
        """Refuse findings or decisive references that cite events the evidence does not hold.

        Args:
            evidence: The evidence this assessment claims to judge.

        Raises:
            ValueError: The evidence differs, or a cited event id is not in it.
        """
        if evidence.identity != self.evidence_sha256 or evidence.episode_id != self.episode_id:
            raise ValueError("assessment does not judge this evidence")
        cited = {ref for finding in self.findings for ref in finding.evidence_refs}
        invented = sorted((cited | set(self.decisive_refs)) - evidence.event_ids())
        if invented:
            raise ValueError(f"assessment cites events not in the evidence: {invented}")


class JudgedEpisode(ContractModel):
    """A downstream record's reference to its saved assessment, never a copied success flag.

    Attributes:
        episode_id: The judged episode.
        assessment: Saved assessment artifact.
        judge: Identity of the judge definition that made it.
    """

    episode_id: str = Field(min_length=1)
    assessment: ArtifactInput
    judge: str = Field(pattern=r"^[0-9a-f]{64}$")


def episode_reward(assessment: EpisodeAssessment) -> float | None:
    """Encode a saved verdict as the only reward on this path.

    Args:
        assessment: A saved episode assessment.

    Returns:
        1.0 for pass, 0.0 for fail, None for unassessable or any operational failure.
    """
    if assessment.status != "judged":
        return None
    return {"pass": 1.0, "fail": 0.0}.get(assessment.verdict)
