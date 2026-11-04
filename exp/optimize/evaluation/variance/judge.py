"""The one final LLM judgment owner of the decision-variance path.

:func:`judge_episode` judges one complete :class:`EpisodeEvidence` with one frozen
:class:`JudgeDefinition` and saves the result as an ``episode-assessment`` artifact whose inputs are
the saved evidence, the rubric and every judge response. Every downstream label reads that saved
assessment through :func:`load_assessment` and
:func:`~exp.optimize.evaluation.variance.episodes.episode_reward`; nothing else on this path decides
task success.

- **One owner, one draw.** Each judge request goes through the shared author-response owner, so a
  draw is bought once, replayed exactly and never redrawn. A malformed answer is saved as
  ``malformed`` and an unobtainable one as ``unavailable``; both are unscored and replay as such.
- **Exact replay.** The assessment is keyed by this protocol, the judge definition identity, the
  evidence identity and the frozen input budget. A saved assessment replays without a call.
  Changing the rubric or profile is a new definition, hence a new assessment.
- **Context-safe evidence.** The whole evidence is rendered when it fits the frozen input budget.
  When it does not, a judge with an evidence-tool budget receives an inventory of every event and
  reads contiguous ranges in full before its single final answer; it may pass only after reading
  every scored event, and it may cite only events it read. Without that budget the episode is
  recorded as insufficient evidence, unscored. Evidence is never truncated, chunk-graded or
  averaged.
- **Validation, not grading.** Code checks the answer's shape, that it finds every rubric criterion
  once, that its verdict follows the rubric's overall rule over its own findings, and that every
  citation exists in the evidence (:meth:`EpisodeAssessment.check_citations`). It never decides
  whether the task succeeded.

The rubric artifact holds exactly one JSON file whose digest is the definition's ``rubric_sha256``,
the ``episode-rubric`` shape the rubric author saves: an object whose ``body`` holds a ``criteria``
list, each entry with a ``criterion_id`` and a ``role`` (``required_correctness`` or
``efficiency_diagnostic``). The rubric is rendered to the judge as trusted configuration, except its
calibration cases: those are build-time controls, never examples for the judge to imitate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelMessage, ModelRequest, ModelResponse, ToolChoice
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.variance.episodes import (
    EVENT_CONTRACT_VERSION,
    CriterionFinding,
    EpisodeAssessment,
    EpisodeEvent,
    EpisodeEvidence,
    JudgeDefinition,
)
from exp.optimize.evaluation.variance.judge_prompt import (
    ASSESS_SCHEMA,
    ASSESS_TOOL,
    JUDGE_PROMPT,
    JUDGE_PROMPT_VERSION,
    READ_SCHEMA,
    EventRange,
    JudgeAnswer,
    judge_prompt_sha256,
    judge_schema_sha256,
)

JUDGE_PROTOCOL = "episode-judge-v1"
"""Version of the judge's rendering, reading and validation; bump it whenever they change."""

EVIDENCE_TYPE = "episode-evidence"
ASSESSMENT_TYPE = "episode-assessment"
_REQUIRED = "required_correctness"
_ROLES = frozenset({_REQUIRED, "efficiency_diagnostic"})


class _Rubric(ContractModel):
    """The loaded rubric: its trusted content and every criterion's role.

    Attributes:
        content: The rubric object exactly as saved.
        criteria: Role of each criterion, in rubric order.
    """

    content: JsonObject
    criteria: tuple[tuple[str, str], ...]

    def required(self) -> frozenset[str]:
        """Return the criteria that decide correctness."""
        return frozenset(c for c, role in self.criteria if role == _REQUIRED)


class _Inspection(ContractModel):
    """What the judge actually saw, saved beside its assessment.

    Attributes:
        protocol: Judge protocol that produced the assessment.
        mode: ``whole`` when the evidence was rendered in one request, ``tools`` when it was read
            through the evidence tool, ``insufficient`` when neither fit the frozen budget.
        reads: Evidence-tool calls made.
        inspected: Event ids whose full content the judge received, in sequence order.
        answer: The validated wire answer, when one was parsed.
    """

    protocol: str
    mode: Literal["whole", "tools", "insufficient"]
    reads: int
    inspected: tuple[str, ...]
    answer: JudgeAnswer | None = None


def save_evidence(
    project: ProjectStore, evidence: EpisodeEvidence, *, code_revision: str
) -> ArtifactInput:
    """Save, or verify the exact saved copy of, one episode's evidence.

    Args:
        project: Project that holds the evidence.
        evidence: The complete harness-owned episode record.
        code_revision: Producing source identity.

    Returns:
        The immutable evidence reference.

    Raises:
        ValueError: A different artifact already holds this evidence identity.
    """
    store = project.artifacts
    artifact_id = stable_id(EVIDENCE_TYPE, evidence)
    if not store.exists(artifact_id):
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=EVIDENCE_TYPE,
            envelope=_envelope((), code_revision),
            files={"evidence.json": evidence},
        )
    manifest = store.read(artifact_id).manifest
    saved = EpisodeEvidence.model_validate_json(store.read_bytes(artifact_id, "evidence.json"))
    if manifest.artifact_type != EVIDENCE_TYPE or saved != evidence:
        raise ValueError(f"artifact {artifact_id} does not hold this episode evidence")
    return artifact_input(manifest)


def load_assessment(project: ProjectStore, reference: ArtifactInput) -> EpisodeAssessment:
    """Load one saved assessment by its exact immutable reference.

    Args:
        project: Project that holds the assessment.
        reference: The assessment artifact reference.

    Returns:
        The saved assessment.

    Raises:
        ValueError: The reference does not name an unchanged episode assessment.
    """
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != ASSESSMENT_TYPE:
        raise ValueError(f"{reference.artifact_id} is not the referenced episode assessment")
    return EpisodeAssessment.model_validate_json(
        store.read_bytes(reference.artifact_id, "assessment.json")
    )


def assessment_identity(
    definition: JudgeDefinition, evidence: EpisodeEvidence, settings: AuthorSettings
) -> str:
    """Return the artifact id that caches one judgment.

    Args:
        definition: The frozen judge definition.
        evidence: The judged evidence.
        settings: Frozen request settings; only the input budget enters the key.

    Returns:
        The content-derived assessment id.
    """
    return stable_id(
        ASSESSMENT_TYPE,
        {
            "protocol": JUDGE_PROTOCOL,
            "judge": definition.identity,
            "evidence": evidence.identity,
            "maximum_input_bytes": settings.maximum_input_bytes,
        },
    )


def judge_episode(
    project: ProjectStore,
    definition: JudgeDefinition,
    evidence: EpisodeEvidence,
    *,
    settings: AuthorSettings,
    code_revision: str,
) -> tuple[ArtifactInput, EpisodeAssessment]:
    """Judge one episode once with the frozen definition, or replay its saved assessment.

    Args:
        project: Project that holds the rubric, evidence, draws and assessment.
        definition: Frozen judge profile, rubric, event contract, task and world.
        evidence: The complete episode record to judge.
        settings: Admitted request settings for the judge model; they must match the profile.
        code_revision: Producing source identity.

    Returns:
        The saved assessment reference and the assessment. A malformed or unavailable answer, or
        evidence that cannot be inspected within the frozen budget, is returned as a saved
        unscored assessment, never redrawn.

    Raises:
        ValueError: The definition, settings, rubric or evidence do not bind together, or the
            request could not be prepared before any draw.
    """
    rubric = _check_binding(project, definition, evidence, settings)
    store = project.artifacts
    identity = assessment_identity(definition, evidence, settings)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            reference = artifact_input(store.read(identity).manifest)
            assessment = load_assessment(project, reference)
            if assessment.judge != definition.identity or (
                assessment.evidence_sha256 != evidence.identity
            ):
                raise ValueError(f"saved assessment {identity} judges something else")
            return reference, assessment
        evidence_ref = save_evidence(project, evidence, code_revision=code_revision)
        run = _Judgment(project, definition, evidence, rubric, settings, code_revision, identity)
        assessment, inspection = run.judge(evidence_ref)
        inputs = (evidence_ref, definition.rubric, *run.responses)
        manifest = store.write_json(
            artifact_id=identity,
            artifact_type=ASSESSMENT_TYPE,
            envelope=_envelope(inputs, code_revision),
            files={"assessment.json": assessment, "inspection.json": inspection},
        )
        return artifact_input(manifest), assessment


class _Judgment:
    """One judgment in progress: its requests, saved responses and inspected events."""

    def __init__(
        self,
        project: ProjectStore,
        definition: JudgeDefinition,
        evidence: EpisodeEvidence,
        rubric: _Rubric,
        settings: AuthorSettings,
        code_revision: str,
        identity: str,
    ) -> None:
        """Bind the judgment's frozen inputs.

        Args:
            project: Project that holds the draws.
            definition: Frozen judge definition.
            evidence: The judged evidence.
            rubric: The loaded rubric.
            settings: Admitted request settings.
            code_revision: Producing source identity.
            identity: The assessment id; draw identities derive from it.
        """
        self.project = project
        self.definition = definition
        self.evidence = evidence
        self.rubric = rubric
        self.settings = settings
        self.code_revision = code_revision
        self.identity = identity
        self.responses: list[ArtifactInput] = []
        self.inspected: set[str] = set()
        self.reads = 0

    def judge(self, evidence_ref: ArtifactInput) -> tuple[EpisodeAssessment, _Inspection]:
        """Render, draw and validate; every outcome becomes one assessment.

        Args:
            evidence_ref: The saved evidence reference, a source of every draw.

        Returns:
            The assessment and the record of what the judge saw.
        """
        sources = (evidence_ref, self.definition.rubric)
        whole = self._request(
            self._payload(inventory=False), tools=(ASSESS_SCHEMA,), final=True, messages=()
        )
        if _size(whole) <= self.settings.maximum_input_bytes:
            self.inspected = set(self.evidence.event_ids())
            return self._loop(whole, sources, mode="whole")
        start = self._request(
            self._payload(inventory=True), tools=(READ_SCHEMA, ASSESS_SCHEMA), final=False
        )
        budget = self.definition.profile.evidence_tool_budget
        if budget == 0 or _size(start) > self.settings.maximum_input_bytes:
            reason = (
                f"insufficient evidence: the whole episode needs {_size(whole)} request bytes, "
                f"over the frozen {self.settings.maximum_input_bytes}, and "
                + (
                    "the judge profile has no evidence-tool budget"
                    if budget == 0
                    else "even its event inventory does not fit"
                )
                + "; it is not graded in parts"
            )
            return self._unscored("unavailable", reason, None), self._record("insufficient")
        return self._loop(start, sources, mode="tools")

    def _loop(
        self,
        request: ModelRequest,
        sources: tuple[ArtifactInput, ...],
        *,
        mode: Literal["whole", "tools"],
    ) -> tuple[EpisodeAssessment, _Inspection]:
        """Draw until the judge assesses, reading evidence ranges while its budget lasts.

        Args:
            request: The first request.
            sources: Saved artifacts every draw was rendered from.
            mode: Whether the evidence was rendered whole or is read through the tool.

        Returns:
            The assessment and the record of what the judge saw.
        """
        budget = self.definition.profile.evidence_tool_budget
        turn = 0
        while True:
            draw = f"{self.identity}-turn-{turn}"
            if _size(request) > self.settings.maximum_input_bytes:
                failure = "insufficient evidence: reading exhausted the frozen input budget"
                return self._unscored("unavailable", failure, None), self._record(mode)
            try:
                reference, response = author_response(
                    self.project,
                    self.settings,
                    request,
                    identity=draw,
                    sources=sources,
                    code_revision=self.code_revision,
                )
            except Exception as error:
                store = self.project.artifacts
                if not store.exists(draw + "-intent") or store.exists(draw + "-response"):
                    raise
                failure = f"no judge response, never redrawn: {type(error).__name__}: {error}"
                return self._unscored("unavailable", failure, None), self._record(mode)
            self.responses.append(reference)
            calls = response.output.tool_calls
            allowed = {tool.name for tool in request.tools}
            if (
                response.finish_reason.value != "completed"
                or len(calls) != 1
                or calls[0].name not in allowed
            ):
                failure = f"judge must complete exactly one call of {sorted(allowed)}"
                return self._unscored("malformed", failure, reference), self._record(mode)
            call = calls[0]
            if call.name == ASSESS_TOOL:
                return self._assess(call.arguments, reference, mode)
            self.reads += 1
            reply = self._read(call.arguments, call.call_id, request, response)
            messages = (
                *request.messages,
                ModelMessage(role="assistant", assistant_action=response.output),
                ModelMessage(role="tool", tool_call_id=call.call_id, content=reply),
            )
            final = self.reads >= budget
            request = self._request(
                None,
                tools=(ASSESS_SCHEMA,) if final else (READ_SCHEMA, ASSESS_SCHEMA),
                final=final,
                messages=messages,
            )
            turn += 1

    def _read(
        self, arguments: JsonObject, call_id: str, request: ModelRequest, response: ModelResponse
    ) -> str:
        """Answer one evidence-tool call with a full range, or say why it was refused.

        Args:
            arguments: The tool call's arguments.
            call_id: The tool call's id, which the result message carries.
            request: The request the call answered.
            response: The response that made the call.

        Returns:
            The tool result text; inspected events are recorded only when they are returned.
        """
        try:
            wanted = EventRange.model_validate(arguments)
        except ValidationError as error:
            return f"read refused: malformed range ({error.error_count()} errors)"
        by_id = {event.event_id: event for event in self.evidence.events}
        first, last = by_id.get(wanted.first_event_id), by_id.get(wanted.last_event_id)
        if first is None or last is None or first.sequence > last.sequence:
            return "read refused: name two existing event ids, the first not after the last"
        events = self.evidence.events[first.sequence : last.sequence + 1]
        reply = json.dumps(
            {"events": [event.model_dump(mode="json") for event in events]}, sort_keys=True
        )
        remaining = self.definition.profile.evidence_tool_budget - self.reads
        trial = self._request(
            None,
            tools=(ASSESS_SCHEMA,) if remaining <= 0 else (READ_SCHEMA, ASSESS_SCHEMA),
            final=remaining <= 0,
            messages=(
                *request.messages,
                ModelMessage(role="assistant", assistant_action=response.output),
                ModelMessage(role="tool", tool_call_id=call_id, content=reply),
            ),
        )
        if _size(trial) > self.settings.maximum_input_bytes:
            return (
                f"read refused: {len(events)} events need {len(reply.encode())} bytes, more than "
                "the remaining input budget; read a smaller range"
            )
        self.inspected.update(event.event_id for event in events)
        return reply

    def _assess(
        self,
        arguments: JsonObject,
        reference: ArtifactInput,
        mode: Literal["whole", "tools"],
    ) -> tuple[EpisodeAssessment, _Inspection]:
        """Validate the final answer into a judged assessment, or a saved malformed one.

        Args:
            arguments: The assess_episode arguments.
            reference: The saved response that made the call.
            mode: How the evidence was presented.

        Returns:
            The assessment and the record of what the judge saw.
        """
        try:
            answer = JudgeAnswer.model_validate(arguments)
            assessment = self._validated(answer, reference)
        except (ValidationError, ValueError) as error:
            failure = f"invalid judge answer, never redrawn: {error}"
            return self._unscored("malformed", failure, reference), self._record(mode)
        return assessment, self._record(mode, answer)

    def _validated(self, answer: JudgeAnswer, reference: ArtifactInput) -> EpisodeAssessment:
        """Check one answer's shape, rubric coverage, verdict rule and citations.

        Args:
            answer: The parsed wire answer.
            reference: The saved response that holds it.

        Returns:
            The judged assessment.

        Raises:
            ValueError: The answer misses or invents criteria, contradicts the overall rule over
                its own findings, lacks citations, or cites events it does not hold or never read.
        """
        found = {row.criterion_id: row for row in answer.criterion_findings}
        expected = [criterion for criterion, _ in self.rubric.criteria]
        if len(found) != len(answer.criterion_findings) or set(found) != set(expected):
            raise ValueError(
                f"findings must cover each rubric criterion {expected} exactly once, "
                f"not {[row.criterion_id for row in answer.criterion_findings]}"
            )
        uncited = sorted(
            row.criterion_id
            for row in answer.criterion_findings
            if row.status in ("met", "not_met") and not row.event_refs
        )
        if uncited:
            raise ValueError(f"met and not_met findings must cite events: {uncited}")
        required = [found[criterion].status for criterion in self.rubric.required()]
        unestablished = {"not_met", "insufficient_evidence"} & set(required)
        if answer.verdict == "pass" and unestablished:
            raise ValueError("pass contradicts a required finding that is not established as met")
        if answer.verdict == "fail" and "not_met" not in required:
            raise ValueError("fail needs a required criterion found not_met")
        if answer.verdict != "unassessable" and not answer.decisive_refs:
            raise ValueError("a pass or fail verdict must cite its decisive events")
        assessment = EpisodeAssessment(
            episode_id=self.evidence.episode_id,
            evidence_sha256=self.evidence.identity,
            judge=self.definition.identity,
            status="judged",
            verdict=answer.verdict,
            findings=tuple(
                CriterionFinding(
                    criterion_id=row.criterion_id,
                    status=row.status,
                    evidence_refs=row.event_refs,
                    rationale=row.reason,
                )
                for row in answer.criterion_findings
            ),
            decisive_refs=answer.decisive_refs,
            limitations=(
                *self._limitations(),
                *answer.missing_evidence,
                *(f"rubric problem: {problem}" for problem in answer.rubric_problems),
            ),
            response=reference,
        )
        assessment.check_citations(self.evidence)
        cited = {ref for row in assessment.findings for ref in row.evidence_refs}
        unread = sorted((cited | set(assessment.decisive_refs)) - self.inspected)
        if unread:
            raise ValueError(f"assessment cites events the judge never read: {unread}")
        unread_scored = [
            event.event_id
            for event in self.evidence.events
            if event.scope == "scored" and event.event_id not in self.inspected
        ]
        if answer.verdict == "pass" and unread_scored:
            raise ValueError(f"pass without reading scored events {unread_scored}")
        return assessment

    def _limitations(self) -> tuple[str, ...]:
        """Return evidence limits the host knows: harness gaps and events the judge never read."""
        unread = [e.event_id for e in self.evidence.events if e.event_id not in self.inspected]
        return (
            *(f"harness could not record: {missing}" for missing in self.evidence.missing),
            *((f"events not inspected by the judge: {unread}",) if unread else ()),
        )

    def _unscored(
        self,
        status: Literal["malformed", "unavailable"],
        failure: str,
        reference: ArtifactInput | None,
    ) -> EpisodeAssessment:
        """Build the operationally unscored assessment of a failed judgment."""
        return EpisodeAssessment(
            episode_id=self.evidence.episode_id,
            evidence_sha256=self.evidence.identity,
            judge=self.definition.identity,
            status=status,
            verdict="unassessable",
            limitations=self._limitations(),
            response=reference,
            failure=failure[:2000],
        )

    def _record(
        self, mode: Literal["whole", "tools", "insufficient"], answer: JudgeAnswer | None = None
    ) -> _Inspection:
        """Record what the judge saw."""
        return _Inspection(
            protocol=JUDGE_PROTOCOL,
            mode=mode,
            reads=self.reads,
            inspected=tuple(
                e.event_id for e in self.evidence.events if e.event_id in self.inspected
            ),
            answer=answer,
        )

    def _payload(self, *, inventory: bool) -> JsonObject:
        """Render the trusted configuration, scope and evidence (or its inventory) as data.

        Args:
            inventory: Render every event without content, for reading through the tool.

        Returns:
            The user message object.
        """
        evidence = self.evidence
        definition = self.definition
        events: list[JsonValue] = [
            _inventory_row(event) if inventory else event.model_dump(mode="json")
            for event in evidence.events
        ]
        record: JsonObject = {"complete": evidence.complete, "missing": list(evidence.missing)}
        payload: JsonObject = {
            "judge_binding": {
                "judge_definition": definition.identity,
                "judge_profile": definition.profile.identity,
                "rubric": definition.rubric.model_dump(mode="json"),
                "rubric_sha256": definition.rubric_sha256,
                "event_contract": definition.event_contract,
                "task_sha256": definition.task_sha256,
                "world_sha256": definition.world_sha256,
                "episode_id": evidence.episode_id,
                "evidence_sha256": evidence.identity,
            },
            "rubric": self.rubric.content,
            "rubric_criteria": [
                {"criterion_id": criterion, "role": role}
                for criterion, role in self.rubric.criteria
            ],
            "assessment_scope": _scope(evidence),
        }
        if inventory:
            payload["evidence_inventory"] = {**record, "events": events}
            payload["reads_allowed"] = definition.profile.evidence_tool_budget
        else:
            payload["episode_evidence"] = {**record, "events": events}
        return payload

    def _request(
        self,
        payload: JsonObject | None,
        *,
        tools: tuple[ToolSchema, ...],
        final: bool,
        messages: tuple[ModelMessage, ...] = (),
    ) -> ModelRequest:
        """Build one judge request from the frozen profile.

        Args:
            payload: The first user message, or None to continue ``messages``.
            tools: Tools offered on this turn.
            final: Force the assessment tool instead of letting the judge choose.
            messages: The conversation so far, when continuing.

        Returns:
            The request.
        """
        if payload is not None:
            messages = (
                ModelMessage(role="system", content=JUDGE_PROMPT),
                ModelMessage(role="user", content=json.dumps(payload, sort_keys=True)),
            )
        profile = self.definition.profile
        return ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice=ToolChoice(name=ASSESS_TOOL) if final else "required",
            temperature=profile.temperature,
            maximum_output_tokens=profile.maximum_output_tokens,
        )


def _scope(evidence: EpisodeEvidence) -> JsonObject:
    """Render the assessment scope: context, scored, imposed and actor-owned event ids."""
    scored = [e for e in evidence.events if e.scope == "scored"]
    return {
        "context_event_ids": [e.event_id for e in evidence.events if e.scope == "context"],
        "scored_event_ids": [e.event_id for e in scored],
        "scored_start": scored[0].event_id if scored else None,
        "experiment_imposed_event_ids": [e.event_id for e in scored if e.owner == "experiment"],
        "actor_scored_event_ids": [e.event_id for e in scored if e.owner == "actor"],
    }


def _inventory_row(event: EpisodeEvent) -> JsonValue:
    """Render one event without its content, stating the content's exact size."""
    row = event.model_dump(mode="json", exclude={"content"})
    row["content_bytes"] = len(json.dumps(event.content, sort_keys=True).encode())
    return row


def _size(request: ModelRequest) -> int:
    """Measure a request exactly as the author-response owner bounds it."""
    return len(request.model_dump_json().encode())


def _envelope(inputs: tuple[ArtifactInput, ...], code_revision: str) -> ArtifactEnvelope:
    """Build provenance over sorted unique inputs."""
    return ArtifactEnvelope(
        schema_version=1,
        created_at=datetime.now(UTC),
        code_revision=code_revision,
        inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
    )


def _check_binding(
    project: ProjectStore,
    definition: JudgeDefinition,
    evidence: EpisodeEvidence,
    settings: AuthorSettings,
) -> _Rubric:
    """Refuse a definition, settings or evidence that do not belong together, before any draw.

    Args:
        project: Project that holds the rubric.
        definition: Frozen judge definition.
        evidence: The evidence to judge.
        settings: Admitted request settings.

    Returns:
        The loaded rubric.

    Raises:
        ValueError: Any binding differs.
    """
    profile = definition.profile
    if (
        profile.prompt_version != JUDGE_PROMPT_VERSION
        or profile.prompt_sha256 != judge_prompt_sha256()
        or profile.output_schema_sha256 != judge_schema_sha256()
    ):
        raise ValueError("judge profile is not bound to this judge prompt and schema")
    if definition.event_contract != EVENT_CONTRACT_VERSION:
        raise ValueError(f"judge definition binds event contract {definition.event_contract}")
    if (evidence.task_sha256, evidence.world_sha256) != (
        definition.task_sha256,
        definition.world_sha256,
    ):
        raise ValueError("evidence comes from a different task or world than the judge's")
    if (
        settings.model_alias != profile.model_alias
        or settings.model != profile.model
        or settings.output_tokens != profile.maximum_output_tokens
        or settings.reasoning_effort != profile.reasoning_effort
    ):
        raise ValueError("judge request settings differ from the frozen judge profile")
    return _load_rubric(project, definition)


def _load_rubric(project: ProjectStore, definition: JudgeDefinition) -> _Rubric:
    """Load the definition's rubric and its criterion roles.

    Args:
        project: Project that holds the rubric.
        definition: Frozen judge definition.

    Returns:
        The rubric.

    Raises:
        ValueError: The rubric changed, is not one JSON object, or lacks well-formed criteria.
    """
    store = project.artifacts
    manifest = store.read(definition.rubric.artifact_id).manifest
    if artifact_input(manifest) != definition.rubric or len(manifest.files) != 1:
        raise ValueError("judge rubric must be the referenced single-file artifact")
    file = manifest.files[0]
    if file.sha256 != definition.rubric_sha256:
        raise ValueError("judge rubric content differs from the definition's rubric digest")
    content = json.loads(store.read_bytes(manifest.artifact_id, file.path))
    body = content.get("body") if isinstance(content, dict) else None
    rows = body.get("criteria") if isinstance(body, dict) else None
    if not isinstance(content, dict) or not isinstance(body, dict) or not isinstance(rows, list):
        raise ValueError("judge rubric must be an object whose body holds a criteria list")
    criteria: list[tuple[str, str]] = []
    for row in rows:
        criterion = row.get("criterion_id") if isinstance(row, dict) else None
        role = row.get("role") if isinstance(row, dict) else None
        if not isinstance(criterion, str) or not criterion or role not in _ROLES:
            raise ValueError("each rubric criterion needs a criterion_id and a known role")
        criteria.append((criterion, str(role)))
    if not criteria or len({c for c, _ in criteria}) != len(criteria):
        raise ValueError("rubric criterion ids must be present and unique")
    shown = {key: value for key, value in body.items() if key != "calibration_cases"}
    content = {**content, "body": shown}
    return _Rubric(content=content, criteria=tuple(criteria))
