"""The agentic dense judge: candidate behaviors decided in context by several prompt variants.

:func:`judge_dense` judges one complete :class:`EpisodeEvidence` for one frozen
:class:`~exp.optimize.evaluation.variance.dense_reward.DenseRewardDefinition` and saves a
``dense-assessment`` artifact holding the versioned export (``dense.json``) and every phase of
every variant (``inspection.json``). It never touches the outcome reward: the frozen episode
judge's verdict stays the only outcome reward and is attached to the export only as a reference.

- **Agentic, several variants.** Every prompt variant of the profile judges the episode in its own
  conversation: a plan of the evidence each candidate needs, range reads when the evidence does
  not fit whole, its findings, and at most one repair after host validation. Draws go through the
  shared author-response owner, so each is bought once and replayed exactly.
- **The judge decides.** Candidates (from tag definitions, error tags and linters) are hypotheses.
  Code validates shape, coverage and citations, and that an established finding on an
  ``obligation_required`` tag cites the obligation, the actor's change, the result and why the
  change weakened it. A finding that still fails validation after the repair becomes uncertain.
- **Strict majority.** A candidate's decision is the decision a strict majority of variants gave;
  otherwise it is uncertain and unscored. A variant without a valid answer votes uncertain, so a
  missing answer can never establish anything.
- **Localized.** Each finding is mapped to the actor's action and token spans
  (:func:`~exp.optimize.evaluation.variance.dense_spans.actor_spans`) and given its bounded reward
  (:func:`~exp.optimize.evaluation.variance.dense_reward.aggregate`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelMessage, ModelRequest, ToolChoice
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.variance.dense_candidates import DenseCandidate
from exp.optimize.evaluation.variance.dense_prompt import (
    DENSE_PHASE_PROMPTS,
    DENSE_PROMPT_VARIANTS,
    PLAN_SCHEMA,
    REPORT_SCHEMA,
    AnswerCitation,
    DenseAnswer,
    DenseAnswerFinding,
    DensePhase,
    dense_prompt_sha256,
    dense_schema_sha256,
)
from exp.optimize.evaluation.variance.dense_reward import (
    Citation,
    Decision,
    DenseFinding,
    DenseRewardDefinition,
    DenseRewardExport,
    EpisodeDense,
    OutcomeReference,
    VariantRecord,
    aggregate,
)
from exp.optimize.evaluation.variance.dense_spans import (
    ActorTrajectory,
    SampledTokens,
    actor_event,
    actor_spans,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence
from exp.optimize.evaluation.variance.judge import save_evidence
from exp.optimize.evaluation.variance.judge_prompt import READ_SCHEMA, READ_TOOL, EventRange

DENSE_JUDGE_PROTOCOL = "dense-judge-v1"
"""Version of the dense judge's rendering, validation and reconciliation."""

DENSE_ASSESSMENT_TYPE = "dense-assessment"
"""Artifact type of a saved dense judgment."""

_DECISIONS: tuple[Decision, ...] = ("established", "acceptable", "absent", "uncertain")


class DenseRewardJudge(ContractModel):
    """One frozen dense definition and the admitted request settings that apply it.

    Attributes:
        definition: The frozen dense reward definition.
        settings: Admitted request settings; they must match the definition's profile.
    """

    definition: DenseRewardDefinition
    settings: AuthorSettings

    @property
    def identity(self) -> str:
        """The dense definition identity every export carries."""
        return self.definition.identity


class DenseTurn(ContractModel):
    """One persisted draw of one variant.

    Attributes:
        variant: Prompt variant.
        turn: Draw index within the variant.
        phase: Phase of the draw.
        tool: Tool the judge called.
        arguments: Its exact arguments.
        response: The saved draw.
        problems: Host validation problems returned to the judge after a findings call.
    """

    variant: str
    turn: int
    phase: DensePhase
    tool: str
    arguments: JsonObject
    response: ArtifactInput
    problems: tuple[str, ...] = ()


class DenseInspection(ContractModel):
    """What every variant saw and did, saved beside the export.

    Attributes:
        protocol: Dense judge protocol.
        candidates: The candidates judged.
        turns: Every draw of every variant, in order.
        inspected: Event ids each variant received in full.
    """

    protocol: str = DENSE_JUDGE_PROTOCOL
    candidates: tuple[DenseCandidate, ...]
    turns: tuple[DenseTurn, ...]
    inspected: dict[str, tuple[str, ...]]


HYBRID_ASSESSMENT_TYPE = "dense-hybrid-assessment"
"""Artifact type of a saved hybrid judgment, whose ``hybrid.json`` holds the merged export."""


def load_dense(project: ProjectStore, reference: ArtifactInput) -> DenseRewardExport:
    """Load one saved dense export by its exact reference, full or hybrid judgment.

    Raises:
        ValueError: The reference does not name an unchanged dense assessment.
    """
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference:
        raise ValueError(f"{reference.artifact_id} is not the referenced dense assessment")
    if manifest.artifact_type == HYBRID_ASSESSMENT_TYPE:
        hybrid = json.loads(project.artifacts.read_bytes(reference.artifact_id, "hybrid.json"))
        return DenseRewardExport.model_validate(hybrid["export"])
    if manifest.artifact_type != DENSE_ASSESSMENT_TYPE:
        raise ValueError(f"{reference.artifact_id} is not the referenced dense assessment")
    return DenseRewardExport.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "dense.json")
    )


def judge_dense(
    project: ProjectStore,
    judge: DenseRewardJudge,
    evidence: EpisodeEvidence,
    candidates: Sequence[DenseCandidate],
    trajectory: ActorTrajectory,
    *,
    code_revision: str,
    outcome: OutcomeReference | None = None,
    sampled: SampledTokens | None = None,
) -> tuple[ArtifactInput, DenseRewardExport]:
    """Judge one episode's candidate behaviors once, or replay the saved judgment.

    Args:
        project: Project holding the draws and the saved assessment.
        judge: Frozen dense definition and admitted settings.
        evidence: The complete episode evidence.
        candidates: Behaviors to check; every candidate's tag must be a frozen tag.
        trajectory: The actor trajectory spans index.
        code_revision: Producing source identity.
        outcome: The separate outcome reward, attached to the returned export only.
        sampled: The actor's sampled tokens, bound from the provider response or trainer
            rollout; without them every span is action-level.

    Returns:
        The saved assessment reference and the export with ``outcome`` attached.

    Raises:
        ValueError: The definition, settings, evidence or candidates do not bind together.
    """
    definition = judge.definition
    _check_binding(judge, evidence, candidates)
    identity = stable_id(
        DENSE_ASSESSMENT_TYPE,
        {
            "protocol": DENSE_JUDGE_PROTOCOL,
            "definition": definition.identity,
            "evidence": evidence.identity,
            "candidates": sha256_json([row.model_dump(mode="json") for row in candidates]),
            "trajectory": sha256_json(trajectory),
            "sampled": None if sampled is None else sampled_digest(sampled),
        },
    )
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            reference = artifact_input(store.read(identity).manifest)
            saved = load_dense(project, reference)
            if saved.definition_sha256 != definition.identity or (
                saved.evidence_sha256 != evidence.identity
            ):
                raise ValueError(f"saved dense assessment {identity} judges something else")
            return reference, saved.model_copy(update={"outcome": outcome})
        evidence_ref = save_evidence(project, evidence, code_revision=code_revision)
        run = _DenseRun(project, judge, evidence, tuple(candidates), code_revision, identity)
        answers = [
            run.variant(index, version, evidence_ref)
            for index, version in enumerate(definition.profile.prompt_variants)
        ]
        findings = _reconcile(evidence, definition, run.candidates, answers)
        located = tuple(
            row.model_copy(
                update={
                    "spans": actor_spans(
                        evidence,
                        trajectory,
                        (
                            *row.observed_refs,
                            *(() if row.change is None else row.change.event_refs),
                        ),
                        quote=row.quote,
                        sampled=sampled,
                    )
                }
            )
            for row in findings
        )
        located = tuple(
            row.model_copy(
                update={
                    "unscored_reason": "no actor-owned action: the cited events are not the "
                    "actor's own scored actions or replies to them"
                }
            )
            if row.decision == "established" and not row.spans
            else row
            for row in located
        )
        rewarded, episode = aggregate(located, definition)
        answered = any(record.status == "answered" for record in run.records)
        export = DenseRewardExport(
            episode_id=evidence.episode_id,
            evidence_sha256=evidence.identity,
            definition_sha256=definition.identity,
            status="judged" if answered else "unavailable",
            findings=rewarded if answered else (),
            episode=episode if answered else _empty(),
            trajectory=trajectory.kind,
            sampled_tokens=None if sampled is None else sampled.binding,
            granularity=_granularity(rewarded),
            variants=tuple(run.records),
            failure=None if answered else "no prompt variant returned a valid answer",
        )
        inspection = DenseInspection(
            candidates=run.candidates,
            turns=tuple(run.turns),
            inspected={key: tuple(sorted(value)) for key, value in run.inspected.items()},
        )
        manifest = store.write_json(
            artifact_id=identity,
            artifact_type=DENSE_ASSESSMENT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted({evidence_ref, *run.responses}, key=lambda ref: ref.artifact_id)
                ),
            ),
            files={"dense.json": export, "inspection.json": inspection},
        )
        return artifact_input(manifest), export.model_copy(update={"outcome": outcome})


def sampled_digest(sampled: SampledTokens) -> str:
    """Digest of a sampled-token binding, part of the saved judgment's identity."""
    return sha256_json(sampled)


def _granularity(findings: Sequence[DenseFinding]) -> Literal["sampled_tokens", "action", "mixed"]:
    """Say whether rewarded spans are token-local, action-level or both."""
    kinds = {span.granularity for row in findings if row.reward for span in row.spans}
    if kinds == {"sampled_tokens"}:
        return "sampled_tokens"
    return "mixed" if len(kinds) > 1 else "action"


def _empty() -> EpisodeDense:
    """The episode total of an unavailable judgment: nothing scored."""
    return EpisodeDense(
        total=0.0, credit=0.0, penalty=0.0, decisive=False, scored=0, unscored=0, diagnostics=0
    )


def _check_binding(
    judge: DenseRewardJudge, evidence: EpisodeEvidence, candidates: Sequence[DenseCandidate]
) -> None:
    """Refuse a definition, settings, evidence or candidates that do not belong together."""
    definition, settings = judge.definition, judge.settings
    profile = definition.profile
    if profile.prompt_sha256 != dense_prompt_sha256(profile.prompt_variants) or (
        profile.output_schema_sha256 != dense_schema_sha256()
    ):
        raise ValueError("dense judge profile is not bound to the current prompts and schemas")
    if (evidence.task_sha256, evidence.world_sha256) != (
        definition.task_sha256,
        definition.world_sha256,
    ):
        raise ValueError("evidence comes from a different task or world than the dense definition")
    if (
        settings.model_alias != profile.model_alias
        or settings.model != profile.model
        or settings.output_tokens != profile.maximum_output_tokens
        or settings.reasoning_effort != profile.reasoning_effort
        or settings.maximum_input_bytes != profile.maximum_input_bytes
    ):
        raise ValueError("dense judge request settings differ from the frozen profile")
    identities = [row.candidate_id for row in candidates]
    if not candidates or len(set(identities)) != len(identities):
        raise ValueError("dense candidates must be present with unique identities")
    known = {tag.tag_id for tag in definition.tags}
    foreign = sorted({row.tag_id for row in candidates} - known)
    if foreign:
        raise ValueError(f"candidates propose tags the definition does not freeze: {foreign}")


class _VariantAnswer(ContractModel):
    """One variant's validated findings by candidate, or nothing.

    Attributes:
        version: Prompt variant.
        findings: Valid findings by candidate id; an invalid one is absent and votes uncertain.
    """

    version: str
    findings: dict[str, DenseAnswerFinding]


class _DenseRun:
    """One dense judgment in progress: its variants, draws and inspected events."""

    def __init__(
        self,
        project: ProjectStore,
        judge: DenseRewardJudge,
        evidence: EpisodeEvidence,
        candidates: tuple[DenseCandidate, ...],
        code_revision: str,
        identity: str,
    ) -> None:
        """Bind the frozen inputs."""
        self.project, self.judge, self.evidence = project, judge, evidence
        self.candidates, self.code_revision, self.identity = candidates, code_revision, identity
        self.responses: list[ArtifactInput] = []
        self.records: list[VariantRecord] = []
        self.turns: list[DenseTurn] = []
        self.inspected: dict[str, set[str]] = {}

    def variant(self, index: int, version: str, evidence_ref: ArtifactInput) -> _VariantAnswer:
        """Run one variant's plan, reads, findings and repair.

        Args:
            index: Variant position, part of each draw identity.
            version: Prompt variant.
            evidence_ref: Saved evidence, a source of every draw.

        Returns:
            The variant's valid findings; empty when it gave no valid answer.
        """
        profile = self.judge.definition.profile
        budget = profile.evidence_tool_budget
        limit = self.judge.settings.maximum_input_bytes
        inspected = self.inspected.setdefault(version, set())
        payload = self._payload(inventory=False)
        reading = _size(self._request(version, payload, (PLAN_SCHEMA,), PLAN_SCHEMA.name)) > limit
        if reading:
            payload = self._payload(inventory=True)
            if budget == 0 or _size(self._request(version, payload, (PLAN_SCHEMA,), "")) > limit:
                failure = (
                    "the evidence exceeds the frozen input budget and cannot be read in ranges"
                )
                return self._close(version, "unavailable", [], failure)
        else:
            inspected.update(self.evidence.event_ids())
        payload["phase_instruction"] = DENSE_PHASE_PROMPTS["plan"]
        messages = self._request(version, payload, (PLAN_SCHEMA,), "").messages
        phase: DensePhase = "plan"
        reads = 0
        drawn: list[ArtifactInput] = []
        valid: dict[str, DenseAnswerFinding] = {}
        for turn in range(budget + 3):
            drawn_phase = phase
            free = reading and phase != "plan" and reads < budget
            own = PLAN_SCHEMA if phase == "plan" else REPORT_SCHEMA
            tools = (READ_SCHEMA, own) if free else (own,)
            request = self._request(version, None, tools, "" if free else own.name, messages)
            if _size(request) > limit:
                return self._close(version, "unavailable", drawn, "reading exhausted the budget")
            try:
                reference, response = author_response(
                    self.project,
                    self.judge.settings,
                    request,
                    identity=f"{self.identity}-{index}-turn-{turn}",
                    sources=(evidence_ref,),
                    code_revision=self.code_revision,
                )
            except Exception as error:
                store, draw = self.project.artifacts, f"{self.identity}-{index}-turn-{turn}"
                if not store.exists(draw + "-intent") or store.exists(draw + "-response"):
                    raise
                failure = f"no response, never redrawn: {type(error).__name__}: {error}"
                return self._close(version, "unavailable", drawn, failure)
            drawn.append(reference)
            calls = response.output.tool_calls
            if len(calls) != 1 or calls[0].name not in {tool.name for tool in tools}:
                return self._close(version, "malformed", drawn, "expected exactly one tool call")
            call = calls[0]
            problems: tuple[str, ...] = ()
            if call.name == READ_TOOL:
                reads += 1
                reply = self._read(call.arguments, inspected)
            elif phase == "plan":
                phase = "investigate"
                reply = json.dumps(
                    {
                        "status": "recorded",
                        "phase_instruction": DENSE_PHASE_PROMPTS[phase],
                        "reads_left": budget - reads,
                    }
                )
            else:
                try:
                    answer = DenseAnswer.model_validate(call.arguments)
                except ValidationError as error:
                    answer, problems = (
                        None,
                        (f"findings do not match the schema ({error.error_count()} errors)",),
                    )
                if answer is not None:
                    valid, problems = self._validate(answer, inspected)
                if not problems or phase == "repair":
                    self._turn(
                        version, turn, drawn_phase, call.name, call.arguments, reference, problems
                    )
                    if answer is None:
                        return self._close(version, "malformed", drawn, problems[0])
                    return self._close(version, "answered", drawn, None, valid)
                phase = "repair"
                reply = json.dumps(
                    {
                        "status": "problems",
                        "problems": list(problems),
                        "phase_instruction": DENSE_PHASE_PROMPTS[phase],
                        "reads_left": budget - reads,
                    }
                )
            self._turn(version, turn, drawn_phase, call.name, call.arguments, reference, problems)
            messages = (
                *request.messages,
                ModelMessage(role="assistant", assistant_action=response.output),
                ModelMessage(role="tool", tool_call_id=call.call_id, content=reply),
            )
        return self._close(version, "malformed", drawn, "no findings within the draw ceiling")

    def _close(
        self,
        version: str,
        status: Literal["answered", "malformed", "unavailable"],
        drawn: list[ArtifactInput],
        failure: str | None,
        findings: dict[str, DenseAnswerFinding] | None = None,
    ) -> _VariantAnswer:
        """Record a variant's outcome and return its valid findings."""
        self.responses.extend(drawn)
        self.records.append(
            VariantRecord(
                prompt_version=version,
                status=status,
                responses=tuple(drawn),
                failure=None if failure is None else failure[:2000],
            )
        )
        return _VariantAnswer(version=version, findings=findings or {})

    def _turn(
        self,
        version: str,
        turn: int,
        phase: DensePhase,
        tool: str,
        arguments: JsonObject,
        reference: ArtifactInput,
        problems: tuple[str, ...],
    ) -> None:
        """Persist one draw in the inspection record."""
        self.turns.append(
            DenseTurn(
                variant=version,
                turn=turn,
                phase=phase,
                tool=tool,
                arguments=arguments,
                response=reference,
                problems=problems,
            )
        )

    def _validate(
        self, answer: DenseAnswer, inspected: set[str]
    ) -> tuple[dict[str, DenseAnswerFinding], tuple[str, ...]]:
        """Check coverage and citations; never decide a behavior.

        Returns:
            The valid findings by candidate, and every problem found.
        """
        definition = self.judge.definition
        by_id = {row.candidate_id: row for row in self.candidates}
        events = {event.event_id: event for event in self.evidence.events}
        problems: list[str] = []
        valid: dict[str, DenseAnswerFinding] = {}
        answered = [row.candidate_id for row in answer.findings]
        missing = sorted(set(by_id) - set(answered))
        if missing:
            problems.append(f"no finding for candidates {missing}")
        for row in answer.findings:
            candidate = by_id.get(row.candidate_id)
            if candidate is None or answered.count(row.candidate_id) > 1:
                problems.append(f"{row.candidate_id}: unknown or repeated candidate")
                continue
            if row.tag_id != candidate.tag_id:
                problems.append(f"{row.candidate_id}: tag must be {candidate.tag_id}")
                continue
            cited = [
                *row.observed_refs,
                *row.result_refs,
                *(() if row.obligation is None else row.obligation.event_refs),
                *(() if row.change is None else row.change.event_refs),
            ]
            unknown = sorted({ref for ref in cited if ref not in events})
            unread = sorted({ref for ref in cited if ref in events} - inspected)
            own = [
                ref
                for ref in row.observed_refs
                if ref in events and actor_event(self.evidence, events[ref]) is not None
            ]
            fault = None
            if unknown or unread:
                fault = f"cites unknown {unknown} or unread {unread} events"
            elif row.decision == "established":
                tag = definition.tag(row.tag_id)
                changes = [] if row.change is None else row.change.event_refs
                actor_change = [
                    ref for ref in changes if actor_event(self.evidence, events[ref]) is not None
                ]
                if not own:
                    fault = (
                        "established needs observed_refs on the actor's own scored actions or "
                        "the replies to them; experiment-imposed actions, their effects and "
                        "events of other owners are not the actor's behavior"
                    )
                elif (
                    tag.obligation_required
                    and tag.polarity == "penalty"
                    and (
                        row.obligation is None
                        or not row.obligation.event_refs
                        or not actor_change
                        or not row.result_refs
                        or not row.weakening
                    )
                ):
                    fault = (
                        "established needs the cited obligation, the actor's cited change, the "
                        "result_refs and why the change weakened the obligation"
                    )
            if fault:
                problems.append(f"{row.candidate_id}: {fault}")
            else:
                valid[row.candidate_id] = row
        return valid, tuple(problems)

    def _read(self, arguments: JsonObject, inspected: set[str]) -> str:
        """Answer one read with a full range, or say why it was refused."""
        try:
            wanted = EventRange.model_validate(arguments)
        except ValidationError as error:
            return f"read refused: malformed range ({error.error_count()} errors)"
        by_id = {event.event_id: event for event in self.evidence.events}
        first, last = by_id.get(wanted.first_event_id), by_id.get(wanted.last_event_id)
        if first is None or last is None or first.sequence > last.sequence:
            return "read refused: name two existing event ids, the first not after the last"
        events = self.evidence.events[first.sequence : last.sequence + 1]
        inspected.update(event.event_id for event in events)
        return json.dumps(
            {"events": [event.model_dump(mode="json") for event in events]}, sort_keys=True
        )

    def _payload(self, *, inventory: bool) -> JsonObject:
        """Render the binding, tags, candidates, scope and evidence (or its inventory)."""
        definition, evidence = self.judge.definition, self.evidence
        events: list[JsonValue] = []
        for event in evidence.events:
            row = event.model_dump(mode="json", exclude={"content"} if inventory else set())
            if inventory:
                row["content_bytes"] = len(json.dumps(event.content, sort_keys=True).encode())
            events.append(row)
        scored = [event for event in evidence.events if event.scope == "scored"]
        record: JsonObject = {
            "complete": evidence.complete,
            "missing": list(evidence.missing),
            "events": events,
        }
        payload: JsonObject = {
            "binding": {
                "definition": definition.identity,
                "task_sha256": definition.task_sha256,
                "world_sha256": definition.world_sha256,
                "episode_id": evidence.episode_id,
                "evidence_sha256": evidence.identity,
            },
            "tags": [tag.model_dump(mode="json") for tag in definition.tags],
            "candidates": [row.model_dump(mode="json") for row in self.candidates],
            "assessment_scope": {
                "context_event_ids": [e.event_id for e in evidence.events if e.scope == "context"],
                "scored_event_ids": [e.event_id for e in scored],
                "experiment_imposed_event_ids": [
                    e.event_id for e in scored if e.owner == "experiment"
                ],
                "actor_scored_event_ids": [e.event_id for e in scored if e.owner == "actor"],
            },
        }
        if inventory:
            payload["evidence_inventory"] = record
            payload["reads_allowed"] = definition.profile.evidence_tool_budget
        else:
            payload["episode_evidence"] = record
        return payload

    def _request(
        self,
        version: str,
        payload: JsonObject | None,
        tools: tuple[ToolSchema, ...],
        forced: str,
        messages: tuple[ModelMessage, ...] = (),
    ) -> ModelRequest:
        """Build one request of one variant from the frozen profile."""
        if payload is not None:
            messages = (
                ModelMessage(role="system", content=DENSE_PROMPT_VARIANTS[version]),
                ModelMessage(role="user", content=json.dumps(payload, sort_keys=True)),
            )
        profile = self.judge.definition.profile
        return ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice=ToolChoice(name=forced) if forced else "required",
            temperature=profile.temperature,
            maximum_output_tokens=profile.maximum_output_tokens,
        )


def _reconcile(
    evidence: EpisodeEvidence,
    definition: DenseRewardDefinition,
    candidates: tuple[DenseCandidate, ...],
    answers: Sequence[_VariantAnswer],
) -> tuple[DenseFinding, ...]:
    """Reconcile variants by strict majority, merge identical findings and order them by event.

    Args:
        evidence: The judged evidence, for event order.
        definition: The frozen definition.
        candidates: The judged candidates.
        answers: Every variant's valid findings.

    Returns:
        One finding per candidate, identical established findings of one tag merged, ordered by
        their first observed event.
    """
    order = {event.event_id: event.sequence for event in evidence.events}
    findings: list[DenseFinding] = []
    for candidate in candidates:
        rows = [answer.findings.get(candidate.candidate_id) for answer in answers]
        votes: dict[str, int] = {}
        for row in rows:
            decision = "uncertain" if row is None else row.decision
            votes[decision] = votes.get(decision, 0) + 1
        majority = [d for d in _DECISIONS if 2 * votes.get(d, 0) > len(answers)]
        winner: Decision = majority[0] if majority else "uncertain"
        chosen = next(
            (row for row in rows if majority and row is not None and row.decision == winner), None
        )
        if chosen is None:
            reasons = "; ".join(
                f"{a.version}: {r.decision}" for a, r in zip(answers, rows, strict=True) if r
            )
            rationale = f"prompt variants disagree or gave no valid answer ({reasons or 'none'})"
        else:
            rationale = chosen.rationale
        tag = definition.tag(candidate.tag_id)
        findings.append(
            DenseFinding(
                finding_id=candidate.candidate_id,
                tag_id=candidate.tag_id,
                candidate_ids=(candidate.candidate_id,),
                decision=winner,
                observed_refs=() if chosen is None else chosen.observed_refs,
                obligation=_citation(None if chosen is None else chosen.obligation),
                change=_citation(None if chosen is None else chosen.change),
                result_refs=() if chosen is None else chosen.result_refs,
                weakening=None if chosen is None else chosen.weakening,
                rationale=rationale[:4000],
                quote=None if chosen is None else chosen.quote,
                votes=votes,
                axis=tag.axis,
                polarity=tag.polarity,
                severity=tag.severity,
            )
        )
    merged: dict[tuple[str, str, tuple[str, ...]], DenseFinding] = {}
    kept: list[DenseFinding] = []
    for row in findings:
        key = (row.tag_id, row.decision, tuple(sorted(row.observed_refs)))
        if row.decision == "established" and key in merged:
            first = merged[key]
            index = kept.index(first)
            kept[index] = first.model_copy(
                update={"candidate_ids": (*first.candidate_ids, *row.candidate_ids)}
            )
            merged[key] = kept[index]
            continue
        merged.setdefault(key, row)
        kept.append(row)
    big = len(order) + 1
    return tuple(
        sorted(
            kept, key=lambda row: min((order.get(r, big) for r in row.observed_refs), default=big)
        )
    )


def _citation(value: AnswerCitation | None) -> Citation | None:
    """Convert a wire citation with at least one event into a finding citation."""
    if value is None or not value.event_refs:
        return None
    return Citation(statement=value.statement, event_refs=value.event_refs)


def _size(request: ModelRequest) -> int:
    """Measure a request exactly as the author-response owner bounds it."""
    return len(request.model_dump_json().encode())
