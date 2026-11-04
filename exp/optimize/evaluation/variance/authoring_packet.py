"""The one evidence packet both the world author and the rubric author read, and nothing else.

An :class:`AuthoringPacket` binds, for one decision family:

- the **original sources**: trace and evidence references (each checked against the frozen
  exposure plan, so no final-evaluation lineage reaches an author), the saved rule book, its
  rules with their evidence status, the contradictions it keeps, and its open research
  questions;
- the **exact task and variant identities**: the public task, and for every scenario variant
  its complete (state, instruction) identity, so an instruction-only variant sharing its world
  state with another is still a distinct task;
- the **pre-decision checkpoint**: the saved checkpoint reference, the before-state and the
  public conversation the actor had seen there;
- the **decision brief**, marked as a proposal: the alternatives, axes and correct options the
  decision and scenario authors proposed. It is never authority; every claim derived from it is
  a ``hypothesis`` and keeps that status;
- the **behavioral coalition** (:class:`~exp.optimize.evaluation.variance.coalition.
  CoalitionEvidence`): every grouped attempt of every tier with its judged label, distinct
  mistakes, recoveries and valid alternatives, selector rationales, critic estimates beside
  judged rates, judged contrast outcomes per option and optional TraceML step labels, each row
  with its source locator and evidence status and none from final evaluation;
- the **evidence contract** of the episode judge: which events the harness and the world must
  record so a rubric criterion can be observed.

Derived claims keep their evidence status: a rule the book supports is ``supported``, one with a
supporting native or specification witness is ``admitted``, an open rule is ``unknown`` and a
contradicted rule is ``contradicted``, never voted away. Neither author's output is part of the
packet, so the two authors stay independent by construction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Self

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
from exp.common.models import ModelMessage, ModelRequest, ModelResponse, ToolChoice
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.variance.coalition import CoalitionEvidence
from exp.optimize.evaluation.variance.contracts import (
    DecisionOption,
    DecisionState,
    InitializationMode,
    Scenario,
    StateEdit,
    VariationAxis,
    WitnessStep,
    apply_edits,
    resolve_pointer,
)
from exp.optimize.evaluation.variance.episodes import EVENT_CONTRACT_VERSION, EventKind
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.induction.contracts import InductionEpisode
from exp.simulation.rules.model import ResearchQuestion, Rule, RuleBook, ToolContract

AUTHORING_PACKET_PROTOCOL = "authoring-packet-v3"
"""Version of the packet contract; change it whenever what authors are shown changes."""

PACKET_TYPE = "authoring-packet"
"""Artifact type of a saved packet."""

PACKET_FILE = "packet.json"
"""Data file of a saved packet."""

EvidenceStatus = Literal["admitted", "supported", "unknown", "contradicted", "hypothesis"]
"""How far a claim is established: independently witnessed, supported by traces, open,
contradicted, or a proposal no evidence has tested."""

ClaimOrigin = Literal["rule_book", "decision_brief", "source"]
"""Where a claim came from."""

SourceKind = Literal["trace", "evidence", "policy", "specification"]
"""What an original source is."""

VariantAxis = Literal["baseline", "state", "instruction", "state_and_instruction"]
"""Which declared axes a variant changes relative to the checkpoint."""

HARNESS_EVENTS: tuple[EventKind, ...] = (
    "user_message",
    "simulator_message",
    "assistant_message",
    "tool_call",
    "final_answer",
    "harness_note",
)
"""Events only the harness can record: dialogue and dispatch, never a per-tool program."""

WORLD_EVENTS: tuple[EventKind, ...] = (
    "tool_result",
    "tool_error",
    "receipt",
    "state_snapshot",
    "exogenous_event",
)
"""Events the world records: replies, errors, receipts, state and exogenous effects."""


class SourceRef(ContractModel):
    """One original source an author may cite.

    Attributes:
        source_id: Stable identity, cited as ``source:<source_id>``.
        kind: What the source is.
        trace_id: Canonical trace identity, for a trace; checked against the exposure plan.
        artifact: Saved artifact holding the source, when it is saved.
        summary: What the source shows, in one or two sentences.
    """

    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    kind: SourceKind
    trace_id: str | None = None
    artifact: ArtifactInput | None = None
    summary: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _trace_identity(self) -> Self:
        """Require a trace identity exactly for trace sources."""
        if (self.kind == "trace") != (self.trace_id is not None):
            raise ValueError("a trace source names its trace_id, and only a trace source does")
        return self


class PacketClaim(ContractModel):
    """One claim an author may rely on, with the evidence status it actually has.

    Attributes:
        claim_id: Stable identity, cited as ``claim:<claim_id>``.
        statement: The claim in one sentence.
        status: How far the claim is established.
        origin: Where it came from.
        supported_by: Packet locators for it.
        contradicted_by: Packet locators against it; kept, never voted away.
    """

    claim_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    statement: str = Field(min_length=1, max_length=2000)
    status: EvidenceStatus
    origin: ClaimOrigin
    supported_by: tuple[str, ...] = ()
    contradicted_by: tuple[str, ...] = ()


class PacketVariant(ContractModel):
    """One exact task variant: its complete world state and instruction.

    Attributes:
        variant_id: Scenario identity, cited as ``variant:<variant_id>``.
        axis: Which declared axes it changes.
        instruction: The exact public instruction of the variant.
        obligations: What the instruction obliges, from the declared instruction axis; None
            for a state-only decision, whose instruction is the task's own.
        obligation_id: Identity of those obligations; a rewording shares its equivalent's.
        initialization: ``resume`` keeps the checkpoint's prefix; ``fresh_briefing`` starts anew.
        briefing: The proposed fresh briefing text, exactly for a fresh briefing.
        edits: The declared state edits applied to the checkpoint's before-state.
        state: The complete edited before-state.
        state_sha256: Digest of that state.
        task_sha256: Digest of the complete (task, instruction, state) identity.
        witness: Proposed legal observation calls that reveal the deciding facts.
    """

    variant_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    axis: VariantAxis
    instruction: str = Field(min_length=1)
    obligations: str | None = None
    obligation_id: str | None = None
    initialization: InitializationMode = "resume"
    briefing: str | None = None
    edits: tuple[StateEdit, ...] = ()
    state: JsonObject
    state_sha256: Sha256
    task_sha256: Sha256
    witness: tuple[WitnessStep, ...] = ()


class DecisionBrief(ContractModel):
    """The decision and scenario authors' proposal: alternatives, axes and correct options.

    The brief is analysis, not authority. It does not list every valid actor action and its
    proposed correct options are hypotheses for the authors to test against the sources.

    Attributes:
        authority: Always ``proposal``.
        decision: The decision the family sets up.
        options: Proposed competing actions.
        axes: Proposed variation axes.
        relevant_paths: State paths the decision is proposed to depend on.
        proposed_correct: Proposed correct option per variant.
        rationales: The scenario author's rationale per variant.
        unresolved: Interpretations the brief could not settle.
    """

    authority: Literal["proposal"] = "proposal"
    decision: str = Field(min_length=1)
    options: tuple[DecisionOption, ...]
    axes: tuple[VariationAxis, ...]
    relevant_paths: tuple[str, ...]
    proposed_correct: dict[str, str]
    rationales: dict[str, str]
    unresolved: tuple[str, ...] = ()


class EvidenceContract(ContractModel):
    """What the episode judge can be shown, and who must record it.

    Attributes:
        event_contract: Version of the episode event contract.
        harness_events: Event kinds the harness records (dialogue and dispatch).
        world_events: Event kinds the world records (replies, errors, receipts, state).
        note: The capture rule authors must respect.
    """

    event_contract: str = EVENT_CONTRACT_VERSION
    harness_events: tuple[EventKind, ...] = HARNESS_EVENTS
    world_events: tuple[EventKind, ...] = WORLD_EVENTS
    note: str = (
        "The harness records every user, simulator and assistant message, every tool request and "
        "the final answer in one ordered append-only ledger. A per-tool transition program cannot "
        "observe or author dialogue; it records only its own replies, errors and state effects."
    )

    def kinds(self) -> frozenset[str]:
        """Every event kind the contract lets a judge observe."""
        return frozenset((*self.harness_events, *self.world_events))


class ResearchRequest(ContractModel):
    """One piece of evidence an author needs before it can return a ready draft.

    Attributes:
        question: What must be established, answerable by a trace, witness or specification.
        why_it_matters: Which behavior, criterion or variant depends on the answer.
        blocks: Variant identities the gap blocks; empty means the whole family.
        cited: Packet locators that raised the question.
    """

    question: str = Field(min_length=1, max_length=1200)
    why_it_matters: str = Field(min_length=1, max_length=1200)
    blocks: tuple[str, ...] = ()
    cited: tuple[str, ...] = ()


class AuthoringPacket(ContractModel):
    """Everything both authors read about one decision family; never either author's output.

    Attributes:
        protocol: Packet contract version.
        family_id: Stable family identity.
        task: The public task, without evaluator answers.
        checkpoint: Saved pre-decision checkpoint the variants start from.
        before_state: Simulated domain state at that checkpoint.
        public_history: The public conversation the actor had seen at the checkpoint.
        remaining_calls: Tool calls the actor has left at the checkpoint.
        sources: Original trace and evidence sources.
        rule_book: The saved rule book artifact.
        rules: Rules about the task's tools and environment-wide rules, every status kept.
        contracts: Observed tool contracts; observed shapes, not complete schemas.
        questions: Open research questions bearing on the family.
        claims: Every claim, with its evidence status.
        observed_episodes: Source-derived replay episodes induced programs are validated on.
        variants: The exact task variants.
        brief: The decision brief, a proposal.
        coalition: Every attempt, tier, pattern, selector, critic and contrast behind the family.
        evidence: The judge's evidence contract.
    """

    protocol: str = AUTHORING_PACKET_PROTOCOL
    family_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    task: TaskCase
    checkpoint: ArtifactInput
    before_state: JsonObject
    public_history: tuple[ModelMessage, ...] = ()
    remaining_calls: int = Field(ge=0)
    sources: tuple[SourceRef, ...] = Field(min_length=1)
    rule_book: ArtifactInput
    rules: tuple[Rule, ...] = ()
    contracts: dict[str, ToolContract] = Field(default_factory=dict)
    questions: tuple[ResearchQuestion, ...] = ()
    claims: tuple[PacketClaim, ...] = ()
    observed_episodes: tuple[InductionEpisode, ...] = ()
    variants: tuple[PacketVariant, ...] = Field(min_length=1)
    brief: DecisionBrief
    coalition: CoalitionEvidence = Field(default_factory=CoalitionEvidence)
    evidence: EvidenceContract = Field(default_factory=EvidenceContract)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        """Require unique identities and distinct complete task identities per variant."""
        for name, identities in (
            ("source", [source.source_id for source in self.sources]),
            ("claim", [claim.claim_id for claim in self.claims]),
            ("variant", [variant.variant_id for variant in self.variants]),
            ("task", [variant.task_sha256 for variant in self.variants]),
        ):
            if len(set(identities)) != len(identities):
                raise ValueError(f"packet {name} identities must be distinct")
        if self.task.reference_answer is not None:
            raise ValueError("an authoring packet never carries a private expected answer")
        return self

    @property
    def identity(self) -> Sha256:
        """Digest of the complete packet."""
        return sha256_json(self)

    def locators(self) -> frozenset[str]:
        """Every fixed locator an author may cite, excluding ``state:`` pointers."""
        return frozenset(
            {
                "task",
                "checkpoint",
                "brief",
                "evidence_contract",
                *(f"prefix:{index}" for index in range(len(self.public_history))),
                *(f"source:{source.source_id}" for source in self.sources),
                *(f"rule:{rule.rule_id}" for rule in self.rules),
                *(f"question:{index}" for index in range(len(self.questions))),
                *(f"claim:{claim.claim_id}" for claim in self.claims),
                *(f"variant:{variant.variant_id}" for variant in self.variants),
                *(f"tool:{tool.name}" for tool in self.task.tools),
                *self.coalition.locators(),
            }
        )

    def unsupplied(self, refs: Sequence[str]) -> tuple[str, ...]:
        """Return cited locators the packet did not supply, sorted.

        A ``state:<pointer>`` locator is supplied when it resolves in the before-state or in
        any variant's state.

        Args:
            refs: Every locator an answer cited.

        Returns:
            The invented locators.
        """
        supplied = self.locators()
        states = (self.before_state, *(variant.state for variant in self.variants))
        return tuple(
            sorted(
                ref
                for ref in set(refs)
                if ref not in supplied
                and not (
                    ref.startswith("state:/")
                    and any(resolve_pointer(state, ref[len("state:") :])[0] for state in states)
                )
            )
        )

    def variant(self, variant_id: str) -> PacketVariant:
        """Return one variant by identity.

        Raises:
            KeyError: No variant has that identity.
        """
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise KeyError(f"packet has no variant {variant_id!r}")


def variant_task_sha256(task: TaskCase, instruction: str, state: JsonObject) -> Sha256:
    """Return the complete task identity of one (state, instruction) variant.

    Args:
        task: The public task.
        instruction: The variant's exact instruction.
        state: The variant's complete before-state.

    Returns:
        The digest; two variants differ exactly when instruction or state differs.
    """
    return sha256_json(
        {
            "task": task.model_copy(update={"instruction": instruction}).model_dump(mode="json"),
            "state": state,
        }
    )


def rule_claims(rules: Sequence[Rule]) -> tuple[PacketClaim, ...]:
    """Turn rules into claims that keep each rule's evidence status.

    Args:
        rules: Rules shown to the authors.

    Returns:
        One claim per rule: ``admitted`` with a supporting independent witness, otherwise the
        rule's own status. Contradicting evidence stays attached.
    """
    return tuple(
        PacketClaim(
            claim_id=f"rule.{rule.rule_id}",
            statement=rule.statement,
            status="admitted"
            if rule.status == "supported" and any(w.supports for w in rule.witnesses)
            else rule.status,
            origin="rule_book",
            supported_by=(f"rule:{rule.rule_id}",),
            contradicted_by=tuple(rule.counter_evidence),
        )
        for rule in rules
    )


def build_authoring_packet(
    *,
    family_id: str,
    task: TaskCase,
    checkpoint: ArtifactInput,
    before_state: JsonObject,
    public_history: tuple[ModelMessage, ...],
    remaining_calls: int,
    sources: tuple[SourceRef, ...],
    exposure: ExposurePlan,
    rule_book: ArtifactInput,
    book: RuleBook,
    decision: DecisionState | None,
    scenarios: tuple[Scenario, ...],
    observed_episodes: tuple[InductionEpisode, ...] = (),
    extra_claims: tuple[PacketClaim, ...] = (),
    unresolved: tuple[str, ...] = (),
    coalition: CoalitionEvidence | None = None,
) -> AuthoringPacket:
    """Assemble the packet from original sources, the coalition, the rule book and the brief.

    Args:
        family_id: Stable family identity.
        task: The public task.
        checkpoint: Saved pre-decision checkpoint.
        before_state: Simulated domain state at the checkpoint.
        public_history: Public conversation at the checkpoint.
        remaining_calls: Tool calls the actor has left there.
        sources: Original sources; every trace must be outside final evaluation.
        exposure: The frozen data roles.
        rule_book: The saved rule book artifact ``book`` was read from.
        book: The rule book.
        decision: The proposed decision state, or None for a whole-task packet: an episode
            judged from the task's start before any decision point exists, such as the
            error-mining task's. Its brief proposes no option, axis or correct choice.
        scenarios: The proposed scenario variants; none for a whole-task packet.
        observed_episodes: Source-derived replay episodes for program validation.
        extra_claims: Further source claims, with their own evidence status.
        unresolved: Interpretations the brief left open.
        coalition: The behavioral coalition from error mining and contrasts.

    Returns:
        The packet. A baseline variant is added when no scenario keeps the checkpoint state and
        instruction, so the unedited task stays comparable.

    Raises:
        ValueError: A source trace or coalition row belongs to final evaluation, a scenario
            edits an absent path, a whole-task packet has scenarios, or two variants have the
            same complete task identity.
    """
    if decision is None and scenarios:
        raise ValueError("a whole-task packet has no decision, so it cannot carry scenarios")
    final = sorted(
        source.trace_id
        for source in sources
        if source.trace_id is not None and exposure.role_of(source.trace_id) == "final_evaluation"
    )
    if final:
        raise ValueError(
            f"{len(final)} sources are reserved for final evaluation, for example {final[0]}; "
            "authors may read only world-learning and mining sources"
        )
    coalition = coalition or CoalitionEvidence()
    coalition.check_exposure(exposure)
    names = {tool.name for tool in task.tools}
    rules = tuple(rule for rule in book.rules if rule.tool is None or rule.tool in names)
    questions = tuple(item for item in book.questions if item.tool is None or item.tool in names)
    variants = [
        _variant(task, before_state, scenario, decision)
        for scenario in scenarios
        if decision is not None
    ]
    baseline = variant_task_sha256(task, task.instruction, before_state)
    if all(variant.task_sha256 != baseline for variant in variants):
        variants.insert(
            0,
            PacketVariant(
                variant_id="baseline",
                axis="baseline",
                instruction=task.instruction,
                state=before_state,
                state_sha256=sha256_json(before_state),
                task_sha256=baseline,
            ),
        )
    brief_claims = tuple(
        PacketClaim(
            claim_id=f"brief.{scenario.scenario_id}",
            statement=(
                f"In variant {scenario.scenario_id} the proposed correct option is "
                f"{scenario.correct_option}: {scenario.rationale}"
            ),
            status="hypothesis",
            origin="decision_brief",
            supported_by=("brief",),
        )
        for scenario in scenarios
    )
    return AuthoringPacket(
        family_id=family_id,
        task=task,
        checkpoint=checkpoint,
        before_state=before_state,
        public_history=public_history,
        remaining_calls=remaining_calls,
        sources=sources,
        rule_book=rule_book,
        rules=rules,
        contracts={name: row for name, row in book.contracts.items() if name in names},
        questions=questions,
        claims=(*rule_claims(rules), *brief_claims, *extra_claims),
        observed_episodes=observed_episodes,
        variants=tuple(variants),
        brief=_brief(decision, scenarios, unresolved),
        coalition=coalition,
    )


def save_packet(
    project: ProjectStore, packet: AuthoringPacket, *, code_revision: str
) -> ArtifactInput:
    """Save a packet, or adopt the identical one already saved.

    Args:
        project: Project receiving the artifact.
        packet: The packet.
        code_revision: Producing source identity.

    Returns:
        The saved packet reference, whose inputs pin the checkpoint, rule book and sources.
    """
    store = project.artifacts
    artifact_id = stable_id("authoring-packet", {"packet": packet.identity})
    if store.exists(artifact_id):
        reference = artifact_input(store.read(artifact_id).manifest)
        if load_packet(project, reference) != packet:
            raise ValueError(f"saved packet {artifact_id} differs from the packet being saved")
        return reference
    inputs = {packet.checkpoint, packet.rule_book}
    inputs |= {source.artifact for source in packet.sources if source.artifact is not None}
    inputs |= set(packet.coalition.references)
    manifest = store.write_json(
        artifact_id=artifact_id,
        artifact_type=PACKET_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files={PACKET_FILE: packet},
    )
    return artifact_input(manifest)


def load_packet(project: ProjectStore, reference: ArtifactInput) -> AuthoringPacket:
    """Read one exact saved packet.

    Args:
        project: Project holding the packet.
        reference: The packet artifact and its manifest digest.

    Returns:
        The packet.

    Raises:
        ValueError: The reference is not the saved packet it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        PACKET_TYPE
    ):
        raise ValueError(
            f"artifact {reference.artifact_id} is not the pinned authoring packet; pass the "
            "reference save_packet returned"
        )
    return AuthoringPacket.model_validate_json(stored.payloads[PACKET_FILE])


WHOLE_TASK = (
    "The whole task, judged from its start; no decision point has been chosen, so the brief "
    "proposes no option, axis or correct choice."
)
"""The decision text of a whole-task packet's brief."""


def _brief(
    decision: DecisionState | None, scenarios: tuple[Scenario, ...], unresolved: tuple[str, ...]
) -> DecisionBrief:
    """Return the proposal the decision and scenario authors made, or the whole-task brief."""
    if decision is None:
        return DecisionBrief(
            decision=WHOLE_TASK,
            options=(),
            axes=(),
            relevant_paths=(),
            proposed_correct={},
            rationales={},
            unresolved=unresolved,
        )
    return DecisionBrief(
        decision=decision.decision,
        options=decision.options,
        axes=decision.axes,
        relevant_paths=decision.relevant_paths,
        proposed_correct={s.scenario_id: s.correct_option for s in scenarios},
        rationales={s.scenario_id: s.rationale for s in scenarios},
        unresolved=unresolved,
    )


def _variant(
    task: TaskCase, before: JsonObject, scenario: Scenario, decision: DecisionState
) -> PacketVariant:
    """Realize one proposed scenario as an exact (state, instruction) variant.

    Raises:
        ValueError: The scenario edits an absent path, or gives an instruction the decision's
            instruction axis does not declare.
    """
    axis_values = decision.instruction_axis
    value = None if axis_values is None else axis_values.value(scenario.instruction)
    if scenario.instruction is not None and value is None:
        raise ValueError(
            f"scenario {scenario.scenario_id} gives an instruction the decision does not "
            "declare as an instruction value"
        )
    state = apply_edits(before, scenario.edits)
    changed = state != before
    instruction = scenario.instruction or task.instruction
    told = instruction != task.instruction
    axis: VariantAxis = (
        "state_and_instruction"
        if changed and told
        else "state"
        if changed
        else "instruction"
        if told
        else "baseline"
    )
    return PacketVariant(
        variant_id=scenario.scenario_id,
        axis=axis,
        instruction=instruction,
        obligations=None if value is None else value.obligations,
        obligation_id=None if value is None else value.equivalent_to or value.value_id,
        initialization=scenario.initialization,
        briefing=None if scenario.briefing is None else scenario.briefing.text,
        edits=scenario.edits,
        state=state,
        state_sha256=sha256_json(state),
        task_sha256=variant_task_sha256(task, instruction, state),
        witness=scenario.witness,
    )


def author_request(
    instruction: str, packet: AuthoringPacket, *, tool: str, schema: JsonObject, output_tokens: int
) -> ModelRequest:
    """Build one forced single-tool author request whose user content is the packet alone.

    Args:
        instruction: The author's system prompt.
        packet: The authoring packet, rendered canonically as data.
        tool: Name of the one tool the author must call.
        schema: JSON schema of the tool's input.
        output_tokens: Response allowance.

    Returns:
        The request.
    """
    payload = {
        "authoring_packet": packet.model_dump(mode="json"),
        "citable_locators": sorted(packet.locators()),
    }
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content=instruction),
            ModelMessage(role="user", content=json.dumps(payload, sort_keys=True)),
        ),
        tools=(ToolSchema(name=tool, description="Typed answer", input_schema=schema),),
        tool_choice=ToolChoice(name=tool),
        maximum_output_tokens=output_tokens,
    )


def buy_author_draw(
    project: ProjectStore,
    settings: AuthorSettings,
    request: ModelRequest,
    *,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[ArtifactInput, ModelResponse] | str:
    """Buy or replay one author response, returning why none exists instead of raising.

    The shared author-response owner saves the intent before dispatch and the response after
    it, so a draw is bought once, replayed exactly and never silently redrawn.

    Args:
        project: Project whose artifacts hold the draw.
        settings: Frozen author model and allowance.
        request: The forced-tool request.
        identity: Stable identity of the draw.
        sources: Saved artifacts the request was rendered from.
        code_revision: Producing source identity.

    Returns:
        The saved response and its reference, or a bounded failure reason.
    """
    try:
        return author_response(
            project,
            settings,
            request,
            identity=identity,
            sources=tuple(sorted(set(sources), key=lambda ref: ref.artifact_id)),
            code_revision=code_revision,
        )
    except (ValueError, ProviderTransportError, DispatchCapExhaustedError, TimeoutError) as error:
        return bounded_reason(error)


def forced_tool_arguments(response: ModelResponse, tool: str) -> object:
    """Return the arguments of the single completed forced tool call.

    Args:
        response: Saved provider response.
        tool: Expected tool name.

    Returns:
        The tool call's arguments.

    Raises:
        ValueError: The response is incomplete or did not make exactly that one call.
    """
    calls = response.output.tool_calls
    if response.finish_reason.value != "completed" or len(calls) != 1 or calls[0].name != tool:
        raise ValueError(f"author must complete exactly one {tool} call; saved output retained")
    return calls[0].arguments


def save_author_result(
    project: ProjectStore,
    *,
    artifact_id: str,
    artifact_type: str,
    result: ContractModel,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> ArtifactInput:
    """Persist one parsed author result beside the saved response and packet it came from.

    Args:
        project: Project receiving the artifact.
        artifact_id: Identity of the result.
        artifact_type: Domain type of the result.
        result: The result, problems and failure included.
        inputs: The saved response and packet.
        code_revision: Producing source identity.

    Returns:
        The saved result reference.
    """
    manifest = project.artifacts.write_json(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
        ),
        files={"draft.json": result},
    )
    return artifact_input(manifest)


def check_research(packet: AuthoringPacket, research: Sequence[ResearchRequest]) -> None:
    """Refuse research requests that cite invented locators or block unknown variants.

    Args:
        packet: The packet the author was shown.
        research: The author's research requests.

    Raises:
        ValueError: A request cites a locator the packet did not supply, or blocks a variant
            the packet does not hold.
    """
    invented = packet.unsupplied([ref for item in research for ref in item.cited])
    if invented:
        raise ValueError(f"research requests cite locators that were not supplied: {invented}")
    known = {variant.variant_id for variant in packet.variants}
    unknown = sorted({block for item in research for block in item.blocks} - known)
    if unknown:
        raise ValueError(f"research requests block unknown variants {unknown}")


def bounded_reason(error: BaseException) -> str:
    """Return a bounded failure reason naming the error type."""
    return (type(error).__name__ + ": " + str(error))[:2000]
