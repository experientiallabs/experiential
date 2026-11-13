"""Bounded read-only evidence tools and the persisted analysis loop of selectors and critics.

A point selector or critic answers through one typed tool call. Its evidence-tool profile decides
how it investigates first:

- ``forced_response``: one forced answer over the complete packet. A valid non-agentic profile,
  not agentic exploration; the class default, and available explicitly everywhere.
- ``read_only_tools``: the role opens on a compact overview and may read evidence through the
  tools in :data:`~exp.optimize.evaluation.variance.role_prompts.EVIDENCE_TOOLS` before answering.
- ``agentic``: the role plans, investigates, drafts, checks its draft against the evidence (with
  the host's validation of it, reading more if needed) and then gives a forced final answer
  (:mod:`~exp.optimize.evaluation.variance.role_prompts` holds every phase prompt).
  :data:`AGENTIC_PROFILE` is the error-mining default.

The tools only read the packet's frozen documents
(:class:`~exp.optimize.evaluation.variance.evidence.EvidencePacket`) and, when registered, the
world-learning-only :class:`~exp.optimize.evaluation.variance.evidence.TraceSearch`. They execute
nothing, write no world or study state, change no grade and never reach the actor: no actor request
is built from them. Unavailable material is refused by name; a result too large for the profile is
refused with its size, never silently truncated.

Every draw has frozen finite budgets: model calls, evidence-tool calls, input and output tokens,
wall-clock seconds and, by default, dollars. Before each call the loop checks there is room for
it and for every later mandatory phase. Without room a reading turn becomes the phase's forced
tool, and a phase without room is skipped and recorded. A draw that cannot afford its final
answer, or whose measured usage exceeds a budget, fails explicitly. Dollars are priced from
usage on each saved response; unpriceable usage makes a dollar budget unverifiable, so no further
call is made. An explicit ``maximum_usd=None`` instead requires an active operation ceiling with
a finite unpriced-attempt allowance before every new call; all other draw limits still apply.

Every model call goes through the shared author-response owner, which saves its exact request and
response; every tool result is saved as an ``evidence-tool-result`` artifact; and the whole
exchange (each call's phase, every plan, draft and self check, the prompt variant, measured usage
and the final answer) is saved as one ``role-analysis-transcript``. A saved transcript, and each
saved call of an interrupted one, replays without any call.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
)
from exp.common.models import (
    ModelCatalog,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolChoice,
    load_model_catalog,
)
from exp.common.models.token_cost import schedule_usage_cost_nano_usd
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.variance.evidence import (
    EvidencePacket,
    TraceSearch,
    execute_evidence_tool,
)
from exp.optimize.evaluation.variance.role_prompts import (
    EVIDENCE_TOOLS,
    PLAN_TOOL,
    SELF_CHECK_TOOL,
    PhaseName,
    PromptAssignment,
    draft_tool,
    loop_instruction,
    phase_prompt,
)
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.spend_ceiling import (
    MAXIMUM_UNPRICED_ATTEMPTS,
    UnpricedDispatchRefused,
    current_spend_ceiling,
)

EVIDENCE_TOOLS_PROTOCOL = "variance-evidence-tools-v2"
"""Version of the evidence tools, their schemas and the analysis loop's budget rules."""

TRANSCRIPT_TYPE = "role-analysis-transcript"
"""Artifact type of one saved role analysis."""

TOOL_RESULT_TYPE = "evidence-tool-result"
"""Artifact type of one saved evidence-tool result."""


class EvidenceBudget(ContractModel):
    """Frozen finite allowance for one selector or critic draw.

    Attributes:
        model_calls: Most model calls, including the final answer.
        tool_calls: Most evidence-tool calls executed across the draw.
        input_tokens: Most measured input tokens across the draw.
        output_tokens: Most output tokens; each call reserves its full output allowance.
        wall_clock_seconds: Most elapsed seconds from the draw's first call.
        maximum_usd: Most priced dollars across the draw; None explicitly uses the active
            operation's bounded unpriced-attempt allowance instead of a draw dollar cap.
    """

    model_calls: int = Field(default=1, ge=1, le=32, strict=True)
    tool_calls: int = Field(default=0, ge=0, le=128, strict=True)
    input_tokens: int = Field(default=400_000, ge=1, le=10_000_000, strict=True)
    output_tokens: int = Field(default=16_384, ge=1, le=1_000_000, strict=True)
    wall_clock_seconds: float = Field(default=900.0, gt=0, le=86_400)
    maximum_usd: float | None = Field(default=5.0, gt=0, le=1_000)

    def require_call_authorization(self, project: ProjectStore, identity: str) -> None:
        """Require bounded operation consent for a new request-capped, unpriced draw call.

        Saved author outcomes replay offline. New calls still pass through the admitted
        transport, which atomically reserves each physical attempt against the operation.

        Raises:
            UnpricedDispatchRefused: No active finite unpriced allowance authorizes a new call.
        """
        if self.maximum_usd is not None or any(
            project.artifacts.exists(identity + suffix) for suffix in ("-response", "-rejection")
        ):
            return
        ceiling = current_spend_ceiling()
        if ceiling is not None:
            summary = ceiling.summary()
            if (
                math.isfinite(summary.ceiling_usd)
                and 0 < summary.unpriced_attempt_ceiling <= MAXIMUM_UNPRICED_ATTEMPTS
                and summary.unpriced_attempts < summary.unpriced_attempt_ceiling
            ):
                return
        raise UnpricedDispatchRefused(
            None, "a request-capped role requires an active finite unpriced-attempt allowance"
        )


class EvidenceToolProfile(ContractModel):
    """How one role reads its evidence, and the budget each of its draws runs under.

    Attributes:
        mode: ``forced_response`` for one forced answer over the complete packet,
            ``read_only_tools`` for bounded evidence reads before the answer, or ``agentic`` for
            the plan, investigate, draft, self-check and final phases.
        budget: Frozen allowance per draw.
        maximum_result_bytes: Largest serialized tool result returned; larger ones are refused.
    """

    mode: Literal["forced_response", "read_only_tools", "agentic"] = "forced_response"
    budget: EvidenceBudget = EvidenceBudget()
    maximum_result_bytes: int = Field(default=32_000, ge=256, le=256_000, strict=True)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        """Keep the forced profile to one call and the tool profiles able to read and answer."""
        if self.mode == "forced_response" and (
            self.budget.model_calls != 1 or self.budget.tool_calls != 0
        ):
            raise ValueError("a forced response makes one model call and no tool call")
        if self.mode == "read_only_tools" and (
            self.budget.model_calls < 2 or self.budget.tool_calls < 1
        ):
            raise ValueError("a tool profile needs two model calls and one tool call at least")
        if self.mode == "agentic" and (self.budget.model_calls < 4 or self.budget.tool_calls < 1):
            raise ValueError("an agentic profile needs four model calls and one tool call at least")
        return self


AGENTIC_PROFILE = EvidenceToolProfile(
    mode="agentic",
    budget=EvidenceBudget(
        model_calls=12,
        tool_calls=24,
        input_tokens=2_000_000,
        output_tokens=98_304,
        wall_clock_seconds=1_800.0,
        maximum_usd=10.0,
    ),
)
"""The default error-mining profile: four phases plus up to eight reading turns per draw."""


class MeasuredUsage(ContractModel):
    """Usage read from saved provider responses, never assumed.

    Attributes:
        calls: Responses counted.
        measured_calls: Responses that reported token usage.
        input_tokens: Reported input tokens.
        output_tokens: Reported output tokens.
        priced_usd: Dollars over the priced responses only.
        unpriced_calls: Responses whose usage or price is unknown; ``priced_usd`` excludes them.
    """

    calls: int = Field(default=0, ge=0)
    measured_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    priced_usd: float = Field(default=0.0, ge=0)
    unpriced_calls: int = Field(default=0, ge=0)

    def plus(self, other: MeasuredUsage) -> MeasuredUsage:
        """Return the sum of two measurements."""
        return MeasuredUsage(
            calls=self.calls + other.calls,
            measured_calls=self.measured_calls + other.measured_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            priced_usd=self.priced_usd + other.priced_usd,
            unpriced_calls=self.unpriced_calls + other.unpriced_calls,
        )


def total_usage(rows: Sequence[MeasuredUsage]) -> MeasuredUsage:
    """Sum measurements."""
    total = MeasuredUsage()
    for row in rows:
        total = total.plus(row)
    return total


def response_usage(catalog: ModelCatalog, alias: str, response: ModelResponse) -> MeasuredUsage:
    """Measure one saved response and price it from the catalog's frozen rates.

    An observed provider charge is used when the response carries one. Otherwise the reported
    usage is priced with the alias's token schedule, or its per-million rates with every input
    token at the uncached rate. Unknown usage or rates stay unpriced, never zero.

    Args:
        catalog: The project's model catalog.
        alias: Alias the response was bought through.
        response: Saved provider response.

    Returns:
        The measurement of this one response.
    """
    usage = response.economics.usage
    observed = response.economics.cost_usd
    measured = usage is not None
    usd: float | None = None
    if observed is not None and math.isfinite(observed.value) and observed.value >= 0:
        usd = observed.value
    elif usage is not None:
        record = catalog.models.get(alias)
        if record is not None and record.token_prices is not None:
            nano = schedule_usage_cost_nano_usd(record.token_prices, usage)
            usd = None if nano is None else nano / 1_000_000_000
        else:
            try:
                _, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(alias)
            except (KeyError, ValueError):
                caps = None
            if caps is not None and caps.input_cost_per_million_tokens_usd is not None:
                output_rate = caps.output_cost_per_million_tokens_usd
                if output_rate is not None or usage.output_tokens == 0:
                    usd = (
                        usage.input_tokens * caps.input_cost_per_million_tokens_usd
                        + usage.output_tokens * (output_rate or 0.0)
                    ) / 1_000_000
    return MeasuredUsage(
        calls=1,
        measured_calls=int(measured),
        input_tokens=0 if usage is None else usage.input_tokens,
        output_tokens=0 if usage is None else usage.output_tokens,
        priced_usd=usd or 0.0,
        unpriced_calls=int(usd is None),
    )


def study_usage(
    store: ArtifactStore, study: ContinuationStudy, catalog: ModelCatalog, alias: str
) -> MeasuredUsage:
    """Measure every saved actor response of one continuation study.

    Args:
        store: Project artifacts holding the study's actor-call receipts.
        study: The saved study.
        catalog: The project's model catalog.
        alias: Actor alias the study bought its calls through.

    Returns:
        The sum over every branch's saved ``continuation-actor-response``; a dispatched call
        without a saved response is not counted here.
    """
    rows: list[MeasuredUsage] = []
    for branch in study.branches:
        for reference in branch.actor_calls:
            if store.read(reference.artifact_id).manifest.artifact_type != (
                "continuation-actor-response"
            ):
                continue
            response = ModelResponse.model_validate_json(
                store.read_bytes(reference.artifact_id, "response.json")
            )
            rows.append(response_usage(catalog, alias, response))
    return total_usage(rows)


def forced_request(
    instruction: str, content: JsonObject, answer: ToolSchema, *, output_tokens: int
) -> ModelRequest:
    """Build the forced single-answer request.

    Args:
        instruction: System instruction.
        content: User content, serialized canonically.
        answer: The one tool the model must call.
        output_tokens: Response allowance.

    Returns:
        The request.
    """
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content=instruction),
            ModelMessage(role="user", content=json.dumps(content, sort_keys=True)),
        ),
        tools=(answer,),
        tool_choice=ToolChoice(name=answer.name),
        maximum_output_tokens=output_tokens,
    )


class AnalysisCall(ContractModel):
    """One model call of a role analysis.

    Attributes:
        index: Call position in the draw.
        response: Saved author response; its input is the saved request intent.
        forced_answer: Whether the call forced the answer tool.
        usage: Measured usage of this call.
        phase: Loop phase the call was made in.
        forced: Tool the call forced, when it forced one.
    """

    index: int = Field(ge=0)
    response: ArtifactInput
    forced_answer: bool
    usage: MeasuredUsage
    phase: PhaseName = "final"
    forced: str | None = None


class ToolResultRecord(ContractModel):
    """One evidence-tool call and its saved result.

    Attributes:
        call: Model call that requested it.
        name: Tool name.
        arguments: Requested arguments.
        status: ``ok`` or ``refused``.
        result: Saved result artifact.
    """

    call: int = Field(ge=0)
    name: str
    arguments: JsonObject
    status: Literal["ok", "refused"]
    result: ArtifactInput


class PhaseOutput(ContractModel):
    """What one agentic phase recorded: the plan, the draft or the self check.

    Attributes:
        phase: ``plan``, ``draft`` or ``self_check``.
        call: Model call that recorded it.
        arguments: The phase tool's exact arguments.
        host_validation: For a draft, the host's validation result shown in the self check.
    """

    phase: PhaseName
    call: int = Field(ge=0)
    arguments: JsonObject
    host_validation: str | None = None


class AnalysisTranscript(ContractModel):
    """The complete persisted analysis of one selector or critic draw.

    Attributes:
        protocol: Evidence-tool protocol the draw ran under.
        identity: Draw identity.
        profile: Frozen profile and budgets.
        answer_tool: Name of the typed answer tool.
        calls: Every model call, in order.
        tools: Every evidence-tool call, in order.
        usage: Measured usage across every call.
        elapsed_seconds: Wall-clock seconds from the first call to the end.
        answer: The final structured answer arguments, or None when none was given.
        stop: ``answered``, ``budget_exhausted`` or ``protocol_failure``.
        exhausted: Budget dimensions that stopped or invalidated the draw.
        failure: Why the draw produced no usable answer.
        prompt: The prompt variant assigned to the draw, with its digests.
        outputs: Every recorded plan, draft and self check, in order.
        skipped: Agentic phases skipped because the budget had no room for them.
    """

    protocol: str = EVIDENCE_TOOLS_PROTOCOL
    identity: str
    profile: EvidenceToolProfile
    answer_tool: str
    calls: tuple[AnalysisCall, ...]
    tools: tuple[ToolResultRecord, ...]
    usage: MeasuredUsage
    elapsed_seconds: float = Field(ge=0)
    answer: JsonObject | None
    stop: Literal["answered", "budget_exhausted", "protocol_failure"]
    exhausted: tuple[str, ...] = ()
    failure: str | None = None
    prompt: PromptAssignment | None = None
    outputs: tuple[PhaseOutput, ...] = ()
    skipped: tuple[PhaseName, ...] = ()


def run_analysis(
    project: ProjectStore,
    settings: AuthorSettings,
    profile: EvidenceToolProfile,
    *,
    instruction: str,
    packet: EvidencePacket,
    content: JsonObject,
    answer: ToolSchema,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
    prompt: PromptAssignment | None = None,
    check: Callable[[JsonObject], str | None] | None = None,
    search: TraceSearch | None = None,
) -> tuple[ArtifactInput, AnalysisTranscript]:
    """Run, or replay, one bounded role analysis and persist its whole transcript.

    The caller holds the operation lock that the author-response owner requires.

    Args:
        project: Project whose artifacts hold the draw.
        settings: Frozen role model and per-call allowance.
        profile: Evidence-tool profile and budgets.
        instruction: Role system instruction (its fixed contract).
        packet: Privileged evidence; its documents back the tools.
        content: Role-specific fields added to the opening message.
        answer: The typed answer tool.
        identity: Stable identity of this draw.
        sources: Saved artifacts the packet was rendered from.
        code_revision: Producing source identity.
        prompt: The draw's prompt variant; its stance is appended to the instruction.
        check: Host validation of a draft against the packet, returning the refusal reason or
            None when valid; its result is shown in the self check.
        search: World-learning retrieval for the tool modes, when one is registered.

    Returns:
        The saved transcript reference and the transcript. Its ``answer`` is unvalidated; the
        caller validates it against the packet.
    """
    store = project.artifacts
    transcript_id = identity + "-transcript"
    if store.exists(transcript_id):
        saved = AnalysisTranscript.model_validate_json(
            store.read_bytes(transcript_id, "transcript.json")
        )
        if saved.profile != profile or saved.answer_tool != answer.name or saved.prompt != prompt:
            raise ValueError("saved role analysis was made under a different profile or prompt")
        return artifact_input(store.read(transcript_id).manifest), saved
    if prompt is not None:
        instruction += f"\n\nSTANCE ({prompt.variant})\n{prompt.stance}"
    loop = _Loop(project, settings, profile, identity, sources, code_revision, prompt)
    if profile.mode == "forced_response":
        request = forced_request(
            instruction,
            {**packet.payload, **content},
            answer,
            output_tokens=settings.output_tokens,
        )
        transcript = loop.forced(request, answer)
    else:
        retrieval = (
            search.inventory()
            if search is not None
            else {"available": False, "reason": "no world-learning retrieval index registered"}
        )
        opening = {**(packet.overview or packet.payload), **content}
        opening["world_learning_retrieval"] = retrieval
        addendum = loop_instruction(profile.mode, profile.budget.model_dump(mode="json"))
        system = instruction + addendum.replace("{answer}", answer.name)
        tools = _Tools(packet, search, profile.maximum_result_bytes)
        transcript = loop.loop(system, opening, tools, answer, check)
    inputs = {call.response for call in transcript.calls} | {t.result for t in transcript.tools}
    manifest = store.write_json(
        artifact_id=transcript_id,
        artifact_type=TRANSCRIPT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files={"transcript.json": transcript},
    )
    return artifact_input(manifest), transcript


class _Tools:
    """The read-only tools of one draw: the packet documents and optional retrieval."""

    def __init__(
        self, packet: EvidencePacket, search: TraceSearch | None, maximum_bytes: int
    ) -> None:
        """Bind the frozen documents, the retrieval and the result-size limit."""
        self.packet, self.search, self.maximum_bytes = packet, search, maximum_bytes

    def run(self, call: ToolCall) -> tuple[Literal["ok", "refused"], JsonValue]:
        """Answer one evidence-tool call."""
        return execute_evidence_tool(
            self.packet.documents, call, maximum_bytes=self.maximum_bytes, search=self.search
        )


_MANDATORY: dict[PhaseName, int] = {"plan": 4, "investigate": 3, "self_check": 2, "final": 1}
"""Model calls a phase needs, including itself, to still reach the final answer."""

_NEXT: dict[PhaseName, PhaseName] = {
    "plan": "investigate",
    "investigate": "self_check",
    "draft": "self_check",
    "self_check": "final",
}
"""The phase that follows each phase."""


class _Loop:
    """State of one analysis draw: calls made, tools run and usage measured."""

    def __init__(
        self,
        project: ProjectStore,
        settings: AuthorSettings,
        profile: EvidenceToolProfile,
        identity: str,
        sources: tuple[ArtifactInput, ...],
        code_revision: str,
        prompt: PromptAssignment | None,
    ) -> None:
        """Bind the frozen draw settings."""
        self.project, self.settings, self.profile = project, settings, profile
        self.identity, self.sources, self.revision = identity, sources, code_revision
        self.prompt = prompt
        self.catalog = load_model_catalog(project.model_catalog_path)
        self.calls: list[AnalysisCall] = []
        self.tools: list[ToolResultRecord] = []
        self.outputs: list[PhaseOutput] = []
        self.skipped: list[PhaseName] = []
        self.executed = 0
        self.usage = MeasuredUsage()
        self.started = time.monotonic()

    def _call(
        self, request: ModelRequest, *, phase: PhaseName, forced: str | None, answer: str
    ) -> ModelResponse:
        """Buy or replay one model call and measure it."""
        index = len(self.calls)
        name = (
            self.identity
            if self.profile.mode == "forced_response"
            else (f"{self.identity}-call-{index}")
        )
        if self.profile.budget.maximum_usd is None:
            name += "-request-capped"
        self.profile.budget.require_call_authorization(self.project, name)
        reference, response = author_response(
            self.project,
            self.settings,
            request,
            identity=name,
            sources=self.sources,
            code_revision=self.revision,
        )
        usage = response_usage(self.catalog, self.settings.model_alias, response)
        self.usage = self.usage.plus(usage)
        self.calls.append(
            AnalysisCall(
                index=index,
                response=reference,
                forced_answer=forced == answer,
                usage=usage,
                phase=phase,
                forced=forced,
            )
        )
        return response

    def _short(self, request: ModelRequest, count: int) -> tuple[str, ...]:
        """Return the budget dimensions without room for ``count`` more calls like ``request``."""
        budget, usage = self.profile.budget, self.usage
        short: list[str] = []
        if len(self.calls) + count > budget.model_calls:
            short.append("model_calls")
        estimate = math.ceil(len(request.model_dump_json().encode()) / 3)
        if usage.input_tokens + count * estimate > budget.input_tokens:
            short.append("input_tokens")
        if usage.output_tokens + count * self.settings.output_tokens > budget.output_tokens:
            short.append("output_tokens")
        elapsed = time.monotonic() - self.started
        pace = elapsed / len(self.calls) if self.calls else 0.0
        if elapsed + count * pace > budget.wall_clock_seconds:
            short.append("wall_clock_seconds")
        if budget.maximum_usd is None:
            return tuple(short)
        if usage.unpriced_calls:
            short.append("maximum_usd_unverifiable")
        else:
            most = max((call.usage.priced_usd for call in self.calls), default=0.0)
            if usage.priced_usd + count * most > budget.maximum_usd:
                short.append("maximum_usd")
        return tuple(short)

    def _over(self) -> tuple[str, ...]:
        """Return the budget dimensions the measured usage already exceeds."""
        budget, usage = self.profile.budget, self.usage
        over = [
            name
            for name, spent, limit in (
                ("input_tokens", usage.input_tokens, budget.input_tokens),
                ("output_tokens", usage.output_tokens, budget.output_tokens),
                ("maximum_usd", usage.priced_usd, budget.maximum_usd),
                ("wall_clock_seconds", time.monotonic() - self.started, budget.wall_clock_seconds),
            )
            if limit is not None and spent > limit
        ]
        return tuple(over)

    def _finish(
        self,
        answer_tool: str,
        stop: Literal["answered", "budget_exhausted", "protocol_failure"],
        *,
        answer: JsonObject | None = None,
        exhausted: tuple[str, ...] = (),
        failure: str | None = None,
    ) -> AnalysisTranscript:
        """Close the draw into its transcript."""
        return AnalysisTranscript(
            identity=self.identity,
            profile=self.profile,
            answer_tool=answer_tool,
            calls=tuple(self.calls),
            tools=tuple(self.tools),
            usage=self.usage,
            elapsed_seconds=time.monotonic() - self.started,
            answer=answer,
            stop=stop,
            exhausted=exhausted,
            failure=failure,
            prompt=self.prompt,
            outputs=tuple(self.outputs),
            skipped=tuple(self.skipped),
        )

    def _answered(self, response: ModelResponse, answer: str) -> AnalysisTranscript | None:
        """Close the draw when the response submitted the answer, checking measured usage."""
        submitted = [call for call in response.output.tool_calls if call.name == answer]
        if not submitted:
            return None
        if len(submitted) != 1 or response.finish_reason.value != "completed":
            return self._finish(
                answer, "protocol_failure", failure=f"role must complete exactly one {answer} call"
            )
        over = self._over()
        if over:
            return self._finish(
                answer,
                "budget_exhausted",
                answer=submitted[0].arguments,
                exhausted=over,
                failure=f"measured usage exceeded the frozen budget: {', '.join(over)}",
            )
        return self._finish(answer, "answered", answer=submitted[0].arguments)

    def forced(self, request: ModelRequest, answer: ToolSchema) -> AnalysisTranscript:
        """Run the non-agentic profile: one forced answer."""
        short = self._short(request, 1)
        if short:
            return self._finish(
                answer.name, "budget_exhausted", exhausted=short, failure="no room for one call"
            )
        response = self._call(request, phase="final", forced=answer.name, answer=answer.name)
        if len(response.output.tool_calls) != 1:
            return self._finish(
                answer.name,
                "protocol_failure",
                failure=f"role must complete exactly one {answer.name} call",
            )
        return self._answered(response, answer.name) or self._finish(
            answer.name,
            "protocol_failure",
            failure=f"role must complete exactly one {answer.name} call",
        )

    def _request(
        self, messages: list[ModelMessage], tools: tuple[ToolSchema, ...], forced: str | None
    ) -> ModelRequest:
        """Build one loop request, forcing ``forced`` when it is set."""
        return ModelRequest(
            messages=tuple(messages),
            tools=tools,
            tool_choice=ToolChoice(name=forced) if forced else "required",
            maximum_output_tokens=self.settings.output_tokens,
        )

    def _run_reads(self, calls: Sequence[ToolCall], tools: _Tools) -> list[ModelMessage]:
        """Execute read calls within the tool budget, persisting each result."""
        messages: list[ModelMessage] = []
        for position, call in enumerate(calls):
            if self.executed >= self.profile.budget.tool_calls:
                status, result = "refused", "evidence-tool budget exhausted; not executed"
            else:
                self.executed += 1
                status, result = tools.run(call)
            self.tools.append(self._save_tool(position, call, status, result))
            messages.append(_tool_message(call, {"status": status, "result": result}))
        return messages

    def loop(
        self,
        instruction: str,
        opening: JsonObject,
        tools: _Tools,
        answer: ToolSchema,
        check: Callable[[JsonObject], str | None] | None,
    ) -> AnalysisTranscript:
        """Run the tool-reading phases until the final answer or an explicit stop.

        The read-only profile has one reading phase whose own tool is the answer. The agentic
        profile runs plan, investigate (ending in a draft), self check and final. A phase runs
        only when the budget has room for it and every later mandatory phase; a reading turn runs
        only with room for one more call beyond that and evidence-tool budget left, otherwise the
        phase's own tool is forced. A phase without room is recorded as skipped (a skipped draft
        also skips the self check); the final answer is attempted while one call fits, and its
        absence is an explicit budget stop.
        """
        agentic = self.profile.mode == "agentic"
        phase: PhaseName = "plan" if agentic else "investigate"
        own_tool: dict[PhaseName, ToolSchema] = {
            "plan": PLAN_TOOL,
            "investigate": draft_tool(answer) if agentic else answer,
            "self_check": SELF_CHECK_TOOL,
            "final": answer,
        }
        mandatory = _MANDATORY if agentic else {"investigate": 1, "final": 1}
        if agentic:
            opening = {**opening, "phase": "plan", "phase_instruction": phase_prompt("plan", "")}
        messages = [
            ModelMessage(role="system", content=instruction),
            ModelMessage(role="user", content=json.dumps(opening, sort_keys=True)),
        ]
        while True:
            own = own_tool[phase]
            reading = phase in ("investigate", "self_check")
            offered = (*EVIDENCE_TOOLS, own) if reading else (own,)
            probe = self._request(messages, offered, None)
            free = (
                reading
                and self.executed < self.profile.budget.tool_calls
                and not self._short(probe, mandatory[phase] + 1)
            )
            short = self._short(probe, mandatory[phase])
            if not free and short:
                if phase == "final" or not agentic:
                    return self._finish(
                        answer.name,
                        "budget_exhausted",
                        exhausted=short,
                        failure="budget exhausted before an answer: " + ", ".join(short),
                    )
                self.skipped.extend(("investigate", "self_check") if reading else (phase,))
                phase = "final" if phase == "investigate" else _NEXT[phase]
                body: JsonObject = {"phase": phase, "skipped": list(self.skipped)}
                body["phase_instruction"] = phase_prompt(phase, answer.name)
                messages.append(ModelMessage(role="user", content=json.dumps(body, sort_keys=True)))
                continue
            forced = None if free else own.name
            try:
                response = self._call(
                    self._request(messages, offered, forced),
                    phase=phase,
                    forced=forced,
                    answer=answer.name,
                )
            except ProviderResponseError as error:
                # A response with no usable output (runaway reasoning past the allowance, no
                # text or tool call) ends this draw as a failure; its rejection is saved, so a
                # replay ends the same way. It never fails the whole stage.
                return self._finish(
                    answer.name,
                    "protocol_failure",
                    failure=f"model call gave no usable response: {str(error)[:300]}",
                )
            done = self._answered(response, answer.name) if own is answer else None
            if done is not None:
                return done
            calls = response.output.tool_calls
            mine = [call for call in calls if call.name == own.name]
            if phase == "final" or (forced and not mine):
                return self._finish(
                    answer.name,
                    "protocol_failure",
                    failure=f"role did not call {own.name} when forced in the {phase} phase",
                )
            if not calls or response.finish_reason.value != "completed":
                return self._finish(
                    answer.name, "protocol_failure", failure="role made no complete tool call"
                )
            messages.append(ModelMessage(role="assistant", assistant_action=response.output))
            messages.extend(self._run_reads([c for c in calls if c.name != own.name], tools))
            if mine:
                phase = self._record(phase, mine, messages, answer.name, check)

    def _record(
        self,
        phase: PhaseName,
        mine: Sequence[ToolCall],
        messages: list[ModelMessage],
        answer: str,
        check: Callable[[JsonObject], str | None] | None,
    ) -> PhaseName:
        """Record a plan, draft or self check, answer it with the next phase, and return it."""
        recorded: PhaseName = "draft" if phase == "investigate" else phase
        validation = None
        if recorded == "draft":
            validation = ("not checked" if check is None else check(mine[0].arguments)) or "valid"
        self.outputs.append(
            PhaseOutput(
                phase=recorded,
                call=len(self.calls) - 1,
                arguments=mine[0].arguments,
                host_validation=validation,
            )
        )
        following = _NEXT[recorded]
        text = phase_prompt(following, answer)
        if recorded == "draft":
            text = phase_prompt("draft", answer) + "\n" + text
        body: JsonObject = {
            "status": "recorded",
            "next_phase": following,
            "phase_instruction": text,
        }
        if validation is not None:
            body["host_validation"] = validation
        messages.append(_tool_message(mine[0], body))
        messages.extend(
            _tool_message(extra, {"status": "refused", "result": "one record per phase"})
            for extra in mine[1:]
        )
        return following

    def _save_tool(
        self, position: int, call: ToolCall, status: Literal["ok", "refused"], result: JsonValue
    ) -> ToolResultRecord:
        """Persist one tool result, reusing an identical saved result on resume."""
        store = self.project.artifacts
        index = len(self.calls) - 1
        artifact_id = f"{self.identity}-call-{index}-tool-{position}"
        body: JsonObject = {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
            "status": status,
            "result": result,
        }
        if store.exists(artifact_id):
            if json.loads(store.read_bytes(artifact_id, "result.json")) != body:
                raise ValueError("saved evidence-tool result differs from the frozen documents")
            reference = artifact_input(store.read(artifact_id).manifest)
        else:
            reference = artifact_input(
                store.write_json(
                    artifact_id=artifact_id,
                    artifact_type=TOOL_RESULT_TYPE,
                    envelope=ArtifactEnvelope(
                        schema_version=1,
                        created_at=datetime.now(UTC),
                        code_revision=self.revision,
                        inputs=(self.calls[-1].response,),
                    ),
                    files={"result.json": body},
                )
            )
        return ToolResultRecord(
            call=index, name=call.name, arguments=call.arguments, status=status, result=reference
        )


def _tool_message(call: ToolCall, body: JsonObject) -> ModelMessage:
    """Return the tool-result message answering one call."""
    return ModelMessage(
        role="tool", tool_call_id=call.call_id, content=json.dumps(body, sort_keys=True)
    )


def evidence_tools_sha256() -> str:
    """Return the identity of the tool set and the loop protocol."""
    return sha256_json(
        {
            "protocol": EVIDENCE_TOOLS_PROTOCOL,
            "tools": [tool.model_dump(mode="json") for tool in EVIDENCE_TOOLS],
            "mandatory": _MANDATORY,
        }
    )
