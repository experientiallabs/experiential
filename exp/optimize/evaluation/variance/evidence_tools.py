"""Bounded read-only evidence tools and the persisted analysis loop of selectors and critics.

A point selector or critic answers through one typed tool call. Its evidence-tool profile decides
how it may read first:

- ``forced_response``: one forced answer over the complete packet. It is a valid non-agentic
  profile, not agentic exploration, and is the default.
- ``read_only_tools``: the role opens on a compact overview and may fetch exact evidence by
  locator, through the tools in :data:`EVIDENCE_TOOLS`, before it submits its answer.

The tools only read the packet's frozen documents
(:class:`~exp.optimize.evaluation.variance.evidence.EvidencePacket`). They execute nothing, write
no world or study state, change no grade and never reach the actor: the packet is privileged
analysis evidence and no actor request is built from it. A result too large for the profile is
refused with that reason, never silently truncated.

Every draw has frozen finite budgets: model calls, evidence-tool calls, input and output tokens,
wall-clock seconds and dollars. Before each call the loop checks there is room for it; when there
is room for only one more call, or the model-call or tool-call budget is spent, that call forces
the answer tool. A draw that runs out before answering, or whose measured usage exceeds a budget,
fails explicitly. Dollars are priced from the usage on each saved response; a call whose usage
cannot be priced makes the dollar budget unverifiable, so no further call is made.

Every model call goes through the shared author-response owner, which saves its exact request
and response; every tool result is saved as an ``evidence-tool-result`` artifact; and the whole
exchange, its measured usage and the final structured answer are saved as one
``role-analysis-transcript``. A saved transcript replays without any call.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
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
from exp.optimize.evaluation.variance.evidence import EvidencePacket
from exp.runtime.models.registry import RuntimeModelCatalog

EVIDENCE_TOOLS_PROTOCOL = "variance-evidence-tools-v1"
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
        maximum_usd: Most priced dollars across the draw.
    """

    model_calls: int = Field(default=1, ge=1, le=32, strict=True)
    tool_calls: int = Field(default=0, ge=0, le=128, strict=True)
    input_tokens: int = Field(default=400_000, ge=1, le=10_000_000, strict=True)
    output_tokens: int = Field(default=16_384, ge=1, le=1_000_000, strict=True)
    wall_clock_seconds: float = Field(default=900.0, gt=0, le=86_400)
    maximum_usd: float = Field(default=5.0, gt=0, le=1_000)


class EvidenceToolProfile(ContractModel):
    """How one role reads its evidence, and the budget each of its draws runs under.

    Attributes:
        mode: ``forced_response`` for one forced answer over the complete packet, or
            ``read_only_tools`` for bounded evidence reads before the answer.
        budget: Frozen allowance per draw.
        maximum_result_bytes: Largest serialized tool result returned; larger ones are refused.
    """

    mode: Literal["forced_response", "read_only_tools"] = "forced_response"
    budget: EvidenceBudget = EvidenceBudget()
    maximum_result_bytes: int = Field(default=32_000, ge=256, le=256_000, strict=True)

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        """Keep the forced profile to exactly one call and the tool profile able to read."""
        if self.mode == "forced_response" and (
            self.budget.model_calls != 1 or self.budget.tool_calls != 0
        ):
            raise ValueError("a forced response makes one model call and no tool call")
        if self.mode == "read_only_tools" and (
            self.budget.model_calls < 2 or self.budget.tool_calls < 1
        ):
            raise ValueError("a tool profile needs two model calls and one tool call at least")
        return self


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


def _schema(properties: JsonObject, required: Sequence[str]) -> JsonObject:
    """Return a closed object schema."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_INT: JsonObject = {"type": "integer", "minimum": 0}

EVIDENCE_TOOLS: tuple[ToolSchema, ...] = (
    ToolSchema(
        name="list_evidence",
        description="List supplied evidence locators that start with a prefix.",
        input_schema=_schema({"prefix": {"type": "string", "maxLength": 128}}, ("prefix",)),
    ),
    ToolSchema(
        name="read_trace_step",
        description="Read one exact tool turn (1-based) of one attempt's trajectory.",
        input_schema=_schema({"attempt": _INT, "step": _INT}, ("attempt", "step")),
    ),
    ToolSchema(
        name="read_checkpoint_prefix",
        description="Read actor-visible prefix messages at one offered point, by slice.",
        input_schema=_schema(
            {"point": _INT, "start": _INT, "count": {"type": "integer", "minimum": 1}},
            ("point", "start", "count"),
        ),
    ),
    ToolSchema(
        name="read_rule",
        description="Read one supplied rule-book entry by rule id.",
        input_schema=_schema({"rule_id": {"type": "string", "maxLength": 128}}, ("rule_id",)),
    ),
    ToolSchema(
        name="read_grader_evidence",
        description="Read one attempt's grade and the evaluator projection it was graded on.",
        input_schema=_schema({"attempt": _INT}, ("attempt",)),
    ),
    ToolSchema(
        name="read_evidence",
        description="Read any supplied evidence by its exact locator.",
        input_schema=_schema({"locator": {"type": "string", "maxLength": 256}}, ("locator",)),
    ),
)
"""The complete read-only tool set; nothing else is ever executed for a role."""

TOOLS_INSTRUCTION = """

You may read evidence before answering. The opening message is an overview: the group, the
verifier, attempt summaries, the checkpoint inventory, the evidence inventory (including material
that was not supplied) and every locator. The read-only tools fetch exact evidence by locator;
they cannot run actions, change any state or grade, or reach the actor. Your budget is frozen:
{budget}. When it runs low you must submit your answer with {answer}. Cite only listed locators."""


def tools_instruction(profile: EvidenceToolProfile, answer: str) -> str:
    """Return the addendum a role reads under the read-only tool profile."""
    return TOOLS_INSTRUCTION.format(
        budget=json.dumps(profile.budget.model_dump(mode="json"), sort_keys=True), answer=answer
    )


def execute_evidence_tool(
    documents: Mapping[str, JsonValue], call: ToolCall, *, maximum_bytes: int
) -> tuple[Literal["ok", "refused"], JsonValue]:
    """Answer one evidence-tool call from frozen documents, reading only.

    Args:
        documents: Exact content per locator.
        call: The role's tool call.
        maximum_bytes: Largest serialized result returned.

    Returns:
        ``ok`` and the result, or ``refused`` and the reason. An unknown tool, a malformed
        argument, an unknown locator or an oversized result is refused, never executed or cut.
    """
    arguments = call.arguments
    locator: str | None = None
    try:
        if call.name == "list_evidence":
            prefix = str(arguments["prefix"])
            found = [key for key in sorted(documents) if key.startswith(prefix)]
            return _bounded({"prefix": prefix, "locators": found}, maximum_bytes)
        if call.name == "read_trace_step":
            locator = f"attempt:{_integer(arguments['attempt'])}/step:{_integer(arguments['step'])}"
        elif call.name == "read_rule":
            locator = f"rule:{arguments['rule_id']}"
        elif call.name == "read_grader_evidence":
            locator = f"grade:{_integer(arguments['attempt'])}"
        elif call.name == "read_evidence":
            locator = str(arguments["locator"])
        elif call.name == "read_checkpoint_prefix":
            locator = f"point:{_integer(arguments['point'])}"
            point = documents.get(locator)
            prefix_rows = point.get("public_prefix") if isinstance(point, dict) else None
            if not isinstance(point, dict) or not isinstance(prefix_rows, list):
                return "refused", f"locator {locator} was not supplied"
            start, count = _integer(arguments["start"]), _integer(arguments["count"])
            if start < 0 or count < 1:
                return "refused", "start must be non-negative and count positive"
            return _bounded(
                {
                    "locator": locator,
                    "actor_calls_spent": point.get("actor_calls_spent"),
                    "messages_total": len(prefix_rows),
                    "start": start,
                    "messages": prefix_rows[start : start + count],
                },
                maximum_bytes,
            )
        else:
            return "refused", f"tool {call.name} is not a read-only evidence tool"
    except (KeyError, TypeError, ValueError):
        return "refused", f"malformed arguments for {call.name}"
    if locator not in documents:
        return "refused", f"locator {locator} was not supplied"
    return _bounded({"locator": locator, "content": documents[locator]}, maximum_bytes)


def _integer(value: JsonValue) -> int:
    """Return an integer argument, refusing any other JSON value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer argument")
    return value


def _bounded(value: JsonValue, maximum_bytes: int) -> tuple[Literal["ok", "refused"], JsonValue]:
    """Return a result, or a refusal naming its size when it is too large to return whole."""
    size = len(json.dumps(value, sort_keys=True).encode())
    if size > maximum_bytes:
        return "refused", f"result is {size} bytes, above the {maximum_bytes}-byte limit; narrow it"
    return "ok", value


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
    """

    index: int = Field(ge=0)
    response: ArtifactInput
    forced_answer: bool
    usage: MeasuredUsage


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
) -> tuple[ArtifactInput, AnalysisTranscript]:
    """Run, or replay, one bounded role analysis and persist its whole transcript.

    The caller holds the operation lock that the author-response owner requires.

    Args:
        project: Project whose artifacts hold the draw.
        settings: Frozen role model and per-call allowance.
        profile: Evidence-tool profile and budgets.
        instruction: Role system instruction.
        packet: Privileged evidence; its documents back the tools.
        content: Role-specific fields added to the opening message.
        answer: The typed answer tool.
        identity: Stable identity of this draw.
        sources: Saved artifacts the packet was rendered from.
        code_revision: Producing source identity.

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
        if saved.profile != profile or saved.answer_tool != answer.name:
            raise ValueError("saved role analysis was made under a different profile")
        return artifact_input(store.read(transcript_id).manifest), saved
    loop = _Loop(project, settings, profile, identity, sources, code_revision)
    if profile.mode == "forced_response":
        request = forced_request(
            instruction,
            {**packet.payload, **content},
            answer,
            output_tokens=settings.output_tokens,
        )
        transcript = loop.forced(request, answer)
    else:
        opening = {**(packet.overview or packet.payload), **content}
        transcript = loop.explore(
            instruction + tools_instruction(profile, answer.name), opening, packet, answer
        )
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
    ) -> None:
        """Bind the frozen draw settings."""
        self.project, self.settings, self.profile = project, settings, profile
        self.identity, self.sources, self.revision = identity, sources, code_revision
        self.catalog = load_model_catalog(project.model_catalog_path)
        self.calls: list[AnalysisCall] = []
        self.tools: list[ToolResultRecord] = []
        self.usage = MeasuredUsage()
        self.started = time.monotonic()

    def _call(self, request: ModelRequest, *, forced: bool) -> ModelResponse:
        """Buy or replay one model call and measure it."""
        index = len(self.calls)
        name = (
            self.identity
            if self.profile.mode == "forced_response"
            else (f"{self.identity}-call-{index}")
        )
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
            AnalysisCall(index=index, response=reference, forced_answer=forced, usage=usage)
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
            if spent > limit
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
        response = self._call(request, forced=True)
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

    def explore(
        self, instruction: str, opening: JsonObject, packet: EvidencePacket, answer: ToolSchema
    ) -> AnalysisTranscript:
        """Run the read-only tool profile until an answer or a budget stop."""
        budget = self.profile.budget
        messages = [
            ModelMessage(role="system", content=instruction),
            ModelMessage(role="user", content=json.dumps(opening, sort_keys=True)),
        ]
        tools = (*EVIDENCE_TOOLS, answer)
        executed = 0
        while True:
            probe = ModelRequest(
                messages=tuple(messages),
                tools=tools,
                tool_choice="required",
                maximum_output_tokens=self.settings.output_tokens,
            )
            short = self._short(probe, 1)
            if short:
                return self._finish(
                    answer.name,
                    "budget_exhausted",
                    exhausted=short,
                    failure="budget exhausted before an answer: " + ", ".join(short),
                )
            final = executed >= budget.tool_calls or bool(self._short(probe, 2))
            request = probe.model_copy(
                update={"tool_choice": ToolChoice(name=answer.name) if final else "required"}
            )
            response = self._call(request, forced=final)
            done = self._answered(response, answer.name)
            if done is not None:
                return done
            calls = response.output.tool_calls
            if final or not calls or response.finish_reason.value != "completed":
                return self._finish(
                    answer.name,
                    "protocol_failure",
                    failure="role made no complete tool call"
                    if not final
                    else f"role did not answer with {answer.name} when forced",
                )
            messages.append(ModelMessage(role="assistant", assistant_action=response.output))
            for position, call in enumerate(calls):
                if executed >= budget.tool_calls:
                    status, result = "refused", "evidence-tool budget exhausted; not executed"
                else:
                    executed += 1
                    status, result = execute_evidence_tool(
                        packet.documents, call, maximum_bytes=self.profile.maximum_result_bytes
                    )
                record = self._save_tool(position, call, status, result)
                self.tools.append(record)
                messages.append(
                    ModelMessage(
                        role="tool",
                        tool_call_id=call.call_id,
                        content=json.dumps({"status": status, "result": result}, sort_keys=True),
                    )
                )

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


def evidence_tools_sha256() -> str:
    """Return the identity of the tool set, its instruction and the loop protocol."""
    return sha256_json(
        {
            "protocol": EVIDENCE_TOOLS_PROTOCOL,
            "tools": [tool.model_dump(mode="json") for tool in EVIDENCE_TOOLS],
            "instruction": TOOLS_INSTRUCTION,
        }
    )
