"""Provider-neutral stream outcome contracts: usage, events, and failures.

Split from :mod:`exp.runtime.gateway.contracts` for the module line budget;
that module re-exports every name here, so import paths are unchanged.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import MAXIMUM_TOOL_CALL_ID_CHARACTERS, ToolCall

ByteValue = Annotated[int, Field(strict=True, ge=0, le=255)]
"""One byte from a provider token representation."""

LogProbability = Annotated[float, Field(strict=True, allow_inf_nan=False)]
"""One finite natural-log probability."""


class LogprobCandidate(ContractModel):
    """One alternate token and its provider probability.

    Attributes:
        token: Provider token text, never normalized.
        logprob: Finite natural-log probability.
        bytes: Provider bytes, or None when omitted or null.
    """

    token: str
    logprob: LogProbability
    bytes: tuple[ByteValue, ...] | None = None


class TokenLogprob(LogprobCandidate):
    """The selected token probability and bounded alternate candidates.

    Attributes:
        top_logprobs: Ordered alternatives, including an explicitly empty tuple.
    """

    top_logprobs: tuple[LogprobCandidate, ...] = Field(max_length=20)


class ChoiceLogprobs(ContractModel):
    """Probability records for one Chat choice.

    Attributes:
        content: Content records; None differs from an observed empty sequence.
        refusal: Refusal records; None differs from an observed empty sequence.
    """

    content: tuple[TokenLogprob, ...] | None = None
    refusal: tuple[TokenLogprob, ...] | None = None


class ChoiceLogprobsDelta(ContractModel):
    """One ordered probability observation scoped to a Chat choice.

    Attributes:
        choice_index: The provider choice index.
        logprobs: Observed channels, or None for a null observation.
    """

    choice_index: int = Field(strict=True, ge=0, le=2**32 - 1)
    logprobs: ChoiceLogprobs | None


class GatewayUsage(ContractModel):
    """Normalized token counts and invoked tool names from one provider attempt.

    Cached-input, cache-write, and reasoning counts are disjoint subsets of
    the total input and output counts when present. They identify differently
    priced portions of those totals and must not be added a second time by
    callers. ``cache_creation_input_tokens`` is the provider cache-write
    surcharge leg; it is disjoint from ``cached_input_tokens`` inside
    ``input_tokens``, so ``fresh = input - cached - cache_creation``.

    A terminal event may carry partial token totals or only ``tool_names``.
    Missing totals remain unknown, never zero. Live usage events require both
    totals; terminal accounting retains an observed leg without pricing the
    missing leg or treating a partial report as a final meter.

    ``web_search_requests`` and ``tool_search_requests`` ride along with either shape but never
    make usage on their own: a count with neither token totals nor tool names is still rejected.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    """Cache-write tokens inside the input total (Anthropic and Bedrock),
    disjoint from the cache-read leg; present only when the provider reported
    a nonzero count."""
    cache_creation_1h_input_tokens: int | None = Field(default=None, ge=0)
    """Observed 1-hour subset of cache writes; zero proves all writes use 5m.
    None means no complete TTL breakdown was reported, so write cost is unknown."""
    reasoning_tokens: int | None = Field(default=None, ge=0)
    tool_names: tuple[str, ...] = ()
    """Invoked tool names in first-use order, names only and never arguments."""
    web_search_requests: int = Field(default=0, ge=0)
    """Gateway-executed web searches billed to this attempt; never a provider
    meter and not a subset of any token total. Zero on every attempt that ran
    no search, including every attempt settled by an engine predating it."""
    tool_search_requests: int = Field(default=0, ge=0)
    """Gateway-executed tool-search rounds billed to this attempt; never a
    provider meter and not a subset of any token total. Zero on every attempt
    that ran no tool search, including every attempt settled by an engine
    predating it."""

    @model_validator(mode="after")
    def _require_complete_tokens_or_tool_names(self) -> GatewayUsage:
        """Require observed tokens or tools and validate available covering totals.

        Returns:
            This validated token or tool-only usage record.

        Raises:
            ValueError: No token or tool was observed, or a subset contradicts its total.
        """
        totals = (self.input_tokens, self.output_tokens)
        if totals == (None, None) and not self.tool_names:
            raise ValueError("usage requires token totals or invoked tool names")
        if self.cache_creation_1h_input_tokens is not None and (
            self.cache_creation_input_tokens is None
            or self.cache_creation_1h_input_tokens > self.cache_creation_input_tokens
        ):
            raise ValueError("1-hour cache writes require a covering cache-creation total")
        return self

    @property
    def has_token_counts(self) -> bool:
        """Return whether both provider token totals are known."""
        return self.input_tokens is not None and self.output_tokens is not None


class GatewayEventKind(StrEnum):
    """Provider-neutral semantic and terminal stream event categories."""

    TEXT_DELTA = "text_delta"
    REFUSAL_DELTA = "refusal_delta"
    CHOICE_LOGPROBS_DELTA = "choice_logprobs_delta"
    PROVIDER_RESPONSES_LOGPROBS = "provider_responses_logprobs"
    REASONING_SUMMARY_DELTA = "reasoning_summary_delta"
    THINKING_DELTA = "thinking_delta"
    THINKING_SIGNATURE = "thinking_signature"
    REDACTED_THINKING = "redacted_thinking"
    ENCRYPTED_REASONING = "encrypted_reasoning"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_ARGUMENTS_DELTA = "tool_arguments_delta"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    USAGE = "usage"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class GatewayEvent(ContractModel):
    """One ordered provider-neutral stream event, including raw tool fragments.

    Attributes:
        choice_logprobs_delta: Optional typed Chat token probabilities for this event.
        responses_output_index: Optional nonnegative Responses output-item index.
        responses_item_id: Optional nonempty Responses item identity, at most 256 characters.
        responses_content_index: Optional nonnegative index within one output item.
        responses_logprobs_phase: Optional delta or completion phase for native probability records.
        responses_logprobs_records: Optional ordered native probability objects, preserving content.
    """

    kind: GatewayEventKind
    sequence_number: int = Field(ge=0)
    text_delta: str | None = None
    reasoning_summary_output_index: int | None = Field(default=None, ge=0)
    reasoning_summary_index: int | None = Field(default=None, ge=0)
    reasoning_item_id: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_block_index: int | None = Field(default=None, ge=0)
    """Provider content-block (or output-item) index grouping reasoning events."""
    thinking_signature: str | None = None
    redacted_thinking_data: str | None = None
    encrypted_content: str | None = None
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = Field(
        default=None, min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS
    )
    tool_name: str | None = Field(default=None, min_length=1, max_length=256)
    raw_arguments_delta: str | None = None
    tool_call: ToolCall | None = None
    usage: GatewayUsage | None = None
    failure: GatewayFailure | None = None
    choice_logprobs_delta: ChoiceLogprobsDelta | None = None
    usage_incomplete_due_to_disconnect: bool = Field(default=False, exclude=True, strict=True)
    """Trusted evidence of caller loss after dispatch but before an observed provider terminal.

    Observed usage remains evidence, not a claim that the provider's bill is
    complete. Hosted ledgers retain unresolved exposure for later resolution;
    the local monthly budget uses the full reserved bound as an unknown-cost
    estimate. Neither policy treats the partial meter as a final provider bill.
    This marker never joins public serialized events or replay identity.
    """
    usage_estimated: bool = Field(default=False, exclude=True, strict=True)
    """The usage on a cancelled disconnect is the gateway's own tokenizer estimate.

    Set only by the accounting registry after a dispatched, opened attempt lost
    its caller before the provider's final meter: the prompt count and the
    generated text observed so far replace the legs the provider never
    reported, so the work the provider billed settles at its estimated cost
    instead of an unknown one. Observed legs are kept; never a public field.
    """
    decision_provider_rejected: bool = Field(default=False, exclude=True, strict=True)
    """Internal decision settlement evidence that an HTTP rejection preceded execution.

    False leaves unmetered decision work financially unresolved. This is not
    provider token usage and never joins serialized events or replay identity.
    """

    responses_output_index: int | None = Field(default=None, ge=0, alias="output_index")
    responses_item_id: str | None = Field(
        default=None, min_length=1, max_length=256, alias="item_id"
    )
    responses_content_index: int | None = Field(default=None, ge=0, alias="content_index")
    responses_logprobs_phase: (
        Literal["delta", "text_done", "content_part_done", "item_done", "terminal"] | None
    ) = Field(default=None, alias="phase")
    responses_logprobs_records: tuple[JsonObject, ...] | None = Field(default=None, alias="records")

    @model_validator(mode="after")
    def _require_event_payload(self) -> GatewayEvent:
        """Require each event kind to carry its one relevant payload.

        Returns:
            The validated stream event.

        Raises:
            ValueError: The selected event kind lacks its required payload.
        """
        if self.usage_incomplete_due_to_disconnect and (
            self.kind is not GatewayEventKind.FAILED
            or self.failure is None
            or self.failure.failure_class is not GatewayFailureClass.CANCELLED
        ):
            raise ValueError("incomplete disconnect usage requires a cancelled terminal")
        if self.usage_estimated and (
            not self.usage_incomplete_due_to_disconnect
            or self.usage is None
            or not self.usage.has_token_counts
        ):
            raise ValueError("estimated usage requires a disconnect with both token totals")
        if self.kind in {GatewayEventKind.TEXT_DELTA, GatewayEventKind.REFUSAL_DELTA}:
            if self.text_delta is None:
                raise ValueError("text and refusal deltas require text_delta")
        elif self.kind == GatewayEventKind.CHOICE_LOGPROBS_DELTA:
            if self.choice_logprobs_delta is None:
                raise ValueError("choice logprobs deltas require their payload")
        elif self.kind == GatewayEventKind.PROVIDER_RESPONSES_LOGPROBS:
            if (
                self.responses_output_index is None
                or self.responses_item_id is None
                or self.responses_content_index is None
                or self.responses_logprobs_phase is None
                or "responses_logprobs_records" not in self.model_fields_set
            ):
                raise ValueError(
                    "Responses probability events require identity, phase, and records"
                )
            if self.responses_logprobs_records is None and self.responses_logprobs_phase in {
                "delta",
                "text_done",
            }:
                raise ValueError("Responses delta and text-done probabilities require arrays")
            _validate_responses_probability_records(
                self.responses_logprobs_records, self.responses_logprobs_phase
            )
        elif self.kind == GatewayEventKind.REASONING_SUMMARY_DELTA:
            if (
                self.text_delta is None
                or self.reasoning_summary_output_index is None
                or self.reasoning_summary_index is None
                or self.reasoning_item_id is None
            ):
                raise ValueError("reasoning summary deltas require item, output, summary, and text")
        elif self.kind == GatewayEventKind.THINKING_DELTA:
            if self.text_delta is None or self.reasoning_block_index is None:
                raise ValueError("thinking deltas require block index and text")
        elif self.kind == GatewayEventKind.THINKING_SIGNATURE:
            if self.thinking_signature is None or self.reasoning_block_index is None:
                raise ValueError("thinking signatures require block index and signature")
        elif self.kind == GatewayEventKind.REDACTED_THINKING:
            if self.redacted_thinking_data is None or self.reasoning_block_index is None:
                raise ValueError("redacted thinking requires block index and data")
        elif self.kind == GatewayEventKind.ENCRYPTED_REASONING:
            if (
                self.encrypted_content is None
                or self.reasoning_block_index is None
                or self.reasoning_item_id is None
            ):
                raise ValueError("encrypted reasoning requires item, block index, and content")
        elif self.kind == GatewayEventKind.TOOL_CALL_STARTED:
            if self.tool_call_index is None or self.tool_call_id is None or self.tool_name is None:
                raise ValueError("tool-call start requires index, ID, and name")
        elif self.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            if self.tool_call_index is None or self.raw_arguments_delta is None:
                raise ValueError("tool argument delta requires index and raw fragment")
        elif self.kind == GatewayEventKind.TOOL_CALL_COMPLETED and self.tool_call is None:
            raise ValueError("tool-call completion requires the complete tool call")
        elif self.kind == GatewayEventKind.USAGE:
            if self.usage is None or not self.usage.has_token_counts:
                raise ValueError("usage event requires complete normalized token usage")
        elif self.kind == GatewayEventKind.FAILED and self.failure is None:
            raise ValueError("failed event requires a normalized failure")
        return self


class GatewayFailureClass(StrEnum):
    """Stable failure classes shared by provider execution and the public protocol."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    QUOTA_EXCEEDED = "quota_exceeded"
    THROTTLED = "throttled"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    PROVIDER_AUTHENTICATION = "provider_authentication"
    PROVIDER_NOT_FOUND = "provider_not_found"
    # The provider ACCOUNT cannot pay for the request (trial quota exhausted,
    # billing not enabled): operator-actionable deadness that fails over in
    # every mode. Distinct from QUOTA_EXCEEDED, the CALLER's gateway credit.
    PROVIDER_QUOTA = "provider_quota"
    REFUSAL = "refusal"
    # The provider closed the turn as complete and delivered nothing the caller
    # can receive (an OpenAI empty assistant message; a reasoning-only turn on a
    # rung whose reasoning the gateway strips). The model's answer to the
    # request content, like REFUSAL: never a deployment-circuit failure, and a
    # 400 the SDKs do not auto-retry.
    EMPTY_COMPLETION = "empty_completion"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_INTERNAL = "provider_internal"
    CANCELLED = "cancelled"
    GUARDRAIL = "guardrail"
    INTERNAL = "internal"
    # A transient control-plane condition (a rolling deploy building the
    # authorized catalog revision) that the caller should simply retry. Unlike
    # INTERNAL it is not a bug signal and does not page; unlike a provider class
    # it never opens a deployment circuit.
    UNAVAILABLE = "unavailable"


class GatewayRefusalReason(StrEnum):
    """The bounded category of a provider refusal, mirroring the native
    ``RefusalReason``.

    A refusal answer names WHICH policy declined the content as a closed
    vocabulary, so a client can branch on it and the control plane can count
    refusals by reason without parsing the free-form provider detail. The
    caller never sees the provider's own prose, only the fixed category.
    """

    CYBER_POLICY = "cyber_policy"
    CBRN = "cbrn"
    CONTENT_POLICY = "content_policy"
    RECITATION = "recitation"
    DATA_INSPECTION = "data_inspection"
    UNSPECIFIED = "unspecified"


class GatewayFailure(ContractModel):
    """Sanitized failure with retry and failover eligibility already classified."""

    failure_class: GatewayFailureClass
    safe_message: str = Field(min_length=1, max_length=2_048)
    retryable_same_deployment: bool = False
    failover_eligible: bool = False
    safe_details: JsonObject = Field(default_factory=dict)
    rejected_parameter: str | None = Field(default=None, min_length=1, max_length=128)
    """Validated provider-named parameter path; never provider prose."""
    provider_detail: str | None = Field(default=None, min_length=1, max_length=240)
    """Provider explanation of a client error, relayed only for that class."""
    retry_after_seconds: int | None = Field(default=None, ge=1)
    """The failure is the caller's own provider configuration: a rejected
    credential or exhausted account on their customer-managed (BYOK) rung. The
    class keeps its ladder semantics; the ledger files it as the caller's
    invalid request and the terminal answer is their 400."""
    customer_owned: bool = False
    """Known wait before a retry can dispatch (a throttle window's remainder).

    When present on a throttled failure, the public mapping advertises this
    value as ``Retry-After`` instead of its fixed default, so the header and
    the message never tell the caller two different waits.
    """
    refusal_reason: GatewayRefusalReason | None = None
    """The bounded refusal category, present only on a ``REFUSAL`` failure.

    Set from the provider's own code and sentence and carried on the public
    error and the settlement argument, so the caller reads the category and
    the control plane counts refusals by reason without parsing detail."""


def _validate_responses_probability_records(
    records: tuple[JsonObject, ...] | None, phase: str
) -> None:
    """Validate phase-shaped provider observations without filling optional fields."""
    if records is None:
        return
    optional_alternatives = phase in {"delta", "text_done"}
    for record in records:
        LogprobCandidate.model_validate(
            {key: record[key] for key in ("token", "logprob", "bytes") if key in record}
        )
        if "top_logprobs" not in record:
            continue
        alternatives = record["top_logprobs"]
        if alternatives is None and optional_alternatives:
            continue
        candidates = TypeAdapter(tuple[JsonObject, ...]).validate_python(alternatives)
        if len(candidates) > 20:
            raise ValueError("Responses probabilities allow at most 20 alternatives")
        for candidate in candidates:
            if optional_alternatives:
                if candidate.get("token") is not None and not isinstance(candidate["token"], str):
                    raise ValueError("Responses alternative token must be text or null")
                if candidate.get("logprob") is not None:
                    TypeAdapter(LogProbability).validate_python(candidate["logprob"])
                if "bytes" in candidate:
                    TypeAdapter(tuple[ByteValue, ...]).validate_python(candidate["bytes"])
            else:
                LogprobCandidate.model_validate(candidate)
