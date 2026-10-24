"""Canonical normalized production-trace contracts."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    Sha256,
    SourceIdentity,
    StructuredFailure,
    validate_artifact_file_path,
)
from exp.common.models import ModelSnapshot, Usage
from exp.common.tasks import ToolSchema
from exp.common.tasks.facts import INSTANCE_FACTS_CONTEXT_KEY

INPUT_MESSAGES_KEY = "gen_ai.input.messages"
"""Span attribute holding one model call's input messages inline."""

INPUT_HISTORY_PREFIX_KEY = "exp.input.history_prefix"
"""Span attribute naming how many leading ``Trace.message_history`` entries form its input."""


class TraceSource(ContractModel):
    """Origin and semantic-convention version for a normalized production trace."""

    identity: SourceIdentity
    semantic_convention_version: str = Field(min_length=1, max_length=128)


class TraceOutcome(ContractModel):
    """Captured terminal outcome for one production trace."""

    status: Literal["success", "failure", "abandoned", "unknown"]
    outcome_name: str | None = Field(default=None, max_length=256)
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_failure_for_failed_outcome(self) -> TraceOutcome:
        if self.status == "failure" and self.failure is None:
            raise ValueError("failed trace outcomes require a structured failure")
        return self


class TraceSpan(ContractModel):
    """One ordered OpenTelemetry-style event in a normalized production trace."""

    span_id: str = Field(min_length=1, max_length=256)
    parent_span_id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    started_at: AwareDatetime
    ended_at: AwareDatetime
    attributes: JsonObject = Field(default_factory=dict)
    model: ModelSnapshot | None = None
    usage: Usage | None = None
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_ordered_timestamps(self) -> TraceSpan:
        if self.ended_at < self.started_at:
            raise ValueError("trace span ended_at cannot be before started_at")
        return self


class Trace(ContractModel):
    """One normalized customer agent trace with source provenance.

    Attributes:
        trace_id: Source trace identity.
        conversation_id: Optional conversation identity.
        task: Initial request text.
        initial_context: Request-visible context. A request context's ``instance_facts``
            entry is moved to ``instance_facts`` on construction, so instance facts never
            become request-visible context, retrieval grounding, mined task context, or
            memory of other traces.
        tools: Declared tools.
        spans: Ordered normalized spans.
        outcome: Captured terminal outcome, if any.
        source: Source provenance.
        instance_facts: Raw instance facts the trace package supplied for this trace's
            instance; validated by the consumer before use. Omitted when absent.
        message_history: One conversation's messages, stored once. A model span whose input
            is a prefix of this conversation names the prefix length in
            ``exp.input.history_prefix`` instead of copying the messages into
            ``gen_ai.input.messages``, so stored size grows linearly with transcript length.
            The longest referenced prefix is the whole history. Omitted when empty; read a
            span's input through :meth:`input_messages`.
    """

    trace_id: str = Field(min_length=1, max_length=512)
    conversation_id: str | None = Field(default=None, max_length=512)
    task: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    tools: tuple[ToolSchema, ...] = ()
    spans: tuple[TraceSpan, ...]
    outcome: TraceOutcome | None = None
    source: TraceSource
    instance_facts: JsonObject | None = Field(default=None, exclude_if=lambda value: value is None)
    message_history: tuple[JsonObject, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="before")
    @classmethod
    def _separate_instance_facts(cls, value: object) -> object:
        """Move a request context's instance facts into their own channel."""
        if not isinstance(value, dict):
            return value
        context: object = value.get("initial_context")
        if not isinstance(context, dict):
            return value
        context = cast(JsonObject, context)
        if INSTANCE_FACTS_CONTEXT_KEY not in context:
            return value
        if value.get("instance_facts") is not None:
            raise ValueError(
                "a trace supplies instance facts both in its request context and separately; "
                "supply them once"
            )
        facts = context[INSTANCE_FACTS_CONTEXT_KEY]
        if not isinstance(facts, dict):
            raise ValueError(
                f"request context entry {INSTANCE_FACTS_CONTEXT_KEY!r} must be a JSON object of "
                "instance facts"
            )
        remaining = {
            key: item for key, item in context.items() if key != INSTANCE_FACTS_CONTEXT_KEY
        }
        return {**value, "initial_context": remaining, "instance_facts": facts}

    @field_validator("spans")
    @classmethod
    def _require_unique_span_ids(cls, value: tuple[TraceSpan, ...]) -> tuple[TraceSpan, ...]:
        if not value:
            raise ValueError("a trace must contain at least one span")
        span_ids = tuple(span.span_id for span in value)
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("trace span IDs must be unique")
        return value

    @model_validator(mode="after")
    def _require_resolvable_history_prefixes(self) -> Trace:
        """Every history reference resolves, and the history holds nothing unreferenced."""
        longest = 0
        for span in self.spans:
            if INPUT_HISTORY_PREFIX_KEY not in span.attributes:
                continue
            prefix = span.attributes[INPUT_HISTORY_PREFIX_KEY]
            if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
                raise ValueError(f"span {span.span_id!r} history prefix must be a count")
            if prefix > len(self.message_history):
                raise ValueError(
                    f"span {span.span_id!r} references {prefix} history messages but the "
                    f"trace stores {len(self.message_history)}"
                )
            if INPUT_MESSAGES_KEY in span.attributes:
                raise ValueError(
                    f"span {span.span_id!r} declares its input both inline and as a history prefix"
                )
            longest = max(longest, prefix)
        if longest != len(self.message_history):
            raise ValueError(
                f"trace stores {len(self.message_history)} history messages but its spans "
                f"reference only {longest}"
            )
        return self

    def input_messages(self, span: TraceSpan) -> JsonValue | None:
        """Return one span's input messages, resolving a shared-history reference.

        Args:
            span: A span of this trace.

        Returns:
            The referenced history prefix as a list, the span's inline
            ``gen_ai.input.messages`` value as declared (possibly JSON-encoded text), or
            ``None`` when the span declares neither.
        """
        prefix = span.attributes.get(INPUT_HISTORY_PREFIX_KEY)
        if isinstance(prefix, int) and not isinstance(prefix, bool):
            return list(self.message_history[:prefix])
        return span.attributes.get(INPUT_MESSAGES_KEY)


class TraceDataset(ArtifactEnvelope):
    """A frozen normalized trace-dataset manifest with auditable exclusions."""

    schema_version: Literal[1, 2] = 2
    dataset_id: ArtifactId
    semantic_convention_version: str = Field(min_length=1, max_length=128)
    traces_path: str = Field(min_length=1)
    traces_sha256: Sha256
    issues_path: str | None = None
    issues_sha256: Sha256 | None = None
    invalid_trace_count: int = Field(default=0, ge=0)
    trace_ids: tuple[str, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_integer_schema_version(cls, value: object) -> object:
        """Reject boolean and floating-point lookalikes at the version boundary."""
        if type(value) is not int:
            raise ValueError("trace dataset schema_version must be an integer")
        return value

    @field_validator("traces_path")
    @classmethod
    def _require_safe_traces_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("issues_path")
    @classmethod
    def _require_safe_issues_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_artifact_file_path(value).as_posix()

    @field_validator("trace_ids")
    @classmethod
    def _require_unique_trace_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a trace dataset must contain at least one trace")
        if len(set(value)) != len(value):
            raise ValueError("trace_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _require_complete_issue_reference(self) -> TraceDataset:
        if (self.issues_path is None) != (self.issues_sha256 is None):
            raise ValueError("trace-dataset issues path and digest must be set together")
        if self.invalid_trace_count and self.issues_path is None:
            raise ValueError("invalid trace count requires an immutable issues report")
        return self
