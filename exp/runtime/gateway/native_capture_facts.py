"""Content-free facts the native collector hands beside each batched capture record.

With ``CaptureCollector.batched(..., bytes_output=True, facts_output=True)``
every record reaches the destination as ``("facts", facts, record)``: ``facts``
is the small JSON document modeled here, computed natively while the record
was encoded, and ``record`` is exactly the value plain bytes output would hand
(the record bytes, or a ``SplitRecord`` tuple). A destination reads identity,
completeness and response flags from the facts and frames ``record`` without
decoding it. A storage plan that keeps the prompt or the response out is
applied with :meth:`CaptureRecordFacts.planned`, a byte splice over the spans,
never a parse. The native module advertises the contract as
``exp_gateway_native.CAPTURE_FACTS_VERSION``.
"""

from __future__ import annotations

import json
from typing import Final, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.native_capture import CaptureScope

CAPTURE_FACTS_VERSION: Final = 1
CAPTURE_FACTS_TAG: Final = "facts"

ByteSpan = tuple[int, int]


class CaptureFactSpans(ContractModel):
    """Byte spans, ``[start, end)``, of the parts a storage plan may drop.

    Attributes:
        request_id: The request id's JSON string; checked before any splice so
            facts paired with another request's record are refused.
        messages: The value of ``request.context.request.messages``; replaced by
            ``[]`` when the prompt is kept out.
        source_json: The whole ``source_json`` member of the request context
            with one separating comma; removed with the prompt (together with
            ``provider_context``).
        provider_context: The whole ``provider_context`` member of the request
            context with one separating comma; removed with the prompt.
        wire_request: The whole ``wire_request`` member of ``transport`` with
            one separating comma; removed with the prompt. It lies inside
            ``transport``, so a plan that keeps the response out removes
            ``transport`` instead.
        response: The value of the record's ``response``; replaced by ``null``
            when the response is kept out.
        response_fields: Values of the response-derived fields present in the
            record (provider reasoning and its source, completed tool calls,
            metrics, the Gemini parts' source); replaced by ``null`` with the
            response, as the collector clears them itself.
        gemini_thought_parts: The non-empty Gemini thought parts value;
            replaced by ``[]`` with the response.
        provider_logprobs: The whole ``provider_logprobs`` member with its
            leading comma; removed with the response.
        transport: The whole ``transport`` member with its leading comma
            (caller-facing headers, timing and the wire request); removed with
            the response, since the collector delivers no relay evidence for a
            request whose response it may not keep.
    """

    request_id: ByteSpan
    messages: ByteSpan | None
    source_json: ByteSpan | None
    provider_context: ByteSpan | None
    wire_request: ByteSpan | None
    response: ByteSpan | None
    response_fields: tuple[ByteSpan, ...]
    gemini_thought_parts: ByteSpan | None
    provider_logprobs: ByteSpan | None
    transport: ByteSpan | None


class CaptureResponseFacts(ContractModel):
    """Content-free flags of a record's response.

    Attributes:
        kind: Whether the response is one JSON body or a list of SSE frames.
        status: Observed HTTP response status.
        truncated: Whether capture limits excluded part of a stream.
        client_disconnected: Whether the consumer stopped before body completion.
    """

    kind: Literal["json", "sse"]
    status: int = Field(ge=100, le=599)
    truncated: bool
    client_disconnected: bool


class CaptureRecordFacts(ContractModel):
    """The facts document of one record; it never carries prompt or response content.

    Attributes:
        facts_version: Facts contract version, always ``CAPTURE_FACTS_VERSION``.
        schema_version: The record's schema: 1 for a whole record, 2 for a
            completion update that references an already delivered prompt.
        request_id: Authenticated request identifier.
        scope: Authority-derived organization, identity and application binding.
        protocol: Public HTTP request surface.
        model_id: Selected model identifier, when routing selected one.
        deployment_id: Selected provider deployment identifier, when available.
        captured_at: Admission timestamp as Unix seconds.
        complete: Whether the record carries the request's response.
        response: Response flags when ``complete``, otherwise None.
        split: Whether ``record`` is a split skeleton with parts on disk.
        record_bytes: Length of the record bytes (or split skeleton) handed beside.
        spans: Storage-plan spans over a whole record; None for a split skeleton.
    """

    facts_version: Literal[1]
    schema_version: Literal[1, 2]
    request_id: str = Field(min_length=1, max_length=512)
    scope: CaptureScope
    protocol: Literal["chat_completions", "responses", "messages", "generate_content"]
    model_id: str | None
    deployment_id: str | None
    captured_at: float = Field(ge=0, allow_inf_nan=False)
    complete: bool
    response: CaptureResponseFacts | None
    split: bool
    record_bytes: int = Field(ge=0)
    spans: CaptureFactSpans | None

    @model_validator(mode="after")
    def _consistent(self) -> CaptureRecordFacts:
        """A split skeleton has no spans; complete means exactly that response facts exist."""
        if self.split == (self.spans is not None):
            raise ValueError("a whole record has spans and a split skeleton has none")
        if self.complete != (self.response is not None):
            raise ValueError("a record is complete exactly when it carries response facts")
        return self

    def planned(self, record: bytes, *, keep_prompt: bool, keep_response: bool) -> bytes:
        """Return ``record`` with the parts a storage plan drops removed, without parsing it.

        Keeping both returns ``record`` itself. Dropping the prompt empties the
        conversation (the messages) and removes every verbatim copy of the
        caller's request (``source_json``, ``provider_context`` and the wire
        request); the other effective request fields (model, parameters, tool
        declarations, metadata) stay, so a destination that must not keep those
        either keeps no record, as the collector itself does when prompt
        retention is denied. Dropping the response nulls it and clears every
        response-derived field and the relay transport exactly as the
        collector does when response retention is denied. The result is the
        same JSON a decoded rewrite would produce.

        Args:
            record: The record bytes handed beside these facts.
            keep_prompt: Whether the prompt may be stored.
            keep_response: Whether the response may be stored.

        Returns:
            The record bytes the plan permits.

        Raises:
            ValueError: The facts describe a split skeleton, or ``record`` is
                not the record they describe (its length, or the request id
                at its span, differs).
        """
        if keep_prompt and keep_response:
            return record
        if self.spans is None:
            raise ValueError("a split skeleton carries no storage-plan spans")
        spans = self.spans
        # Constant-time binding: the length and the request id where the facts put it.
        start, end = spans.request_id
        request_id = json.dumps(self.request_id, ensure_ascii=False).encode()
        if len(record) != self.record_bytes or record[start:end] != request_id:
            raise ValueError("record bytes do not match their facts")
        edits: list[tuple[ByteSpan, bytes]] = []
        if not keep_prompt:
            if spans.messages is not None:
                edits.append((spans.messages, b"[]"))
            # The wire request goes with the whole transport when the response does.
            wire_request = spans.wire_request if keep_response else None
            edits.extend(
                (span, b"")
                for span in (spans.source_json, spans.provider_context, wire_request)
                if span is not None
            )
        if not keep_response:
            if spans.response is not None:
                edits.append((spans.response, b"null"))
            edits.extend((span, b"null") for span in spans.response_fields)
            if spans.gemini_thought_parts is not None:
                edits.append((spans.gemini_thought_parts, b"[]"))
            if spans.provider_logprobs is not None:
                edits.append((spans.provider_logprobs, b""))
            if spans.transport is not None:
                edits.append((spans.transport, b""))
        edits.sort()
        view = memoryview(record)
        pieces: list[bytes | memoryview] = []
        cursor = 0
        for (start, end), replacement in edits:
            if not cursor <= start <= end <= len(record):
                raise ValueError("capture facts spans are out of order or out of range")
            pieces.extend((view[cursor:start], replacement))
            cursor = end
        pieces.append(view[cursor:])
        return b"".join(pieces)
