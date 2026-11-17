"""Versioned exposure fingerprints of recorded trace content.

A scheme's identity is its name, version and the SHA-256 of its canonical specification
(:data:`TRACE_CONTENT_SPECIFICATION` for the current scheme). The specification is
authoritative: any change to its normalization semantics needs a new version, even when every
golden vector still passes, because vectors cannot cover every input. Golden vectors ship with
the scheme as a conformance check; a scheme whose vectors do not reproduce exactly is not
computable on this machine and is never used to compute or match.

The current scheme hashes an include-list only: the task instruction and the order-insensitive
multiset of paired exchanges ``(tool name, canonical arguments, paired recorded reply)`` over
the whole recorded trace, public prefix included, keeping duplicate counts. Calls and results
are paired by call ID before the IDs are dropped, never by position, so concurrent results that
finish out of order pair correctly; a missing, duplicate or colliding call ID makes the trace
unresolved (no fingerprint). Being order-insensitive, it can over-match two different execution
sequences of the same exchanges: conservative exposure matching, which errs in the safe
direction.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cached_property

from pydantic import Field, JsonValue

from exp.common.core.artifacts import ContractModel, SourceIdentity, sha256_json
from exp.common.traces import Trace, TraceSource, TraceSpan

CALL_ID = "gen_ai.tool.call.id"
TOOL_NAME = "gen_ai.tool.name"
OPERATION = "gen_ai.operation.name"
RESULT_OPERATION = "execute_tool"

TRACE_CONTENT_SPECIFICATION: dict[str, JsonValue] = {
    "name": "trace-content",
    "version": 1,
    "include": [
        "task: Trace.task, verbatim",
        "exchanges: every recorded tool call of the whole trace, public prefix included, as "
        "[tool name, canonical arguments, paired recorded reply]",
    ],
    "excluded_by_construction": [
        "trace, span and tool-call IDs",
        "span timestamps and their precision",
        "import, dataset and source identities",
        "recording-agent metadata and OTEL resource attributes",
        "request-context keys, including task_lineage and replay_from",
        "assistant and user messages other than the task",
    ],
    "call": f"a span with a non-empty string {TOOL_NAME} whose {OPERATION} is not "
    f"{RESULT_OPERATION}",
    "result": f"a span whose {OPERATION} is {RESULT_OPERATION}",
    "pairing": f"by {CALL_ID} only, never by position: every call carries a non-empty string ID "
    "unique among calls; every result carries the ID of exactly one call; a call has at most "
    "one result; any violation makes the trace unresolved, with no fingerprint",
    "tool_name": f"{TOOL_NAME}, verbatim",
    "arguments": "gen_ai.tool.call.arguments; a string that parses as JSON is replaced by the "
    "parsed value, other strings are kept verbatim; absent arguments are {}",
    "reply": "gen_ai.tool.message of the paired result; a string that parses as JSON is "
    "replaced by the parsed value, other strings are kept verbatim; a call without a result "
    "has reply null",
    "unicode": "no normalization; strings compare by code point",
    "numbers": "as parsed by Python json: integers exact, floats by shortest repr",
    "canonical_json": "keys sorted, separators ',' and ':', non-ASCII kept",
    "multiset": "each exchange becomes the canonical JSON text of [name, arguments, reply]; "
    "the texts are sorted, duplicates kept",
    "hash": "SHA-256 hex of the canonical JSON of {scheme, version, task, exchanges}",
}


class SchemeIdentity(ContractModel):
    """Full identity of a fingerprint scheme.

    Attributes:
        name: Scheme name.
        version: Scheme version; bumped for any change to the specification's semantics.
        definition_sha256: SHA-256 of the canonical specification.
        vectors_sha256: SHA-256 of the golden vector set.
    """

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vectors_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def key(self) -> str:
        """Short stable label used in ledger vertex keys."""
        return f"{self.name}@v{self.version}#{self.definition_sha256[:12]}"

    def conflicts(self, other: "SchemeIdentity") -> bool:
        """Same name and version, but a different definition or vector set."""
        return (self.name, self.version) == (other.name, other.version) and self != other


class GoldenVector(ContractModel):
    """One sample trace with its expected fingerprint (None for an unresolved trace).

    Attributes:
        trace: Sample trace.
        fingerprint: Expected fingerprint.
    """

    trace: Trace
    fingerprint: str | None


@dataclass(frozen=True)
class Scheme:
    """A scheme implementation bound to its specification and golden vectors.

    Attributes:
        specification: Authoritative canonical specification.
        compute: Fingerprint of one trace, or None when it is unresolved.
        vectors: Golden vectors that must reproduce before the scheme computes or matches.
    """

    specification: dict[str, JsonValue]
    compute: Callable[[Trace], str | None] = field(compare=False)
    vectors: tuple[GoldenVector, ...]

    @cached_property
    def identity(self) -> SchemeIdentity:
        """Name, version, definition digest and vector-set digest."""
        return SchemeIdentity(
            name=str(self.specification["name"]),
            version=int(str(self.specification["version"])),
            definition_sha256=sha256_json(self.specification),
            vectors_sha256=sha256_json([vector.model_dump(mode="json") for vector in self.vectors]),
        )

    @cached_property
    def computable(self) -> bool:
        """Whether every golden vector reproduces exactly on this machine."""
        try:
            return all(self.compute(vector.trace) == vector.fingerprint for vector in self.vectors)
        except Exception:  # noqa: BLE001 - a crashing implementation is not computable.
            return False


def trace_content_fingerprint(trace: Trace) -> str | None:
    """Fingerprint under ``trace-content`` v1, or None when call pairing is ambiguous.

    Args:
        trace: One normalized trace.

    Returns:
        The SHA-256 hex digest, or None for an unresolved trace.
    """
    exchanges = paired_exchanges(trace)
    if exchanges is None:
        return None
    return sha256_json(
        {
            "scheme": TRACE_CONTENT_SPECIFICATION["name"],
            "version": TRACE_CONTENT_SPECIFICATION["version"],
            "task": trace.task,
            "exchanges": sorted(_canonical(list(exchange)) for exchange in exchanges),
        }
    )


def paired_exchanges(trace: Trace) -> list[tuple[str, JsonValue, JsonValue]] | None:
    """Every recorded call with its result paired by call ID, or None when ambiguous.

    Args:
        trace: One normalized trace.

    Returns:
        ``(tool name, arguments, reply)`` per call in no particular order, or None when a call
        ID is missing, repeated among calls, or a result does not name exactly one call.
    """
    calls: dict[str, tuple[str, JsonValue]] = {}
    results: dict[str, JsonValue] = {}
    for span in trace.spans:
        attributes = span.attributes
        call_id = attributes.get(CALL_ID)
        if attributes.get(OPERATION) == RESULT_OPERATION:
            if not isinstance(call_id, str) or not call_id or call_id in results:
                return None
            results[call_id] = _parsed(attributes.get("gen_ai.tool.message"))
            continue
        name = attributes.get(TOOL_NAME)
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(call_id, str) or not call_id or call_id in calls:
            return None
        arguments = attributes.get("gen_ai.tool.call.arguments")
        calls[call_id] = (name, {} if arguments is None else _parsed(arguments))
    if set(results) - set(calls):
        return None
    return [(name, arguments, results.get(call_id)) for call_id, (name, arguments) in calls.items()]


def _parsed(value: JsonValue) -> JsonValue:
    """A JSON string's parsed value; any other value unchanged."""
    if not isinstance(value, str):
        return value
    try:
        parsed: JsonValue = json.loads(value)
    except ValueError:
        return value
    return parsed


def _canonical(value: JsonValue) -> str:
    """Canonical JSON text: sorted keys, compact separators, non-ASCII kept."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def vector_trace(
    task: str,
    exchanges: Sequence[tuple[str | None, str, JsonValue, JsonValue | None]],
    *,
    trace_id: str = "vector",
    context: dict[str, JsonValue] | None = None,
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    result_order: Sequence[int] | None = None,
) -> Trace:
    """A compact synthetic trace for golden vectors and tests.

    Args:
        task: Task instruction.
        exchanges: ``(call ID, tool name, arguments, reply)``; a None reply records no result.
        trace_id: Trace ID.
        context: Request context.
        start: First span time.
        result_order: Order the results are recorded in; the call order by default.

    Returns:
        A normalized trace with one call span and one result span per exchange.
    """
    spans: list[TraceSpan] = []
    for position, (call_id, name, arguments, _) in enumerate(exchanges):
        attributes: dict[str, JsonValue] = {
            OPERATION: "chat",
            TOOL_NAME: name,
            "gen_ai.tool.call.arguments": json.dumps(arguments),
        }
        if call_id is not None:
            attributes[CALL_ID] = call_id
        at = start + timedelta(seconds=position)
        spans.append(
            TraceSpan(
                span_id=f"{trace_id}-call-{position}",
                name="tool.call",
                started_at=at,
                ended_at=at,
                attributes=attributes,
            )
        )
    order = list(result_order) if result_order is not None else list(range(len(exchanges)))
    for offset, position in enumerate(order):
        call_id, name, _, reply = exchanges[position]
        if reply is None:
            continue
        attributes = {OPERATION: RESULT_OPERATION, TOOL_NAME: name, "gen_ai.tool.message": reply}
        if call_id is not None:
            attributes[CALL_ID] = call_id
        at = start + timedelta(seconds=len(exchanges) + offset)
        spans.append(
            TraceSpan(
                span_id=f"{trace_id}-result-{position}",
                name="tool.result",
                started_at=at,
                ended_at=at,
                attributes=attributes,
            )
        )
    return Trace(
        trace_id=trace_id,
        task=task,
        initial_context=context or {},
        spans=tuple(spans),
        source=TraceSource(
            identity=SourceIdentity(kind="generated", source_id="golden-vector"),
            semantic_convention_version="1.37.0",
        ),
    )


_REFUND = (
    ("call-a", "get_order", {"order_id": "#W1"}, '{"status": "pending", "total": 10.5}'),
    ("call-b", "get_user", {"user_id": "u1"}, '{"orders": ["#W1"]}'),
    ("call-c", "cancel", {"order_id": "#W1"}, "cancelled"),
)
TRACE_CONTENT_VECTORS = (
    GoldenVector(
        trace=vector_trace("Cancel order #W1 for u1.", _REFUND, result_order=(1, 0, 2)),
        fingerprint="22cebf985857a5527c5502ec51a1d28aeda2d97bd6bdda546f4c5adb3137e43c",
    ),
    GoldenVector(
        trace=vector_trace(
            "Cancel order #W1 for u1.",
            (*_REFUND, ("call-d", "get_user", {"user_id": "u1"}, '{"orders": ["#W1"]}')),
        ),
        fingerprint="f9fb8bf9219896d6c723cc3db2046aef6ab042b7fb6a3c0a1774688f67c865ad",
    ),
    GoldenVector(
        trace=vector_trace("Cancel order #W1 for u1.", ((None, "get_order", {}, "{}"),)),
        fingerprint=None,
    ),
)
TRACE_CONTENT = Scheme(
    specification=TRACE_CONTENT_SPECIFICATION,
    compute=trace_content_fingerprint,
    vectors=TRACE_CONTENT_VECTORS,
)
SCHEMES: tuple[Scheme, ...] = (TRACE_CONTENT,)
"""Every scheme this code implements, newest last; the newest computable one is primary."""
