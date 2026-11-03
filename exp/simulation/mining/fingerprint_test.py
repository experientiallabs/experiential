"""Exposure fingerprints: metadata-invariant, paired by call ID, versioned by specification."""

from datetime import UTC, datetime, timedelta

from exp.common.traces import Trace, TraceSpan
from exp.simulation.mining.fingerprint import (
    TRACE_CONTENT,
    TRACE_CONTENT_SPECIFICATION,
    GoldenVector,
    Scheme,
    trace_content_fingerprint,
    vector_trace,
)

EXCHANGES = (
    ("call-a", "get_order", {"order_id": "#W1"}, '{"status": "pending", "total": 10.5}'),
    ("call-b", "get_user", {"user_id": "u1"}, '{"orders": ["#W1"]}'),
    ("call-c", "cancel", {"order_id": "#W1"}, "cancelled at 2026-01-01T00:00:00Z"),
)
TASK = "Cancel order #W1 for u1."


def fingerprint(trace: Trace) -> str | None:
    """The current scheme's fingerprint."""
    return trace_content_fingerprint(trace)


def renamed(trace: Trace, prefix: str) -> Trace:
    """The same trace with every span ID rewritten."""
    return trace.model_copy(
        update={
            "spans": tuple(
                span.model_copy(update={"span_id": f"{prefix}-{span.span_id}"})
                for span in trace.spans
            )
        }
    )


def test_metadata_never_changes_the_fingerprint_but_content_does() -> None:
    """A1: IDs, timestamps, exporter metadata, call order and replay_from are outside."""
    base = vector_trace(TASK, EXCHANGES, context={"task_lineage": "u1", "replay_from": 1})
    expected = fingerprint(base)
    assert expected is not None
    variants = [
        renamed(base, "other"),
        vector_trace(TASK, EXCHANGES, trace_id="reimported"),
        vector_trace(TASK, EXCHANGES, start=datetime(2030, 5, 5, 1, 2, 3, 456789, tzinfo=UTC)),
        vector_trace(TASK, EXCHANGES, context={"task_lineage": "elsewhere", "replay_from": 2}),
        vector_trace(TASK, tuple(reversed(EXCHANGES))),
        base.model_copy(
            update={
                "spans": tuple(
                    span.model_copy(
                        update={
                            "attributes": {
                                **span.attributes,
                                "exporter.version": "9.9",
                                "service.name": "recorder",
                            }
                        }
                    )
                    for span in base.spans
                )
            }
        ),
        base.model_copy(
            update={
                "spans": tuple(
                    span.model_copy(
                        update={
                            "started_at": span.started_at.replace(microsecond=0),
                            "ended_at": span.ended_at.replace(microsecond=0),
                        }
                    )
                    for span in base.spans
                )
            }
        ),
    ]
    assert {fingerprint(variant) for variant in variants} == {expected}
    changed_argument = (EXCHANGES[0], EXCHANGES[1], ("call-c", "cancel", {"order_id": "#W2"}, "ok"))
    assert fingerprint(vector_trace(TASK, changed_argument)) != expected
    changed_reply = (*EXCHANGES[:2], ("call-c", "cancel", {"order_id": "#W1"}, "refused"))
    assert fingerprint(vector_trace(TASK, changed_reply)) != expected
    assert fingerprint(vector_trace(TASK + " ", EXCHANGES)) != expected, "content is verbatim"
    duplicated = (*EXCHANGES, ("call-d", *EXCHANGES[1][1:]))
    assert fingerprint(vector_trace(TASK, duplicated)) != expected, "duplicates are counted"


def test_tied_timestamps_with_different_span_ids_do_not_change_the_fingerprint() -> None:
    """A1: identical timestamps never let span IDs order or identify anything."""
    at = datetime(2026, 1, 1, tzinfo=UTC)
    tied = vector_trace(TASK, EXCHANGES, start=at)
    flattened = tied.model_copy(
        update={
            "spans": tuple(
                span.model_copy(update={"started_at": at, "ended_at": at}) for span in tied.spans
            )
        }
    )
    assert fingerprint(renamed(flattened, "z")) == fingerprint(renamed(flattened, "a"))
    assert fingerprint(flattened) == fingerprint(tied)


def test_results_pair_by_call_id_and_ambiguous_ids_leave_the_trace_unresolved() -> None:
    """A2: out-of-order results pair correctly; missing or repeated IDs resolve nothing."""
    in_order = vector_trace(TASK, EXCHANGES)
    out_of_order = vector_trace(TASK, EXCHANGES, result_order=(2, 0, 1))
    assert fingerprint(out_of_order) == fingerprint(in_order) is not None
    # Pairing by result position would swap these two replies; pairing by ID never does.
    swapped = (
        ("call-a", "get_order", {"order_id": "#W1"}, '{"orders": ["#W1"]}'),
        ("call-b", "get_user", {"user_id": "u1"}, '{"status": "pending", "total": 10.5}'),
        EXCHANGES[2],
    )
    assert fingerprint(vector_trace(TASK, swapped, result_order=(1, 0, 2))) != fingerprint(in_order)
    missing = ((None, *EXCHANGES[0][1:]), *EXCHANGES[1:])
    assert fingerprint(vector_trace(TASK, missing)) is None
    call_only = ((None, "get_order", {"order_id": "#W1"}, None), *EXCHANGES[1:])
    assert fingerprint(vector_trace(TASK, call_only)) is None, "a call without an ID"
    repeated = (EXCHANGES[0], ("call-a", *EXCHANGES[1][1:]), EXCHANGES[2])
    assert fingerprint(vector_trace(TASK, repeated)) is None
    orphan = in_order.model_copy(
        update={
            "spans": (
                *in_order.spans,
                TraceSpan(
                    span_id="orphan",
                    name="tool.result",
                    started_at=in_order.spans[-1].started_at + timedelta(seconds=1),
                    ended_at=in_order.spans[-1].started_at + timedelta(seconds=1),
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.call.id": "call-z",
                        "gen_ai.tool.message": "stray",
                    },
                ),
            )
        }
    )
    assert fingerprint(orphan) is None, "a result naming no call collides"
    no_result = (*EXCHANGES[:2], ("call-c", "cancel", {"order_id": "#W1"}, None))
    assert fingerprint(vector_trace(TASK, no_result)) not in (None, fingerprint(in_order))


def test_vectors_gate_computability_and_the_specification_fixes_the_identity() -> None:
    """A4: a vector mismatch makes a scheme not computable; a spec edit changes its digest."""
    assert TRACE_CONTENT.computable
    [first, *rest] = TRACE_CONTENT.vectors
    broken = Scheme(
        specification=TRACE_CONTENT_SPECIFICATION,
        compute=trace_content_fingerprint,
        vectors=(GoldenVector(trace=first.trace, fingerprint="0" * 64), *rest),
    )
    assert not broken.computable
    assert broken.identity.conflicts(TRACE_CONTENT.identity), "same name and version"
    edited = Scheme(
        specification={**TRACE_CONTENT_SPECIFICATION, "unicode": "NFC before hashing"},
        compute=trace_content_fingerprint,
        vectors=TRACE_CONTENT.vectors,
    )
    assert edited.computable, "vectors cannot prove the semantics are unchanged"
    assert edited.identity.definition_sha256 != TRACE_CONTENT.identity.definition_sha256
    assert edited.identity.conflicts(TRACE_CONTENT.identity)
