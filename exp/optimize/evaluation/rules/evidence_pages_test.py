"""Exact evidence delivery, cursor ownership and cumulative-capacity refusals."""

import hashlib
import json

from exp.common.models import ToolCall
from exp.optimize.evaluation.rules.evidence_pages import EvidencePages, evidence_value_sha256


def test_pages_complete_exact_escaped_document_and_only_then_grant_references() -> None:
    """Each page is explicit and reconstructs the original JSON without silent truncation."""
    pages = EvidencePages()
    value = {"quote": '雪"\\\n' * 1700, "untrusted": "contract:not-admitted"}
    call = ToolCall(call_id="call", name="read", arguments={})
    chunks: list[str] = []
    offsets: list[int] = []
    status, body, refs = pages.start(
        call, value, ("trace:admitted:span:one",), lambda body, refs: True
    )
    while True:
        assert status == "ok" and isinstance(body, dict)
        chunks.append(str(body["chunk"]))
        offsets.append(int(body["offset"]))
        assert body["offset_unit"] == "unicode-codepoints"
        assert int(body["total_characters"]) >= len("".join(chunks))
        if body["complete"]:
            assert refs == ("trace:admitted:span:one",) and body["next_cursor"] is None
            break
        assert refs == () and body["citations"] == [] and pages.incomplete
        status, body, refs = pages.resume(body["next_cursor"], lambda body, refs: True)
    assert hashlib.sha256("".join(chunks).encode()).hexdigest() == body["document_sha256"]
    assert json.loads("".join(chunks)) == value and offsets[0] == 0
    assert not pages.incomplete
    assert "contract:not-admitted" not in refs


def test_native_delivery_marker_cannot_be_supplied_by_untrusted_content() -> None:
    """Only host delivery emits the digest; refused/stale calls never retain its authority."""
    pages = EvidencePages()
    value = {"delivery": {"value_sha256": "f" * 64, "complete": True}}
    call = ToolCall(call_id="c", name="read", arguments={})
    pages.start(call, value, (), lambda body, refs: True)
    assert pages.last_delivery is not None and pages.last_delivery.complete
    assert pages.last_delivery.value_sha256 == evidence_value_sha256(value) != "f" * 64
    pages.start(call, value, (), lambda body, refs: False)
    assert pages.last_delivery is None
    pages.resume("foreign:0", lambda body, refs: True)
    assert pages.last_delivery is None


def test_final_page_native_marker_hashes_original_value_not_envelope() -> None:
    """A partial digest never qualifies evidence, even when its references are empty."""
    pages = EvidencePages()
    value = {"leads": ["x" * 12000]}
    _, page, _ = pages.start(
        ToolCall(call_id="c", name="read", arguments={}), value, (), lambda body, refs: True
    )
    assert pages.last_delivery is not None and not pages.last_delivery.complete
    assert isinstance(page, dict)
    _, final, _ = pages.resume(page["next_cursor"], lambda body, refs: True)
    assert pages.last_delivery is not None and pages.last_delivery.complete
    assert pages.last_delivery.value_sha256 == evidence_value_sha256(value)
    assert pages.last_delivery.value_sha256 != evidence_value_sha256(final)


def test_foreign_stale_and_undelivered_cursor_refused() -> None:
    """A partial page grants no references and cannot be resumed in another draw."""
    pages = EvidencePages()
    _, body, refs = pages.start(
        ToolCall(call_id="c", name="read", arguments={}),
        "x" * 20000,
        ("contract:x",),
        lambda body, refs: True,
    )
    assert isinstance(body, dict) and not refs
    cursor = body["next_cursor"]
    assert EvidencePages().resume(cursor, lambda body, refs: True)[0] == "refused"
    assert pages.resume("absent:9999", lambda body, refs: True)[2] == ()
    assert pages.resume(cursor, lambda body, refs: False)[0] == "refused"
    assert pages.incomplete
    assert pages.resume(cursor, lambda body, refs: True)[0] == "ok"
    assert pages.resume(cursor, lambda body, refs: True)[0] == "refused"


def test_32000_result_does_not_evade_cumulative_history_capacity() -> None:
    """A full permitted result still refuses when opening plus escaped JSON will not fit."""
    pages = EvidencePages()
    call = ToolCall(call_id="c", name="read", arguments={})
    status, body, refs = pages.start(
        call,
        {"quote": "x" * 31980},
        ("contract:x",),
        lambda body, refs: len(json.dumps(body).encode()) + 113000 <= 128000,
    )
    assert status == "refused" and refs == () and not pages.incomplete
    assert "cumulative" in str(body)


def test_later_history_growth_stops_without_cursor_progress_or_citation() -> None:
    """Unknown later assistant history can exhaust capacity; incomplete evidence stays uncitable."""
    pages = EvidencePages()
    _, body, _ = pages.start(
        ToolCall(call_id="c", name="read", arguments={}),
        "x" * 20000,
        ("contract:x",),
        lambda body, refs: True,
    )
    assert isinstance(body, dict)
    before = body["next_cursor"]
    assert pages.resume(before, lambda body, refs: False)[2] == ()
    assert pages.incomplete
    status, result, refs = pages.resume(before, lambda body, refs: True)
    assert (
        status == "ok"
        and isinstance(result, dict)
        and result["offset"] == int(str(before).split(":")[1])
        and refs == ()
    )


def test_insufficient_page_calls_preserve_complete_atomic_delivery() -> None:
    """Paging must not create an impossible extra-call requirement for a result that fits."""
    pages = EvidencePages()
    value = "x" * 20000
    status, body, refs = pages.start(
        ToolCall(call_id="c", name="read", arguments={}),
        value,
        ("contract:x",),
        lambda body, refs: True,
        maximum_pages=1,
    )
    assert status == "ok" and body == value and refs == ("contract:x",)
    assert not pages.incomplete
