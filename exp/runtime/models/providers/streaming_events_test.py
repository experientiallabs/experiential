"""Bounded provider-indexed streaming-event state tests."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind
from exp.runtime.models.providers import streaming_events
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.streaming_events import (
    OpenAIReasoningSummaryParser,
    retain_provider_entry,
)


def _create(kind: GatewayEventKind, **payload: object) -> GatewayEvent:
    """Build one normalized event for parser-only tests."""
    return GatewayEvent.model_validate({"kind": kind, "sequence_number": 0, **payload})


def test_empty_reasoning_deltas_do_not_allocate_provider_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct empty provider indices cannot exhaust retained-entry capacity."""
    monkeypatch.setattr(streaming_events, "MAXIMUM_RETAINED_PROVIDER_ENTRIES", 1)
    parser = OpenAIReasoningSummaryParser()

    for output_index in range(10):
        consumed, event = parser.consume(
            "response.reasoning_summary_text.delta",
            {"output_index": output_index, "summary_index": 0, "delta": ""},
            create=_create,
        )
        assert consumed is True
        assert event is None

    consumed, event = parser.consume(
        "response.reasoning_summary_text.delta",
        {"output_index": 0, "summary_index": 0, "delta": "bounded"},
        create=_create,
    )
    assert consumed is True
    assert event is not None
    assert event.text_delta == "bounded"


def test_reasoning_summary_entries_fail_at_the_retained_state_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty provider keys fail closed once the hard ceiling is reached."""
    monkeypatch.setattr(streaming_events, "MAXIMUM_RETAINED_PROVIDER_ENTRIES", 1)
    parser = OpenAIReasoningSummaryParser()
    parser.consume(
        "response.reasoning_summary_text.delta",
        {"output_index": 0, "summary_index": 0, "delta": "first"},
        create=_create,
    )

    with pytest.raises(
        ProviderResponseError,
        match=streaming_events.PROVIDER_OUTPUT_OVERFLOW_MESSAGE,
    ):
        parser.consume(
            "response.reasoning_summary_text.delta",
            {"output_index": 1, "summary_index": 0, "delta": "second"},
            create=_create,
        )


def test_provider_entry_insertion_is_atomic_at_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every provider tool-map call site inherits the same bounded insertion."""
    monkeypatch.setattr(streaming_events, "MAXIMUM_RETAINED_PROVIDER_ENTRIES", 1)
    entries: dict[int, str] = {}
    retain_provider_entry(entries, 0, "first")

    with pytest.raises(
        ProviderResponseError,
        match=streaming_events.PROVIDER_OUTPUT_OVERFLOW_MESSAGE,
    ):
        retain_provider_entry(entries, 1, "second")

    assert entries == {0: "first"}
