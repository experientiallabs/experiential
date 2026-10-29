"""Disk collection splitting preserves supported wrapper and conversation boundaries."""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject
from exp.common.traces.ingest.otlp_test import _payload, _payload_spans
from exp.common.traces.ingest.persistence import ingest_traces, read_ingested_traces
from exp.common.traces.ingest.sources import load_trace_source
from exp.common.traces.ingest.sources_test import _MINIMAL_VENDOR_EXPORTS


@pytest.mark.parametrize("source,payload", list(_MINIMAL_VENDOR_EXPORTS.items()))
def test_wrapped_sources_preserve_canonical_evidence(
    tmp_path: Path, source: str, payload: object
) -> None:
    """Explicit collection wrappers normalize identically through both source surfaces."""
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    expected = load_trace_source(source, path)
    _, receipt = ingest_traces("powerset", source_format=source, path=path, root=tmp_path / "state")
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected


@pytest.mark.parametrize(
    "shape", ["bare", "wrapped", "nested", "duplicate-wrapper", "invalid-tail"]
)
def test_chat_arrays_preserve_conversation_boundaries(tmp_path: Path, shape: str) -> None:
    """Bare message arrays stay one trace and invalid wrappers reject the full document."""
    messages = [{"role": "user", "content": "Research"}, {"role": "assistant", "content": "Done"}]
    conversation = {"messages": messages}
    if shape == "bare":
        text = json.dumps(messages)
    elif shape == "nested":
        text = json.dumps({"data": [{"conversations": [conversation]}]})
    elif shape == "duplicate-wrapper":
        text = '{"data":[{"wrong":true}],"data":[' + json.dumps(conversation) + "]}"
    elif shape == "invalid-tail":
        text = json.dumps([conversation, 7])
    else:
        text = json.dumps({"conversations": [conversation]})
    path = tmp_path / "input.json"
    path.write_text(text)
    expected = load_trace_source("chat-json", path)
    _, receipt = ingest_traces(
        "powerset", source_format="chat-json", path=path, root=tmp_path / "state"
    )
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected


def test_invalid_resource_attributes_reject_the_complete_document(tmp_path: Path) -> None:
    """A bad later resource cannot leave its document's earlier spans silently accepted."""
    payload = _payload()
    resources = payload["resourceSpans"]
    assert isinstance(resources, list)
    payload["resourceSpans"] = [
        *resources,
        {"resource": {"attributes": "invalid"}, "scopeSpans": []},
    ]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    expected = load_trace_source("otlp", path)
    assert not expected.traces and expected.issues
    _, receipt = ingest_traces("powerset", source_format="otlp", path=path, root=tmp_path / "state")
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected


def test_direct_otlp_span_with_non_envelope_metadata_keeps_its_shape(tmp_path: Path) -> None:
    """A non-array resourceSpans field does not override explicit direct-span identity."""
    spans = TypeAdapter(list[JsonObject]).validate_python(_payload_spans(_payload()))
    for span in spans:
        span["resourceSpans"] = None
    path = tmp_path / "input.json"
    path.write_text(json.dumps(spans))
    expected = load_trace_source("otlp", path)
    assert expected.traces and not expected.issues
    _, receipt = ingest_traces("powerset", source_format="otlp", path=path, root=tmp_path / "state")
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected
