"""Normalize completed native gateway chat captures through the canonical chat loader."""

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.traces.ingest.chat_json import _conversation_observations
from exp.common.traces.ingest.vendor_observations import VendorObservation
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError, flatten_records
from exp.common.traces.ingest.vendor_source import VendorSource


def _records(payload: JsonValue) -> tuple[JsonObject, ...]:
    """Read a capture record, array, or explicit capture-export envelope."""
    return flatten_records(
        payload,
        vendor="experiential",
        wrapper_keys=("captures", "records"),
        record_keys=("request",),
    )


def _observations(record: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Preserve captured prompt and tool schemas while excluding incomplete responses."""
    request = record.get("request")
    response = record.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise VendorTraceFormatError("Experiential captures require request and response objects")
    if request.get("protocol") != "chat_completions":
        raise VendorTraceFormatError(
            "export chat-completions captures or canonical chat-json for this importer"
        )
    context = request.get("context")
    if not isinstance(context, dict) or not isinstance(context.get("messages"), list):
        raise VendorTraceFormatError("capture request.context.messages is missing")
    if response.get("kind") != "json" or response.get("status") != 200:
        raise VendorTraceFormatError(
            "capture is not a completed JSON response; export reconstructed chat-json for streams"
        )
    body = response.get("body")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise VendorTraceFormatError("capture requires one completed assistant choice")
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") in (None, "length", "content_filter") or not isinstance(
        message, dict
    ):
        raise VendorTraceFormatError("capture assistant response is incomplete or refused")
    if message.get("refusal"):
        raise VendorTraceFormatError("capture assistant response was refused")
    messages = context["messages"]
    assert isinstance(messages, list)
    conversation: JsonObject = {"messages": [*messages, message]}
    if "tools" in context:
        conversation["tools"] = context["tools"]
    if isinstance(request.get("request_id"), str):
        conversation["trace_id"] = request["request_id"]
    return _conversation_observations(conversation, ordinal)


EXPERIENTIAL_SOURCE = VendorSource(vendor="experiential", records=_records, convert=_observations)
