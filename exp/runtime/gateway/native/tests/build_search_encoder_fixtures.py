"""Freeze full conversation encodings independently of the native rebuild implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.native_recovery import recovery_prefix_digest
from exp.runtime.gateway.native_search_subject import subject_registration
from exp.runtime.gateway.native_search_wires import encoder_options
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.messages_payloads import (
    anthropic_messages_stream_payload,
    bedrock_converse_stream_payload,
    gemini_generate_content_stream_payload,
)
from exp.runtime.models.providers.openai_payloads import (
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)

FIELDS = (
    "messages",
    "input",
    "contents",
    "system",
    "systemInstruction",
    "instructions",
    "reasoning_history",
)
MARK: JsonObject = {"type": "ephemeral", "ttl": "1h"}


def messages() -> Iterator[Sequence[object]]:
    """Include replacements, media, hidden replay fields, and cache-marker displacement."""
    user: JsonObject = {"role": "user", "content": "Where? 世界"}
    call: JsonObject = {
        "role": "assistant",
        "tool_calls": [
            {"call_id": "c1", "name": "lookup", "arguments": {"city": "Zürich"}},
            {"call_id": "c2", "name": "lookup", "arguments": {"n": 2}},
        ],
    }
    result: JsonObject = {"role": "tool", "tool_call_id": "c1", "content": "Found"}
    yield [user]
    yield [{"role": "system", "content": "Only instructions"}]
    yield [{"role": "system", "content": " "}, user]
    yield [{"role": "system", "content": "One"}, {"role": "developer", "content": "Two"}, user]
    yield [user, {"role": "system", "content": "Late"}, {"role": "developer", "content": "Later"}]
    yield [user, call, result]
    yield [
        user,
        {
            **call,
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": {"n": 10**60, "$serde_json::private::Number": "12"},
                }
            ],
        },
        result,
    ]
    yield [user, call, {"role": "assistant", "content": "Progress"}, result]
    yield [
        user,
        call,
        {"role": "tool", "tool_call_id": "c2", "content": "Second"},
        result,
        {"role": "user", "content": "Next"},
    ]
    yield [user, call, {**result, "tool_is_error": True, "cache_control": MARK}]
    yield [
        user,
        call,
        {
            **result,
            "provider_tool_name": "lookup",
            "provider_tool_namespace": "ns",
            "provider_tool_caller": {"type": "direct"},
        },
    ]
    yield [user, call, {**result, "tool_references": ["lookup", "other"]}]
    yield [
        user,
        call,
        {**result, "content": "rewritten"},
        {"role": "user", "content": "Policy annotation"},
    ]
    yield [{"role": "user", "content": "Complete replacement of signed history"}]
    for role in ("system", "developer", "user", "assistant"):
        for blocks in (
            [
                {"type": "text", "text": "one", "cache_control": MARK},
                {"type": "text", "text": "two"},
            ],
            [{"type": "text", "text": "", "cache_control": MARK}, {"type": "text", "text": "one"}],
            [{"type": "text", "text": "one"}, {"type": "text", "text": "", "cache_control": MARK}],
            [{"type": "text", "text": "", "cache_control": MARK}],
            [{"type": "text", "text": "  "}],
            [{"type": "text", "text": None}],
            [{"type": "text", "text": False}],
            [{"type": "text", "text": {"n": 10**60}}],
            [{"type": "text"}],
        ):
            text = "".join(str(b.get("text", "")) for b in blocks)
            yield [user, {"role": role, "content": text, "provider_text_blocks": blocks}]
            if role in {"system", "developer"}:
                yield [{"role": role, "content": text, "provider_text_blocks": blocks}, user]
    media: list[JsonObject] = [
        {
            "kind": "image",
            "media_type": "image/png",
            "data": "aGVsbG8=",
            "detail": "high",
            "cache_control": MARK,
        },
        {"kind": "image", "url": "https://media.test/image.png"},
        {
            "kind": "document",
            "media_type": "application/pdf",
            "data": "aGVsbG8=",
            "name": " résumé / notes (A) ",
            "cache_control": MARK,
        },
        {"kind": "document", "url": "https://media.test/document.pdf"},
        {"kind": "audio", "media_type": "audio/mpeg", "data": "aGVsbG8="},
        {"kind": "video", "media_type": "video/mp4", "data": "aGVsbG8="},
        {"kind": "video", "url": "https://media.test/video.mp4"},
    ]
    for provider, reference in (
        ("openai", "file-123"),
        ("anthropic", "file_123"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/files/abc"),
        ("vertex", "gs://bucket/object"),
        ("bedrock", "s3://bucket/object"),
    ):
        for kind, mime in (("image", "image/png"), ("document", "application/pdf")):
            media.append(
                {
                    "kind": kind,
                    "media_type": mime,
                    "handle": {"provider": provider, "reference": reference},
                }
            )
    for part in media:
        message: JsonObject = {
            "role": "user",
            "content": "caption",
            "content_parts": [
                {"kind": "text", "text": "caption", "cache_control": MARK},
                part,
            ],
        }
        yield [message]
        yield [
            {
                **message,
                "provider_text_blocks": [
                    {"type": "text", "text": "caption", "cache_control": MARK}
                ],
            }
        ]
        if part["kind"] in {"image", "document"}:
            yield [user, call, {**message, "role": "tool", "tool_call_id": "c1"}]
        if part["kind"] == "image":
            yield [user, {**message, "role": "assistant"}]
    yield [
        user,
        {
            "role": "assistant",
            "content": "Answer",
            "provider_item_id": "msg_valid",
            "provider_output_index": 1,
            "provider_status": "completed",
            "provider_phase": "final_answer",
            "provider_reasoning": [
                {"kind": "encrypted_reasoning", "encrypted_content": "sealed", "output_index": 0}
            ],
        },
    ]
    yield [
        user,
        {
            "role": "assistant",
            "content": "Answer",
            "provider_item_id": "msg_foreign-id",
            "provider_output_index": 0,
        },
    ]
    for item in (
        {
            "type": "message",
            "id": "msg_good",
            "role": "user",
            "status": "completed",
            "content": [{"type": "output_text", "text": "old", "logprobs": []}],
        },
        {"type": "reasoning", "id": "item_foreign", "summary": []},
        {
            "type": "reasoning",
            "id": "rs_bad-id",
            "encrypted_content": "sealed",
            "status": "completed",
        },
    ):
        yield [user, {"role": "assistant", "provider_native_item": item}]
    yield [
        user,
        {
            "role": "assistant",
            "content": "Answer",
            "provider_reasoning": [{"kind": "thinking", "text": "Thought", "signature": "sig"}],
        },
    ]
    yield [
        user,
        {
            "role": "assistant",
            "content": "Answer",
            "provider_reasoning": [{"kind": "exposed_reasoning_content", "content": "Thought"}],
        },
    ]
    yield [
        user,
        {
            **call,
            "provider_reasoning": [
                {"kind": "reasoning_content", "content": "Thought", "route_sha256": "a" * 64}
            ],
        },
        result,
        {"role": "tool", "tool_call_id": "c2", "content": "Second"},
    ]
    yield [
        user,
        {
            **call,
            "tool_calls": [
                {
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": {"n": 1},
                    "provider_thought_signature": "sig",
                }
            ],
        },
        result,
    ]


def reference(profile: GatewayWireProfile, request: GatewayRequest) -> JsonObject:
    """Use the existing payload encoders directly to isolate conversation contracts."""
    if profile.dialect == "openai_compatible":
        return openai_compatible_stream_payload(
            profile.model_id,
            request,
            forwards_cache_control=profile.forwards_cache_control,
            forwards_prompt_cache_options=profile.forwards_prompt_cache_options,
            deepseek_reasoning_history=profile.deepseek_reasoning_history,
            system_messages_leading_only=profile.system_messages_leading_only,
            reasoning_output_exposed=profile.reasoning_output_exposed,
            fireworks_reasoning_route_sha256=profile.fireworks_reasoning_route_sha256,
        )
    if profile.dialect == "openai_responses":
        return openai_responses_stream_payload(
            profile.model_id,
            request,
            supports_temperature=True,
            forwards_prompt_cache_options=profile.forwards_prompt_cache_options,
        )
    if profile.dialect == "anthropic_messages":
        return anthropic_messages_stream_payload(profile.model_id, request)
    if profile.dialect == "gemini_generate_content":
        return gemini_generate_content_stream_payload(profile.model_id, request)
    return bedrock_converse_stream_payload(profile.model_id, request)


def cases() -> Iterator[JsonObject]:
    """Exercise each conversation with ordinary, marked, and JSON-output requests."""
    profiles = [
        GatewayWireProfile(
            dialect=d,
            url="https://provider.test/api",
            model_id="claude-sonnet-4-6" if d == "anthropic_messages" else "model",
        )
        for d in (
            "openai_compatible",
            "openai_responses",
            "anthropic_messages",
            "gemini_generate_content",
            "bedrock_converse_stream",
        )
    ]
    for kwargs in (
        {"forwards_cache_control": True},
        {"forwards_prompt_cache_options": True},
        {"deepseek_reasoning_history": True},
        {"system_messages_leading_only": True},
        {"reasoning_output_exposed": True},
        {"fireworks_reasoning_route_sha256": "a" * 64},
    ):
        profiles.append(replace(profiles[0], **kwargs))
    profiles.append(
        GatewayWireProfile(
            dialect="openai_responses",
            url="https://provider.test/api",
            model_id="model",
            forwards_prompt_cache_options=True,
        )
    )
    for i, raw in enumerate(messages()):
        canonical = tuple(GatewayMessage.model_validate(m) for m in raw)
        for mode in range(4):
            request = GatewayRequest(
                surface=GatewayApiSurface.MESSAGES if mode == 3 else GatewayApiSurface.RESPONSES,
                messages=canonical,
                maximum_output_tokens=512,
                json_object_output=mode == 2,
                prompt_cache_breakpoints=mode == 1,
                provider_cache_control=MARK if mode == 3 else None,
            )
            encodings: list[JsonValue] = []
            for p, profile in enumerate(profiles):
                expected: JsonValue
                try:
                    payload = reference(profile, request)
                    expected = {key: value for key, value in payload.items() if key in FIELDS}
                except (ValueError, RuntimeError) as exc:
                    expected = None
                    error = type(exc).__name__
                else:
                    error = None
                encodings.append(
                    {
                        "name": f"profile-{p}",
                        "dialect": profile.dialect,
                        "options": encoder_options(profile),
                        "expected": expected,
                        "error": error,
                    }
                )
            yield {
                "name": f"message-{i}/mode-{mode}",
                "subject": subject_registration(request, request.tools),
                "prefix": recovery_prefix_digest(request),
                "encodings": encodings,
            }


if __name__ == "__main__":
    target = Path(__file__).with_name("generated") / "search_encoders.jsonl"
    target.write_text(
        "".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n" for c in cases())
    )
