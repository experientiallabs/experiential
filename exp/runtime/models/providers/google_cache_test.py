"""Provider-free Google cache planning tests using synthetic captured-shape text."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, canonical_json_bytes, sha256_json
from exp.common.models import ToolCall
from exp.common.models.content import ImageContentPart, TextContentPart
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers import google_cache as google_cache_module
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import cache_markers
from exp.runtime.models.providers.google_cache import (
    GoogleCachePlan,
    VertexCacheProject,
    build_automatic_google_cache_plans,
    build_google_cache_plan,
)
from exp.runtime.models.providers.messages_payloads import gemini_generate_content_stream_payload

_MODEL = "gemini-2.5-pro"
_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:streamGenerateContent?alt=sse"
_MARKER: JsonObject = {"type": "ephemeral"}


def _profile(url: str = _URL) -> GatewayWireProfile:
    """Return an official native profile without any authenticated credentials."""
    return GatewayWireProfile(dialect="gemini_generate_content", url=url, model_id=_MODEL)


def _marked(text: str, *, suffix: str = "", marker: JsonObject | None = None) -> GatewayMessage:
    """Build one user checkpoint with an optional uncached text-block suffix."""
    blocks: tuple[JsonObject, ...] = (
        {"type": "text", "text": text, "cache_control": marker or _MARKER},
    )
    if suffix:
        suffix_block: JsonObject = {"type": "text", "text": suffix}
        blocks += (suffix_block,)
    return GatewayMessage(role="user", content=text + suffix, provider_text_blocks=blocks)


def _request() -> GatewayRequest:
    """Return the customer's common system, large marked prefix, question pattern."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="system", content="Use only the provided fruit inventory."),
            _marked("Apple inventory: 42.\n" * 1_024),
            GatewayMessage(role="user", content="How many apples are available?"),
        ),
        maximum_output_tokens=128,
        temperature=0.25,
    )


def _payload(request: GatewayRequest) -> JsonObject:
    """Use the real native wire builder rather than a parallel test projection."""
    return gemini_generate_content_stream_payload(_MODEL, request)


def _plan(request: GatewayRequest | None = None) -> GoogleCachePlan:
    """Require eligibility for a benign canonical text request."""
    request = request or _request()
    plan = build_google_cache_plan(_profile(), request, _payload(request))
    assert plan is not None
    return plan


def test_common_marked_prefix_preserves_system_contents_and_generation() -> None:
    """A three-message marked request moves only the cached prefix and system."""
    request = _request()
    upstream = _payload(request)
    before = deepcopy(upstream)
    plan = build_google_cache_plan(_profile(), request, upstream)
    assert plan is not None
    assert plan.create_url == "https://generativelanguage.googleapis.com/v1beta/cachedContents"
    assert plan.model == f"models/{_MODEL}"
    assert plan.resource_prefix == "cachedContents/"
    assert plan.ttl_seconds == 300
    assert plan.create_payload == {
        "model": f"models/{_MODEL}",
        "ttl": "300s",
        "systemInstruction": upstream["systemInstruction"],
        "contents": cast("list[JsonObject]", upstream["contents"])[:1],
    }
    assert plan.generation_payload == {
        "contents": cast("list[JsonObject]", upstream["contents"])[1:],
        "generationConfig": upstream["generationConfig"],
    }
    assert plan.conservative_input_bound == len(canonical_json_bytes(plan.create_payload))
    assert "generationConfig" not in plan.create_payload
    assert "cache_control" not in canonical_json_bytes(plan.create_payload).decode()
    assert "cachedContent" not in plan.generation_payload
    assert upstream == before
    assert request.messages[1].provider_text_blocks[0]["cache_control"] == _MARKER
    assert _profile().preserves_cache_control is False


def test_captured_shape_decoder_retains_positions_and_last_explicit_checkpoint() -> None:
    """Synthetic data follows retained multi-system-block and user-block shapes."""
    request = decode_messages(
        {
            "model": "fruit-assistant",
            "max_tokens": 128,
            "system": [
                {"type": "text", "text": "Read the inventory."},
                {"type": "text", "text": "Answer briefly.", "cache_control": _MARKER},
                {"type": "text", "text": "Do not invent counts.", "cache_control": _MARKER},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Fruit context. "},
                        {"type": "text", "text": "Apples: 42.", "cache_control": _MARKER},
                    ],
                },
                {"role": "user", "content": "Count apples."},
            ],
        }
    ).request
    plan = _plan(request)
    assert plan.create_payload["systemInstruction"] == {
        "parts": [{"text": "Read the inventory.\n\nAnswer briefly.\n\nDo not invent counts."}]
    }
    assert plan.create_payload["contents"] == [
        {"role": "user", "parts": [{"text": "Fruit context. Apples: 42."}]}
    ]
    assert plan.generation_payload["contents"] == [
        {"role": "user", "parts": [{"text": "Count apples."}]}
    ]


def test_last_of_several_user_checkpoints_includes_all_prior_content() -> None:
    """Selecting a later leading-user checkpoint never caches only that message."""
    request = _request().model_copy(
        update={
            "messages": (
                _marked("First marked context."),
                GatewayMessage(role="user", content="Unmarked middle context."),
                _marked("Last marked context.", marker={"type": "ephemeral", "ttl": "5m"}),
                GatewayMessage(role="user", content="Question."),
            )
        }
    )
    plan = _plan(request)
    assert (
        plan.create_payload["contents"]
        == cast("list[JsonObject]", _payload(request)["contents"])[:3]
    )


@pytest.mark.parametrize("split_native_parts", (False, True))
def test_partial_user_checkpoint_preserves_exact_text_suffix(split_native_parts: bool) -> None:
    """A block boundary inside flattened user text remains byte-exact after split."""
    prefix = "Apple context: 蘋果\n"
    suffix = "\nQuestion: count apples."
    request = _request().model_copy(update={"messages": (_marked(prefix, suffix=suffix),)})
    upstream = _payload(request)
    if split_native_parts:
        upstream["contents"] = [
            {"role": "user", "parts": [{"text": "Apple "}, {"text": "context: 蘋果\n" + suffix}]}
        ]
    plan = build_google_cache_plan(_profile(), request, upstream)
    assert plan is not None
    cached = cast("list[JsonObject]", plan.create_payload["contents"])
    remaining = cast("list[JsonObject]", plan.generation_payload["contents"])
    cached_text = "".join(
        cast("str", part["text"]) for part in cast("list[JsonObject]", cached[0]["parts"])
    )
    remaining_text = "".join(
        cast("str", part["text"]) for part in cast("list[JsonObject]", remaining[0]["parts"])
    )
    assert cached_text == prefix
    assert remaining_text == suffix
    assert cached_text + remaining_text == request.messages[0].content


def test_full_leading_system_checkpoint_retains_uncached_user_contents() -> None:
    """A complete leading system-only checkpoint needs no invented cache contents."""
    request = _request().model_copy(
        update={
            "messages": (
                GatewayMessage(role="system", content="First instruction."),
                GatewayMessage(
                    role="system",
                    content="Second instruction.",
                    provider_text_blocks=(
                        {"type": "text", "text": "Second instruction.", "cache_control": _MARKER},
                    ),
                ),
                GatewayMessage(role="user", content="Question."),
            )
        }
    )
    plan = _plan(request)
    assert "contents" not in plan.create_payload
    assert plan.create_payload["systemInstruction"] == _payload(request)["systemInstruction"]
    assert plan.generation_payload["contents"] == _payload(request)["contents"]


@pytest.mark.parametrize(
    "choice", (None, "auto", "none", "required", GatewayNamedToolChoice(name="lookup"))
)
def test_function_tools_and_config_move_with_cached_system_and_text(
    choice: str | GatewayNamedToolChoice | None,
) -> None:
    """Ordinary functions are cached in full while sampling remains per generation."""
    request = _request().model_copy(
        update={
            "tools": (
                GatewayToolDefinition(name="lookup", parameters={"type": "object"}),
                GatewayToolDefinition(
                    name="count", description="Count fruit", parameters={"type": "object"}
                ),
            ),
            "tool_choice": choice,
        }
    )
    plan = _plan(request)
    assert plan.create_payload["tools"] == _payload(request)["tools"]
    assert "tools" not in plan.generation_payload
    assert "toolConfig" not in plan.generation_payload
    if choice is not None:
        assert plan.create_payload["toolConfig"] == _payload(request)["toolConfig"]
    assert plan.generation_payload["generationConfig"] == _payload(request)["generationConfig"]


@pytest.mark.parametrize("location", ("global", "us-central1", "europe-west4"))
def test_vertex_scope_and_region_are_exact(location: str) -> None:
    """Global and regional official Vertex URLs produce project-bound resources."""
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    scope = f"projects/test-project/locations/{location}"
    model = f"{scope}/publishers/google/models/{_MODEL}"
    profile = replace(
        _profile(f"https://{host}/v1/{model}:streamGenerateContent?alt=sse"),
        operational_region=location,
    )
    plan = build_google_cache_plan(profile, _request(), _payload(_request()))
    assert plan is not None
    assert plan.create_url == f"https://{host}/v1/{scope}/cachedContents"
    assert plan.model == model
    assert plan.resource_prefix == f"{scope}/cachedContents/"
    assert plan.apply(f"{scope}/cachedContents/cache-123")["cachedContent"] == (
        f"{scope}/cachedContents/cache-123"
    )
    with pytest.raises(ValueError, match="planned collection"):
        plan.apply("projects/other-project/locations/global/cachedContents/cache-123")


@pytest.mark.parametrize(
    "url",
    (
        _URL.replace("https://", "http://"),
        _URL.replace("generativelanguage.googleapis.com", "example.com"),
        _URL.replace(
            "generativelanguage.googleapis.com", "generativelanguage.googleapis.com.evil.test"
        ),
        _URL.replace("generativelanguage.googleapis.com", "generativelanguage.googleapis.com:443"),
        _URL.replace("generativelanguage.googleapis.com", "user@generativelanguage.googleapis.com"),
        _URL.replace("/v1beta/", "/v1/"),
        _URL.replace("streamGenerateContent", "generateContent"),
        _URL.replace(_MODEL, "other-model"),
        _URL.replace(_MODEL, "../" + _MODEL),
        _URL.replace(_MODEL, "%2E%2E%2F" + _MODEL),
        _URL + "#fragment",
        _URL + "&alt=sse",
        _URL + "&key=",
        _URL + "&unknown=value",
        _URL.replace("alt=sse", "alt=json"),
        "https://[invalid/",
        f"https://aiplatform.googleapis.com/v1/publishers/google/models/{_MODEL}:streamGenerateContent",
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/{_MODEL}:streamGenerateContent",
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/{_MODEL}:streamGenerateContent",
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-east1/publishers/google/models/{_MODEL}:streamGenerateContent",
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/other/models/{_MODEL}:streamGenerateContent",
        f"https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/{_MODEL}:streamGenerateContent?key=synthetic",
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/123:streamGenerateContent",
    ),
)
def test_unsupported_endpoints_do_not_plan(url: str) -> None:
    """No lookalike host, mismatched scope, alternate product or malformed URL qualifies."""
    assert build_google_cache_plan(_profile(url), _request(), _payload(_request())) is None


def test_profile_dialect_signed_body_and_vertex_region_must_match() -> None:
    """Native payload planning neither rewrites signed bodies nor trusts foreign dialects."""
    for profile in (
        replace(_profile(), dialect="openai_compatible"),
        replace(_profile(), signs_request_body=True),
        replace(
            _profile(
                f"https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/{_MODEL}:streamGenerateContent"
            ),
            operational_region="us-central1",
        ),
    ):
        assert build_google_cache_plan(profile, _request(), _payload(_request())) is None


def test_query_key_is_preserved_for_auth_but_absent_from_repr_and_digest() -> None:
    """A synthetic query credential is neither logged nor part of prefix identity."""
    request = _request()
    keyed = build_google_cache_plan(
        _profile(_URL + "&key=synthetic-only-not-a-secret"), request, _payload(request)
    )
    assert keyed is not None
    assert keyed.create_url.endswith("/cachedContents?key=synthetic-only-not-a-secret")
    assert "synthetic-only" not in repr(keyed)
    assert "Apple inventory" not in repr(keyed)
    assert keyed.prefix_sha256 == _plan().prefix_sha256


def test_payload_snapshots_and_apply_are_deeply_independent_and_plan_is_frozen() -> None:
    """Callers can mutate returned JSON without corrupting another request's plan."""
    request = _request()
    upstream = _payload(request)
    plan = build_google_cache_plan(_profile(), request, upstream)
    assert plan is not None
    expected_create = plan.create_payload
    expected_generation = plan.generation_payload
    upstream.clear()
    create = plan.create_payload
    cast("list[JsonObject]", create["contents"])[0]["parts"] = []
    generation = plan.generation_payload
    cast("list[JsonObject]", generation["contents"])[0]["parts"] = []
    applied = plan.apply("cachedContents/test_123")
    cast("list[JsonObject]", applied["contents"])[0]["parts"] = []
    assert plan.create_payload == expected_create
    assert plan.generation_payload == expected_generation
    assert plan.apply("cachedContents/next")["contents"] == expected_generation["contents"]
    with pytest.raises(FrozenInstanceError):
        plan.prefix_sha256 = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "name",
    (
        "cachedContents/",
        "cachedContents/.",
        "cachedContents/..",
        "cachedContents/a/b",
        "cachedContents/a?key=synthetic",
        "cachedContents/a#fragment",
        "cachedContents/%2E%2E",
        "cachedContents/a%2Fb",
        "cachedContents/a\\b",
        "cachedContents/a\n",
        "cachedContents/ a",
        "cachedContents/a.b",
        "projects/p/locations/global/cachedContents/a",
        "https://generativelanguage.googleapis.com/v1beta/cachedContents/a",
    ),
)
def test_apply_refuses_foreign_or_ambiguous_resource_names(name: str) -> None:
    """Only one simple identifier below the exact collection is a cache handle."""
    with pytest.raises(ValueError, match="planned collection"):
        _plan().apply(name)


def test_digest_is_canonical_prefix_only_and_includes_model_system_and_tools() -> None:
    """Suffix and generation changes reuse identity; resource semantics never do."""
    request = _request()
    plan = _plan(request)
    resource = plan.create_payload
    resource.pop("ttl")
    assert plan.prefix_sha256 == sha256_json(
        {"resource": resource, "policy": "google-explicit-text-prefix-v1"}
    )
    changed_suffix = request.model_copy(
        update={
            "messages": (
                *request.messages[:2],
                GatewayMessage(role="user", content="Other question"),
            ),
            "temperature": 0.75,
            "maximum_output_tokens": 32,
        }
    )
    assert _plan(changed_suffix).prefix_sha256 == plan.prefix_sha256
    changed_system = request.model_copy(
        update={
            "messages": (
                GatewayMessage(role="system", content="Other system"),
                *request.messages[1:],
            )
        }
    )
    assert _plan(changed_system).prefix_sha256 != plan.prefix_sha256
    changed_tools = request.model_copy(
        update={"tools": (GatewayToolDefinition(name="lookup", parameters={"type": "object"}),)}
    )
    assert _plan(changed_tools).prefix_sha256 != plan.prefix_sha256
    changed_profile = replace(
        _profile(), model_id="gemini-2.5-flash", url=_URL.replace(_MODEL, "gemini-2.5-flash")
    )
    changed_model = build_google_cache_plan(changed_profile, request, _payload(request))
    assert changed_model is not None
    assert changed_model.prefix_sha256 != plan.prefix_sha256
    reordered_payload = dict(reversed(tuple(_payload(request).items())))
    reordered = build_google_cache_plan(_profile(), request, reordered_payload)
    assert reordered is not None
    assert reordered.prefix_sha256 == plan.prefix_sha256


@pytest.mark.parametrize(
    "marker",
    (
        {"type": "ephemeral", "ttl": "1h"},
        {"type": "ephemeral", "ttl": None},
        {"type": "ephemeral", "ttl": "300s"},
        {"type": "persistent"},
        {"type": "ephemeral", "unknown": True},
    ),
)
def test_unsupported_or_malformed_hints_never_claim_an_explicit_plan(marker: JsonObject) -> None:
    """The decoder owns rejection; the pure helper still fails closed on bad hints."""
    request = _request()
    request = request.model_copy(
        update={"messages": (_marked("Context", marker=marker), request.messages[-1])}
    )
    assert build_google_cache_plan(_profile(), request, _payload(request)) is None


def test_unmarked_auto_marked_tool_marked_and_conflicting_ttls_are_ineligible() -> None:
    """No automatic checkpoint or tool declaration substitutes for a text boundary."""
    request = _request()
    variants = (
        request.model_copy(update={"messages": (GatewayMessage(role="user", content="Question"),)}),
        request.model_copy(update={"provider_cache_control": _MARKER}),
        request.model_copy(
            update={
                "tools": (
                    GatewayToolDefinition(name="lookup", parameters={}, cache_control=_MARKER),
                )
            }
        ),
        request.model_copy(
            update={
                "messages": (
                    _marked("Hour", marker={"type": "ephemeral", "ttl": "1h"}),
                    *request.messages[1:],
                )
            }
        ),
    )
    for variant in variants:
        assert build_google_cache_plan(_profile(), variant, _payload(variant)) is None


def test_unsupported_history_and_carriers_refuse_without_shifting_checkpoint() -> None:
    """Media, calls, late instructions and assistant markers do not migrate to text."""
    request = _request()
    media = GatewayMessage(
        role="user",
        content="Image question",
        content_parts=(
            TextContentPart(text="Image question"),
            ImageContentPart(url="https://example.com/synthetic.png"),
        ),
    )
    call = GatewayMessage(
        role="assistant", tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),)
    )
    tool = GatewayMessage(role="tool", tool_call_id="call-1", content="42", cache_control=_MARKER)
    for messages in (
        (*request.messages, media),
        (*request.messages, call, tool),
        (*request.messages, GatewayMessage(role="system", content="Late instruction")),
        (*request.messages, GatewayMessage(role="developer", content="Late instruction")),
        (*request.messages, _marked("Marked model reply").model_copy(update={"role": "assistant"})),
        (
            *request.messages,
            GatewayMessage(role="assistant", content="Reply"),
            _marked("Late user"),
        ),
    ):
        variant = request.model_copy(update={"messages": messages})
        assert build_google_cache_plan(_profile(), variant, _payload(request)) is None
    for update in (
        {"provider_server_tools": ({"type": "web_search_20250305", "name": "web_search"},)},
        {"provider_native_tools": ({"type": "web_search"},)},
    ):
        assert (
            build_google_cache_plan(
                _profile(), request.model_copy(update=update), _payload(request)
            )
            is None
        )


def test_ambiguous_system_boundaries_empty_continuation_and_text_mapping_refuse() -> None:
    """A partial system cannot coexist with cached system, and suffix is never invented."""
    request = _request()
    partial_system = GatewayMessage(
        role="system",
        content="First\n\nSecond",
        provider_text_blocks=(
            {"type": "text", "text": "First", "cache_control": _MARKER},
            {"type": "text", "text": "Second"},
        ),
    )
    ambiguous_user = GatewayMessage(
        role="user",
        content="First\n\nSecond",
        provider_text_blocks=partial_system.provider_text_blocks,
    )
    for messages in (
        (partial_system, request.messages[-1]),
        (_marked("Only content"),),
        (ambiguous_user, request.messages[-1]),
        (_marked(""), request.messages[-1]),
    ):
        variant = request.model_copy(update={"messages": messages})
        assert build_google_cache_plan(_profile(), variant, _payload(variant)) is None
    json_mode = request.model_copy(update={"json_object_output": True})
    assert build_google_cache_plan(_profile(), json_mode, _payload(json_mode)) is None


@pytest.mark.parametrize(
    "field", ("contents", "systemInstruction", "tools", "toolConfig", "cachedContent", "unknown")
)
def test_mismatched_or_extra_native_payload_fields_refuse(field: str) -> None:
    """A plan is produced only for the exact native semantics the canonical text explains."""
    request = _request()
    payload = _payload(request)
    payload[field] = {"unexpected": "synthetic"}
    assert build_google_cache_plan(_profile(), request, payload) is None


@pytest.mark.parametrize("location", ["global", "us-central1"])
def test_vertex_project_alias_needs_exact_host_number_before_resource_reuse(location: str) -> None:
    """An ID-addressed create uses only the verified numeric response namespace."""
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    url = (
        f"https://{host}/v1/projects/fruit-project/locations/{location}"
        f"/publishers/google/models/{_MODEL}:streamGenerateContent?alt=sse"
    )
    request = _request()
    plan = build_google_cache_plan(_profile(url), request, _payload(request))
    assert plan is not None
    assert plan.bind_vertex_project(None) is None
    bound = plan.bind_vertex_project(VertexCacheProject("fruit-project", "123456789"))
    assert bound is not None
    assert bound.create_url == plan.create_url
    assert bound.create_payload == plan.create_payload
    assert bound.generation_payload == plan.generation_payload
    assert bound.prefix_sha256 != plan.prefix_sha256
    resource = f"projects/123456789/locations/{location}/cachedContents/synthetic"
    assert bound.apply(resource)["cachedContent"] == resource
    for foreign in (
        resource.replace("123456789", "987654321"),
        resource.replace("123456789", "fruit-project"),
        resource.replace(f"/{location}/", "/europe-west1/"),
    ):
        with pytest.raises(ValueError, match="planned collection"):
            bound.apply(foreign)
    with pytest.raises(ValueError, match="admitted endpoint"):
        plan.bind_vertex_project(VertexCacheProject("other-project", "123456789"))
    assert bound.bind_vertex_project(VertexCacheProject("fruit-project", "123456789")) == bound
    other = plan.bind_vertex_project(VertexCacheProject("fruit-project", "987654321"))
    assert other is not None
    assert other.prefix_sha256 != bound.prefix_sha256


def test_numeric_vertex_namespace_does_not_need_alias_but_cannot_be_remapped() -> None:
    """Numeric endpoint scope is exact, while Gemini must never carry Vertex authority."""
    url = (
        "https://aiplatform.googleapis.com/v1/projects/123456789/locations/global"
        f"/publishers/google/models/{_MODEL}:streamGenerateContent"
    )
    request = _request()
    plan = build_google_cache_plan(_profile(url), request, _payload(request))
    assert plan is not None
    assert plan.bind_vertex_project(None) is plan
    assert plan.bind_vertex_project(VertexCacheProject("123456789", "123456789")) is plan
    with pytest.raises(ValueError, match="canonical number"):
        VertexCacheProject("123456789", "987654321")
    with pytest.raises(ValueError, match="must not contain"):
        _plan().bind_vertex_project(VertexCacheProject("fruit-project", "123456789"))
    assert _plan().bind_vertex_project(None) is not None


@pytest.mark.parametrize("number", ["", "0", "0123", "1.2", "-1", "１２３", "1/2", "9" * 21])
def test_vertex_project_number_is_canonical_decimal(number: str) -> None:
    """No normalization or caller-controlled path fragment can expand resource scope."""
    with pytest.raises(ValueError, match="canonical number"):
        VertexCacheProject("fruit-project", number)


@pytest.mark.parametrize("project", ["", ".", "..", "a/b", "a?b", "a%2fb", "a" * 257])
def test_vertex_project_alias_is_one_bounded_segment(project: str) -> None:
    """Host alias data must still satisfy the frozen endpoint segment grammar."""
    with pytest.raises(ValueError, match="canonical number"):
        VertexCacheProject(project, "123456789")


def test_plain_assistant_history_after_prefix_is_retained_verbatim() -> None:
    """An unmarked text-only suffix can contain ordinary conversation turns."""
    request = _request().model_copy(
        update={
            "messages": (
                _marked("Fruit context"),
                GatewayMessage(role="assistant", content="Ready"),
                GatewayMessage(role="user", content="Question"),
            )
        }
    )
    plan = _plan(request)
    assert (
        plan.generation_payload["contents"]
        == cast("list[JsonObject]", _payload(request)["contents"])[1:]
    )


def test_automatic_vertex_plans_preserve_changed_suffix_and_schema() -> None:
    """Whole-message caches share exact prefix bytes while schemas stay per request."""

    profile = _profile(
        f"https://aiplatform.us.rep.googleapis.com/v1/projects/fruit-project/locations/us/"
        f"publishers/google/models/{_MODEL}:streamGenerateContent?alt=sse"
    )
    first = _request().model_copy(
        update={
            "messages": (
                GatewayMessage(role="system", content="Follow the fictional inventory."),
                GatewayMessage(role="user", content="Apple inventory: 42.\n" * 1024),
                GatewayMessage(role="user", content="Batch 1"),
                GatewayMessage(role="user", content="Return the count."),
            )
        }
    )
    second = first.model_copy(
        update={
            "messages": (
                *first.messages[:2],
                GatewayMessage(role="user", content="Batch 2"),
                GatewayMessage(role="user", content="Return the count in another shape."),
            )
        }
    )
    a, b = _payload(first), _payload(second)
    b["generationConfig"] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    }
    pa = build_automatic_google_cache_plans(profile, first, a)
    pb = build_automatic_google_cache_plans(profile, second, b)
    assert pa and pb and pa[0].prefix_sha256 == pb[0].prefix_sha256
    for plan, original in [(pa[0], a), (pb[0], b)]:
        assert plan.create_url.startswith("https://aiplatform.us.rep.googleapis.com/")
        restored = deepcopy(plan.generation_payload)
        resource = plan.create_payload
        restored["systemInstruction"] = resource["systemInstruction"]
        restored["contents"] = cast("list[JsonValue]", resource["contents"]) + cast(
            "list[JsonValue]", restored["contents"]
        )
        assert restored == original
    assert pa[0].generation_payload != pb[0].generation_payload
    assert "generationConfig" not in pa[0].create_payload
    # A changed instruction changes every candidate. Nothing is normalized away.
    changed = first.model_copy(
        update={
            "messages": (
                GatewayMessage(role="system", content="Different instructions."),
                *first.messages[1:],
            )
        }
    )
    pc = build_automatic_google_cache_plans(profile, changed, _payload(changed))
    assert not {p.prefix_sha256 for p in pa} & {p.prefix_sha256 for p in pc}


@pytest.mark.parametrize("location", ["us", "eu"])
def test_jurisdictional_endpoint_requires_exact_host_and_location(location: str) -> None:
    """No suffix match, credential-bearing authority, or cross-location URL is trusted."""
    url = (
        f"https://aiplatform.{location}.rep.googleapis.com/v1/projects/fruit-project/"
        f"locations/{location}/publishers/google/models/{_MODEL}:streamGenerateContent"
    )
    plan = build_google_cache_plan(_profile(url), _request(), _payload(_request()))
    assert plan is not None and f"/locations/{location}/cachedContents" in plan.create_url
    for invalid in [
        url.replace(".rep.googleapis.com", ".rep.googleapis.com.evil.test"),
        url.replace(f"/locations/{location}/", "/locations/global/"),
        url.replace("https://", "https://name:password@"),
        url.replace(f"aiplatform.{location}.rep", "aiplatform.global.rep"),
    ]:
        assert build_google_cache_plan(_profile(invalid), _request(), _payload(_request())) is None


@pytest.mark.parametrize("shape", ["history", "single", "oversized", "gemini"])
def test_automatic_cache_skips_unsupported_shapes(shape: str) -> None:
    """Automatic selection never invents message boundaries or caches history."""

    profile = _profile(
        f"https://aiplatform.us.rep.googleapis.com/v1/projects/fruit-project/locations/us/publishers/google/models/{_MODEL}:streamGenerateContent"
    )
    request = _request()
    if shape == "history":
        request = request.model_copy(
            update={
                "messages": (
                    GatewayMessage(role="user", content="Fruit context." * 1024),
                    GatewayMessage(role="assistant", content="Ready."),
                    GatewayMessage(role="user", content="Count fruit."),
                )
            }
        )
    elif shape == "single":
        request = request.model_copy(
            update={"messages": (GatewayMessage(role="user", content="Fruit context." * 1024),)}
        )
    elif shape == "oversized":
        request = request.model_copy(
            update={
                "messages": (
                    GatewayMessage(role="user", content="a" * 262145),
                    GatewayMessage(role="user", content="Count fruit."),
                )
            }
        )
    elif shape == "gemini":
        profile = _profile()
    assert build_automatic_google_cache_plans(profile, request, _payload(request)) == ()


def test_automatic_plain_function_tools_are_excluded() -> None:
    """Unmarked function definitions must not become provider-held automatic resources."""
    request = _request().model_copy(
        update={
            "messages": (
                GatewayMessage(role="user", content="Fictional fruit context. " * 1000),
                GatewayMessage(role="user", content="Count oranges."),
            ),
            "tools": (GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        }
    )
    profile = _profile(
        f"https://aiplatform.us.rep.googleapis.com/v1/projects/fruit-project/locations/us/publishers/google/models/{_MODEL}:streamGenerateContent"
    )
    assert build_automatic_google_cache_plans(profile, request, _payload(request)) == ()


def test_automatic_cache_stops_before_copying_prefixes_past_the_byte_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge early message cannot make every later boundary copy the whole payload."""
    profile = _profile(
        f"https://aiplatform.us.rep.googleapis.com/v1/projects/fruit-project/locations/us/publishers/google/models/{_MODEL}:streamGenerateContent"
    )
    request = _request().model_copy(
        update={
            "messages": (
                GatewayMessage(role="user", content="Fictional fruit context. " * 400),
                GatewayMessage(role="user", content="a" * 262145),
                GatewayMessage(role="user", content="Batch 1"),
                GatewayMessage(role="user", content="Count fruit."),
            )
        }
    )
    real = google_cache_module._plan_at_checkpoint
    calls: list[int] = []

    def counted(
        endpoint: tuple[str, str, str],
        request: GatewayRequest,
        upstream_payload: JsonObject,
        message_index: int,
        offset: int,
    ) -> GoogleCachePlan | None:
        """Record each attempted checkpoint before building its real plan."""
        calls.append(message_index)
        return real(endpoint, request, upstream_payload, message_index, offset)

    monkeypatch.setattr(google_cache_module, "_plan_at_checkpoint", counted)
    plans = build_automatic_google_cache_plans(profile, request, _payload(request))
    assert calls == [0]
    assert len(plans) == 1


def test_automatic_cache_ignores_caller_text_markers() -> None:
    """A marked request caches exactly like its unmarked twin instead of opting out."""
    profile = _profile(
        f"https://aiplatform.us.rep.googleapis.com/v1/projects/fruit-project/locations/us/"
        f"publishers/google/models/{_MODEL}:streamGenerateContent?alt=sse"
    )
    marked = _request()
    assert cache_markers(marked)
    plain = marked.model_copy(
        update={
            "messages": tuple(
                GatewayMessage(role=m.role, content=m.content) for m in marked.messages
            )
        }
    )
    assert not cache_markers(plain)
    for request in (
        marked,
        marked.model_copy(update={"provider_cache_control": {"type": "ephemeral"}}),
    ):
        plans = build_automatic_google_cache_plans(profile, request, _payload(request))
        twins = build_automatic_google_cache_plans(profile, plain, _payload(plain))
        assert plans and [p.prefix_sha256 for p in plans] == [p.prefix_sha256 for p in twins]
