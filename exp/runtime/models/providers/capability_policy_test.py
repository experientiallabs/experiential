"""Tests for the route capability-preservation policy."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    StructuredTextFormat,
    ThinkingBlock,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import (
    coerce_capability,
    coerce_generation_parameters,
    coerce_structured_text_schema,
)
from exp.runtime.models.providers.reasoning_compat import efforts_by_nearness


def _budgeted_haiku_profile() -> GatewayWireProfile:
    """Build one Anthropic budgeted-enabled reasoning wire profile (haiku-4-5)."""
    return GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-haiku-4-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )


def _messages_request(**overrides: object) -> GatewayRequest:
    """Build one canonical Messages-surface request with overrides applied."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
    )
    return request.model_copy(update=dict(overrides))


def _replayed_thinking_turn() -> GatewayMessage:
    """Build one assistant history turn carrying a signed thinking block."""
    return GatewayMessage(
        role="assistant",
        content="answer",
        provider_reasoning=(ThinkingBlock(text="prior reasoning", signature="sig"),),
    )


def _request(**overrides: object) -> GatewayRequest:
    """Build one canonical request with overrides applied."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
    )
    return request.model_copy(update=dict(overrides)) if overrides else request


def _reasoning_profile(model_id: str) -> GatewayWireProfile:
    """Build one reasoning-capable OpenAI-compatible wire profile."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://provider.test",
        model_id=model_id,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )


def test_efforts_order_by_nearness_and_prefer_the_lower_level_on_ties() -> None:
    """Distance is ladder positions; a tie never spends more than requested."""
    assert efforts_by_nearness("ultra", ("low", "medium", "high", "xhigh", "max")) == (
        "xhigh",
        "max",
        "high",
        "medium",
        "low",
    )
    assert efforts_by_nearness("medium", ("low", "high")) == ("low", "high")
    assert efforts_by_nearness("minimal", ("high",)) == ("high",)
    assert efforts_by_nearness("high", ()) == ()
    assert efforts_by_nearness("bogus", ("high",)) == ()


def test_effort_snaps_to_the_nearest_route_level_with_disclosure() -> None:
    """An unsupported effort snaps onto the route ladder, disclosed by name."""
    # gpt-5.1 supports none/low/medium/high; xhigh must snap down to high.
    coercion = coerce_generation_parameters(
        (_reasoning_profile("gpt-5.1"),),
        _request(reasoning_effort="xhigh"),
    )
    assert coercion is not None
    assert coercion.request.reasoning_effort == "high"
    assert coercion.disclosures == ("reasoning_effort->high",)


def test_effort_snap_skips_levels_that_admit_no_rung() -> None:
    """The snap is the nearest level that actually serves, not the nearest
    level on paper: a rung carrying the closer effort may reject another
    requested control, and the coercion must not dead-end there."""
    xhigh_only_no_temperature = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://a.test",
        model_id="gpt-5.2-pro",
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )
    request = _request(reasoning_effort="ultra", temperature=0.5)
    coercion = coerce_generation_parameters(
        (xhigh_only_no_temperature, _reasoning_profile("gpt-5.1")),
        request,
    )
    assert coercion is not None
    # xhigh is nearer to ultra but only the temperature-rejecting rung has
    # it; high is the closest level that admits a serving rung.
    assert coercion.request.reasoning_effort == "high"
    assert coercion.disclosures == ("reasoning_effort->high",)


def test_any_effort_drops_on_a_route_with_no_reasoning_at_all() -> None:
    """A zero-reasoning route serves the request without its effort, disclosed.

    First-party clients pin effort globally (Claude Code sends its
    configured effortLevel to every model), so a named rejection here made
    whole sessions unusable against non-reasoning models the provider
    itself serves fine without the parameter (owner decision, 2026-09-01).
    """
    bare = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")
    for effort in ("none", "high", "xhigh"):
        coercion = coerce_generation_parameters((bare,), _request(reasoning_effort=effort))
        assert coercion is not None, effort
        assert coercion.request.reasoning_effort is None
        assert coercion.disclosures == ("reasoning_effort",)

    # The Messages surface carries the same effort inside output_config; the
    # drop strips exactly that key so nothing effort-shaped reaches a
    # provider that rejects it by name, while other verbatim keys survive.
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    marked = _request(reasoning_effort="high").model_copy(
        update={
            "surface": GatewayApiSurface.MESSAGES,
            "provider_output_config": {"effort": "high", "format": {"type": "text"}},
        }
    )
    coercion = coerce_generation_parameters((anthropic,), marked)
    assert coercion is not None
    assert coercion.request.provider_output_config == {"format": {"type": "text"}}
    effort_only = _request(reasoning_effort="high").model_copy(
        update={
            "surface": GatewayApiSurface.MESSAGES,
            "provider_output_config": {"effort": "high"},
        }
    )
    coercion = coerce_generation_parameters((anthropic,), effort_only)
    assert coercion is not None
    assert coercion.request.provider_output_config is None


def test_effort_drop_takes_adaptive_thinking_with_it_but_keeps_a_budget() -> None:
    """Adaptive thinking is the effort's own channel; a budget is not.

    Claude Code pins ``thinking: {type: adaptive}`` alongside effortLevel, and
    a route with no reasoning rung rejects the adaptive object by name after
    dispatch, so it drops with the effort and is disclosed as ``thinking``.
    A budgeted config carries semantics of its own and travels verbatim.
    """
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    adaptive = _request(reasoning_effort="high").model_copy(
        update={
            "surface": GatewayApiSurface.MESSAGES,
            "provider_output_config": {"effort": "high"},
            "provider_thinking_config": {"type": "adaptive"},
        }
    )
    coercion = coerce_generation_parameters((anthropic,), adaptive)
    assert coercion is not None
    assert coercion.request.reasoning_effort is None
    assert coercion.request.provider_output_config is None
    assert coercion.request.provider_thinking_config is None
    assert coercion.disclosures == ("reasoning_effort", "thinking")

    budgeted = adaptive.model_copy(
        update={"provider_thinking_config": {"type": "enabled", "budget_tokens": 2048}}
    )
    coercion = coerce_generation_parameters((anthropic,), budgeted)
    assert coercion is not None
    assert coercion.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 2048,
    }
    assert coercion.disclosures == ("reasoning_effort",)


def test_portable_effort_is_never_snapped() -> None:
    """A failure elsewhere must not trigger an effort substitution."""
    coercion = coerce_generation_parameters(
        (_reasoning_profile("gpt-5.1"),),
        _request(reasoning_effort="high", temperature=1.9),
    )
    assert coercion is None
    assert coerce_generation_parameters((_reasoning_profile("gpt-5.1"),), _request()) is None


def test_strict_tools_degrade_only_as_a_disclosed_drop() -> None:
    """strict:true weakens to best-effort schemas with the drop disclosed."""
    request = _request(
        tools=(
            GatewayToolDefinition(name="lookup", parameters={"type": "object"}, strict=True),
            GatewayToolDefinition(name="plain", parameters={"type": "object"}),
        )
    )
    coercion = coerce_capability("strict_tools", request)
    assert coercion is not None
    assert tuple(tool.strict for tool in coercion.request.tools) == (False, False)
    assert coercion.disclosures == ("tools.strict->false",)

    # Every other capability names a feature with no approximation.
    assert coerce_capability("developer_messages", request) is None
    assert coerce_capability("strict_tools", _request()) is None


def test_service_tier_drops_only_as_a_disclosed_coercion() -> None:
    """A route with no tier-preserving rung serves with the drop disclosed."""
    request = _request(service_tier="flex")

    coercion = coerce_capability("service_tier", request)

    assert coercion is not None
    assert coercion.request.service_tier is None
    assert coercion.disclosures == ("service_tier",)
    # A rejection that names the capability without the field stays closed.
    assert coerce_capability("service_tier", _request()) is None


def test_route_wide_capability_requires_unanimous_rejection() -> None:
    """Mixed per-rung rejections never produce the route-wide claim."""
    from exp.runtime.models.providers.capability_policy import route_wide_capability
    from exp.runtime.models.providers.errors import (
        ProviderCapabilityError,
        ProviderParameterError,
    )

    strict = ProviderCapabilityError(capability="strict_tools")
    developer = ProviderCapabilityError(capability="developer_messages")
    parameter = ProviderParameterError(
        message="The value 3 for 'top_k' is not supported.",
        param="top_k",
        code="invalid_parameter",
    )
    assert route_wide_capability((strict, strict), 2) == "strict_tools"
    assert route_wide_capability((strict, developer), 2) is None
    assert route_wide_capability((strict, parameter), 2) is None
    assert route_wide_capability((strict,), 2) is None
    assert route_wide_capability((), 0) is None


def test_effort_snap_requires_route_wide_construction_to_survive() -> None:
    """The snapped candidate must survive route-wide construction, not just
    per-rung admission: the narrowed rung set changes with the candidate, and
    the homogeneous encrypted-reasoning gate rejects a mixed Responses and
    Fireworks set that a farther candidate narrows past."""
    responses_medium = GatewayWireProfile(
        dialect="openai_responses",
        url="https://a.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("medium",),
    )
    fireworks_medium_high = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://b.test",
        model_id="kimi-k3",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("medium", "high"),
        fireworks_reasoning_route_sha256="a" * 64,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        reasoning_effort="low",
        include_encrypted_reasoning=True,
        response_store=False,
    )
    coercion = coerce_generation_parameters((responses_medium, fireworks_medium_high), request)
    assert coercion is not None
    # medium is nearer to low but admits both rungs, and the mixed set fails
    # the homogeneous encrypted-reasoning channel; high narrows to the
    # Fireworks rung alone and serves.
    assert coercion.request.reasoning_effort == "high"
    assert coercion.disclosures == ("reasoning_effort->high",)


def test_effort_snap_honors_the_admission_probe() -> None:
    """A candidate the downstream pipeline rejects must not block a farther
    one: the policy layer sees only wire profiles, so admission probes each
    candidate through deployment preflight before the snap is offered."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://a.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("medium", "high"),
    )
    probed: list[str | None] = []

    def only_high_serves(candidate: GatewayRequest) -> bool:
        probed.append(candidate.reasoning_effort)
        return candidate.reasoning_effort == "high"

    coercion = coerce_generation_parameters(
        (profile,),
        _request(reasoning_effort="low"),
        admits=only_high_serves,
    )
    assert coercion is not None
    assert coercion.request.reasoning_effort == "high"
    assert coercion.disclosures == ("reasoning_effort->high",)
    # medium is nearer to low and passes every profile-level check; only the
    # probe knows its rungs die at deployment preflight.
    assert probed == ["medium", "high"]


def test_effort_none_drop_honors_the_admission_probe() -> None:
    """The disclosed none-drop is withheld when downstream cannot serve it."""
    no_reasoning = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://a.test",
        model_id="kimi-k3",
    )
    assert (
        coerce_generation_parameters(
            (no_reasoning,),
            _request(reasoning_effort="none"),
            admits=lambda _candidate: False,
        )
        is None
    )


def test_open_structured_output_schema_closes_for_an_anthropic_rung() -> None:
    """Every object gains additionalProperties false, once, with disclosure.

    The Anthropic Messages validator rejects open objects that the
    OpenAI-family validators accept, so a caller who tested against one
    provider otherwise takes a post-dispatch 400 from the other.
    """
    schema: JsonObject = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "additionalProperties": False,
            },
            "tags": {"type": "array", "items": {"properties": {"label": {"type": "string"}}}},
            "either": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        },
        "$defs": {"leaf": {"type": "object", "additionalProperties": True}},
    }
    request = _request(structured_text=StructuredTextFormat(name="answer", json_schema=schema))
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    openai = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")

    coercion = coerce_structured_text_schema((openai, anthropic), request)
    assert coercion is not None
    assert coercion.disclosures == ("json_schema.additionalProperties->false",)
    assert coercion.request.structured_text is not None
    closed = coercion.request.structured_text.json_schema
    assert closed["additionalProperties"] is False
    properties = closed["properties"]
    assert isinstance(properties, dict)
    address = properties["address"]
    assert isinstance(address, dict)
    assert address["additionalProperties"] is False
    tags = properties["tags"]
    assert isinstance(tags, dict)
    items = tags["items"]
    assert isinstance(items, dict)
    assert items["additionalProperties"] is False
    assert "additionalProperties" not in tags
    either = properties["either"]
    assert isinstance(either, dict)
    assert either["anyOf"] == [{"type": "object", "additionalProperties": False}, {"type": "null"}]
    assert closed["$defs"] == {"leaf": {"type": "object", "additionalProperties": False}}
    assert properties["name"] == {"type": "string"}
    # The caller's own schema object is never mutated in place.
    assert "additionalProperties" not in schema

    # A route with no Anthropic rung dispatches the schema verbatim.
    assert coerce_structured_text_schema((openai,), request) is None
    # An already-closed schema needs no coercion and discloses nothing.
    closed_request = _request(
        structured_text=StructuredTextFormat(name="answer", json_schema=closed)
    )
    assert coerce_structured_text_schema((anthropic,), closed_request) is None
    # No structured output, nothing to close.
    assert coerce_structured_text_schema((anthropic,), _request()) is None


def test_non_strict_schema_is_left_open_for_an_anthropic_rung() -> None:
    """A permissive (non-strict) schema — notably a translated json_object "any JSON
    object" — is NOT force-closed, which would invert it into "no properties allowed"."""
    request = _request(
        structured_text=StructuredTextFormat(
            name="json_object", json_schema={"type": "object"}, strict=False
        )
    )
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    openai = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")

    assert coerce_structured_text_schema((openai, anthropic), request) is None


def test_mixed_rejections_coerce_only_the_service_tier() -> None:
    """Rungs declining differently drop the tier but never a guarantee."""
    from exp.runtime.models.providers.capability_policy import coerce_route_rejections
    from exp.runtime.models.providers.errors import ProviderCapabilityError

    tier = ProviderCapabilityError(capability="service_tier")
    parallel = ProviderCapabilityError(capability="parallel_tool_calls")
    strict = ProviderCapabilityError(capability="strict_tools")
    tiered = _request(service_tier="flex")

    # The Greptile mixed-waterfall shape: one rung declines parallel tool
    # calls, the other declines the tier; the disclosed drop serves it.
    mixed = coerce_route_rejections((parallel, tier), 2, tiered)
    assert mixed is not None
    assert mixed.request.service_tier is None
    assert mixed.disclosures == ("service_tier",)

    # A unanimous rejection keeps the existing coercion path.
    unanimous = coerce_route_rejections((tier, tier), 2, tiered)
    assert unanimous is not None and unanimous.disclosures == ("service_tier",)

    # Mixed rejections never degrade strict tools: some rung offered to
    # preserve the guarantee, so the named rejection stays the answer.
    strict_request = _request(
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}, strict=True),)
    )
    assert coerce_route_rejections((parallel, strict), 2, strict_request) is None


def test_disabled_thinking_drops_only_on_adaptive_only_anthropic_routes() -> None:
    """An explicit disabled config is dropped with disclosure where no rung honors it."""
    adaptive_only = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-opus-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    budgeted = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-haiku-4-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    shim = GatewayWireProfile(dialect="openai_compatible", url="https://shim.test")
    request = _request(
        surface=GatewayApiSurface.MESSAGES,
        provider_thinking_config={"type": "disabled"},
    )

    coercion = coerce_generation_parameters((adaptive_only, shim), request)
    assert coercion is not None
    assert coercion.disclosures == ("thinking.type->adaptive",)
    assert coercion.request.provider_thinking_config is None

    # A rung that honors ``disabled`` verbatim leaves the config alone.
    assert coerce_generation_parameters((budgeted, shim), request) is None
    # No Anthropic rung at all: nothing to translate onto.
    assert coerce_generation_parameters((shim,), request) is None
    # Only a disabled config is coercible; other types keep their own path.
    assert (
        coerce_generation_parameters(
            (adaptive_only, shim),
            _request(
                surface=GatewayApiSurface.MESSAGES,
                provider_thinking_config={"type": "adaptive"},
            ),
        )
        is None
    )
    # The admission probe still gates the offer.
    assert (
        coerce_generation_parameters((adaptive_only, shim), request, admits=lambda _c: False)
        is None
    )


def test_adaptive_thinking_translates_to_a_budget_on_a_budgeted_route() -> None:
    """A no-budget adaptive config becomes a legal enabled config, disclosed."""
    request = _messages_request(
        provider_thinking_config={"type": "adaptive"},
        maximum_output_tokens=8_000,
    )
    coercion = coerce_generation_parameters((_budgeted_haiku_profile(),), request)
    assert coercion is not None
    # min(max(8000 // 2, 1024), 16384) == 4000, and 1024 <= 4000 < 8000.
    assert coercion.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 4_000,
    }
    assert coercion.disclosures == ("thinking.type->enabled",)


def test_adaptive_thinking_with_a_legal_caller_budget_keeps_that_budget() -> None:
    """The model rejects ``adaptive`` by name, so a budget-carrying adaptive
    config still translates to enabled — honoring the caller's own depth."""
    request = _messages_request(
        provider_thinking_config={"type": "adaptive", "budget_tokens": 2_048},
        maximum_output_tokens=8_000,
    )
    coercion = coerce_generation_parameters((_budgeted_haiku_profile(),), request)
    assert coercion is not None
    assert coercion.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 2_048,
    }
    assert coercion.disclosures == ("thinking.type->enabled",)

    # With no output ceiling the caller's legal budget is likewise honored.
    unbounded = _messages_request(
        provider_thinking_config={"type": "adaptive", "budget_tokens": 30_000},
    )
    coercion = coerce_generation_parameters((_budgeted_haiku_profile(),), unbounded)
    assert coercion is not None
    assert coercion.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 30_000,
    }


def test_adaptive_thinking_with_an_illegal_caller_budget_falls_back_to_derived() -> None:
    """A caller budget below the floor or at/above max_tokens is replaced, never sent."""
    too_small = _messages_request(
        provider_thinking_config={"type": "adaptive", "budget_tokens": 512},
        maximum_output_tokens=8_000,
    )
    coercion = coerce_generation_parameters((_budgeted_haiku_profile(),), too_small)
    assert coercion is not None
    assert coercion.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 4_000,
    }
    # The substitution changes the requested depth, so it is disclosed by
    # itself beside the type translation.
    assert coercion.disclosures == ("thinking.type->enabled", "thinking.budget_tokens")

    at_ceiling = _messages_request(
        provider_thinking_config={"type": "adaptive", "budget_tokens": 8_000},
        maximum_output_tokens=8_000,
    )
    coercion = coerce_generation_parameters((_budgeted_haiku_profile(),), at_ceiling)
    assert coercion is not None
    assert coercion.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 4_000,
    }
    assert coercion.disclosures == ("thinking.type->enabled", "thinking.budget_tokens")


def test_adaptive_thinking_drops_when_no_legal_budget_fits_max_tokens() -> None:
    """A ceiling too small for any legal budget drops thinking, disclosed.

    History thinking blocks are left in place: Anthropic accepts replayed
    thinking blocks with no live thinking config, so the drop never strips them.
    """
    request = _messages_request(
        messages=(
            GatewayMessage(role="user", content="hi"),
            _replayed_thinking_turn(),
            GatewayMessage(role="user", content="again"),
        ),
        provider_thinking_config={"type": "adaptive"},
        # No budget in [1024, max_tokens) exists when max_tokens <= 1024.
        maximum_output_tokens=1_024,
    )
    coercion = coerce_generation_parameters((_budgeted_haiku_profile(),), request)
    assert coercion is not None
    assert coercion.request.provider_thinking_config is None
    # The replayed thinking block survives; the coercion only drops the config.
    assert any(message.provider_reasoning for message in coercion.request.messages)
    assert coercion.disclosures == ("thinking",)


def test_adaptive_thinking_drops_on_a_zero_reasoning_route_without_an_effort() -> None:
    """Adaptive thinking alone (no effort beside it) still drops, disclosed.

    Claude Code pins ``thinking: {type: adaptive}`` on every request but sends
    an effort only when one is configured; a genuinely non-reasoning model
    rejects the adaptive object by name either way, so it drops here rather than
    reaching the provider. A budgeted-enabled route (haiku) instead translates
    the same config; an adaptive-accepting reasoning route keeps it verbatim. A
    clear-thinking context edit rides on the thinking config and goes with it,
    while other edits stay verbatim.
    """
    anthropic = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    adaptive_only = _request().model_copy(
        update={
            "surface": GatewayApiSurface.MESSAGES,
            "provider_thinking_config": {"type": "adaptive"},
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "clear_tool_uses_20250919"},
                ]
            },
        }
    )
    coercion = coerce_generation_parameters((anthropic,), adaptive_only)
    assert coercion is not None
    assert coercion.request.reasoning_effort is None
    assert coercion.request.provider_thinking_config is None
    assert coercion.request.context_management == {"edits": [{"type": "clear_tool_uses_20250919"}]}
    assert coercion.disclosures == ("thinking",)

    only_clear_thinking = adaptive_only.model_copy(
        update={"context_management": {"edits": [{"type": "clear_thinking_20251015"}]}}
    )
    coercion = coerce_generation_parameters((anthropic,), only_clear_thinking)
    assert coercion is not None
    assert coercion.request.context_management is None

    # A budgeted config on its own is honored by the model and is not coerced.
    budgeted_only = adaptive_only.model_copy(
        update={
            "provider_thinking_config": {"type": "enabled", "budget_tokens": 1024},
            "context_management": None,
        }
    )
    assert coerce_generation_parameters((anthropic,), budgeted_only) is None

    # An adaptive-accepting reasoning route keeps the adaptive object verbatim.
    reasoning = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-sonnet-4-6",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    bare_adaptive = adaptive_only.model_copy(update={"context_management": None})
    assert coerce_generation_parameters((reasoning,), bare_adaptive) is None


def test_caller_budgeted_config_is_left_verbatim_on_a_budgeted_route() -> None:
    """A caller enabled+budget config needs no coercion; history is untouched."""
    request = _messages_request(
        messages=(
            GatewayMessage(role="user", content="hi"),
            _replayed_thinking_turn(),
            GatewayMessage(role="user", content="again"),
        ),
        provider_thinking_config={"type": "enabled", "budget_tokens": 2_048},
        maximum_output_tokens=8_000,
    )
    # An enabled config the route honors is not adaptive, so nothing coerces
    # and the signed history blocks survive for byte-exact replay.
    assert coerce_generation_parameters((_budgeted_haiku_profile(),), request) is None


def _tool_image_request(*, user_image: bool = False) -> GatewayRequest:
    """One request whose only (or not only) images live in a tool result."""
    from exp.common.models.content import ImageContentPart, TextContentPart

    messages: list[GatewayMessage] = [GatewayMessage(role="user", content="go")]
    if user_image:
        messages[0] = GatewayMessage(
            role="user",
            content="go",
            content_parts=(
                TextContentPart(text="go"),
                ImageContentPart(media_type="image/png", data="aGk="),
            ),
        )
    messages.append(
        GatewayMessage(
            role="tool",
            tool_call_id="call-1",
            content="tool said:",
            content_parts=(
                TextContentPart(text="tool said:"),
                ImageContentPart(media_type="image/png", data="aGk="),
            ),
        )
    )
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=tuple(messages),
    )


def test_tool_result_images_degrade_with_placeholder_on_a_non_vision_route() -> None:
    """A route-wide image_input rejection coerces when every image is a tool
    screenshot: the block is baked into history, so a rejection wedges the
    caller's whole session while a disclosed placeholder keeps it alive."""
    from exp.runtime.models.providers.streaming_requests import (
        TOOL_RESULT_IMAGE_DROP_DISCLOSURE,
        TOOL_RESULT_IMAGE_PLACEHOLDER,
    )

    coercion = coerce_capability("image_input", _tool_image_request())
    assert coercion is not None
    tool_message = coercion.request.messages[-1]
    assert tool_message.content_parts == ()
    assert tool_message.content == "tool said:" + TOOL_RESULT_IMAGE_PLACEHOLDER
    assert coercion.disclosures == (TOOL_RESULT_IMAGE_DROP_DISCLOSURE,)


def test_top_level_user_images_keep_the_fail_closed_contract() -> None:
    """A user image the caller can re-send never degrades silently."""
    assert coerce_capability("image_input", _tool_image_request(user_image=True)) is None
    # And a request with no images at all offers nothing to coerce.
    assert (
        coerce_capability(
            "image_input",
            GatewayRequest(
                surface=GatewayApiSurface.MESSAGES,
                messages=(GatewayMessage(role="user", content="go"),),
            ),
        )
        is None
    )
