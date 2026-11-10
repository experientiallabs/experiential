"""Tests for probability admission and exact effort preservation."""

from dataclasses import replace

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.replay_identity import canonical_request_sha256, provider_replay_authority
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import coerce_generation_parameters
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_route_compat import (
    compatible_generation_parameter_profile_indexes,
)
from exp.runtime.models.providers.logprobs import (
    capture_logprobs_eligible,
    require_chat_logprobs,
    require_unmodified_probability_output,
)
from exp.runtime.models.providers.streaming_requests import (
    dialect_stream_payload,
    route_generation_parameter_requests,
)
from exp.runtime.models.providers.streaming_requests_test import _chat_request
from exp.runtime.openai_protocol.requests import decode_chat


def _profile() -> GatewayWireProfile:
    """Build an explicitly capable nonreasoning Chat deployment."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.invalid",
        headers={},
        model_id="test",
        supports_logprobs=True,
    )


def test_probability_narrowing_keeps_order_and_rejects_unknown_support() -> None:
    """Unsupported fallbacks are removed without changing the requested probabilities."""
    request = _chat_request().model_copy(update={"logprobs": True, "top_logprobs": 0})
    profiles = (_profile(), replace(_profile(), supports_logprobs=False), _profile())
    assert compatible_generation_parameter_profile_indexes(profiles, request) == (0, 2)
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((profiles[1],), request)


def test_reasoning_logprobs_require_explicit_effective_effort_support() -> None:
    """Unknown combinations reject, and a declared default counts when effort is omitted."""
    request = _chat_request().model_copy(update={"logprobs": True})
    profile = replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="low",
        supported_reasoning_efforts=("none", "low"),
    )
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((profile,), request)
    profile = replace(profile, logprobs_reasoning_efforts=("none",))
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((profile,), request)
    require_chat_logprobs((profile,), request.model_copy(update={"reasoning_effort": "none"}))
    assert (
        coerce_generation_parameters(
            (profile,), request.model_copy(update={"reasoning_effort": "high"})
        )
        is None
    )


def test_other_surface_and_count_without_opt_in_reject() -> None:
    """Direct canonical requests cannot bypass Chat-only and count-dependency gates."""
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs((_profile(),), _chat_request().model_copy(update={"top_logprobs": 0}))
    with pytest.raises(ProviderParameterError):
        require_chat_logprobs(
            (_profile(),),
            _chat_request().model_copy(
                update={"logprobs": True, "surface": GatewayApiSurface.RESPONSES}
            ),
        )


def test_output_rewriting_rejects_only_active_probability_requests() -> None:
    """Output transforms must not silently invalidate the requested token records."""
    request = _chat_request().model_copy(update={"logprobs": True})
    with pytest.raises(ProviderParameterError, match="output guardrails"):
        require_unmodified_probability_output(request, True)
    require_unmodified_probability_output(request, False)
    require_unmodified_probability_output(_chat_request(), True)
    responses_request = _chat_request().model_copy(
        update={
            "surface": GatewayApiSurface.RESPONSES,
            "include_output_text_logprobs": True,
        }
    )
    with pytest.raises(ProviderParameterError, match="Responses probabilities"):
        require_unmodified_probability_output(responses_request, True)


def test_qualified_optional_reasoning_default_remains_omitted_on_wire() -> None:
    """The provider default qualifies probabilities without becoming request intent."""
    profile = replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="none",
        reasoning_effort_required=False,
        supported_reasoning_efforts=("none", "low"),
        logprobs_reasoning_efforts=("none",),
    )
    request = _chat_request().model_copy(update={"logprobs": True})
    require_chat_logprobs((profile,), request)
    assert "reasoning_effort" not in dialect_stream_payload(profile, request)


@pytest.mark.parametrize("intent", [{"include_output_text_logprobs": True}, {"top_logprobs": 0}])
def test_responses_probabilities_never_coerce_reasoning_intent(
    intent: dict[str, bool | int],
) -> None:
    """Both probability selectors disable effort repair rather than altering the experiment."""
    profile = replace(
        _profile(),
        dialect="openai_responses",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        supported_reasoning_efforts=("low",),
        supports_responses_logprobs=True,
    )
    request = _chat_request().model_copy(
        update={"surface": GatewayApiSurface.RESPONSES, "reasoning_effort": "high", **intent}
    )
    assert coerce_generation_parameters((profile,), request) is None


@pytest.mark.parametrize("probabilities", [False, True])
def test_numeric_thinking_budget_cannot_borrow_nonreasoning_probability_qualification(
    probabilities: bool,
) -> None:
    """A numeric budget enables thinking, so a qualified none default cannot authorize it."""
    profile = replace(
        _profile(),
        url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        model_id="qwen3.8-max",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        reasoning_effort="none",
        supported_reasoning_efforts=("none", "low"),
        logprobs_reasoning_efforts=("none",),
    )
    request = decode_chat(
        {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 128,
            "max_output_tokens": 256,
            "logprobs": probabilities,
        }
    ).request
    assert request.reasoning_effort is None
    if probabilities:
        with pytest.raises(ProviderParameterError) as error:
            route_generation_parameter_requests((profile,), request)
        assert error.value.param == "thinking_budget"
        assert error.value.code == "unsupported_parameter"
        with pytest.raises(ProviderParameterError):
            require_chat_logprobs((profile,), request)
    else:
        public, provider = route_generation_parameter_requests((profile,), request)
        payload = dialect_stream_payload(profile, provider)
        assert public.thinking_budget == payload["thinking_budget"] == 128
        assert payload["enable_thinking"] is True
        assert "logprobs" not in payload
        assert "reasoning_effort" not in payload


@pytest.mark.parametrize("intent", [{"include_output_text_logprobs": True}, {"top_logprobs": 0}])
def test_native_responses_probabilities_cannot_carry_numeric_chat_budget(
    intent: dict[str, bool | int],
) -> None:
    """The existing native-wire budget guard refuses this unsupported Responses combination."""
    profile = replace(_profile(), dialect="openai_responses", supports_responses_logprobs=True)
    request = _chat_request().model_copy(
        update={"surface": GatewayApiSurface.RESPONSES, "thinking_budget": 128, **intent}
    )
    with pytest.raises(ProviderParameterError) as error:
        route_generation_parameter_requests((profile,), request)
    assert error.value.param == "thinking_budget"
    with pytest.raises(ProviderParameterError):
        dialect_stream_payload(profile, request)


def test_responses_probability_intent_uses_existing_optional_replay_authority() -> None:
    """Ordinary serialized requests stay unchanged while an active selector changes identity."""
    plain = _chat_request().model_copy(update={"surface": GatewayApiSurface.RESPONSES})
    selected = plain.model_copy(update={"include_output_text_logprobs": True})
    assert "include_output_text_logprobs" not in plain.model_dump()
    assert selected.model_dump() == plain.model_dump()
    assert provider_replay_authority(plain) is None
    assert canonical_request_sha256(plain) != canonical_request_sha256(selected)


def test_capture_eligibility_matches_explicit_support_without_a_caller_request() -> None:
    """Only a verified, platform-funded Chat wire is asked for probabilities nobody requested."""
    request = _chat_request()
    assert capture_logprobs_eligible(_profile(), request)
    assert not capture_logprobs_eligible(replace(_profile(), supports_logprobs=False), request)
    assert not capture_logprobs_eligible(
        replace(_profile(), billing_customer_managed=True), request
    )
    assert not capture_logprobs_eligible(
        replace(_profile(), dialect="openai_responses", supports_responses_logprobs=True), request
    )
    for update in ({"logprobs": True}, {"logprobs": False}, {"top_logprobs": 0}):
        assert not capture_logprobs_eligible(_profile(), request.model_copy(update=update))


def test_capture_eligibility_respects_qualified_reasoning_efforts() -> None:
    """A reasoning rung qualifies only at an effort its probabilities were verified for."""
    profile = replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="low",
        supported_reasoning_efforts=("none", "low"),
    )
    request = _chat_request()
    assert not capture_logprobs_eligible(profile, request)
    assert capture_logprobs_eligible(replace(profile, logprobs_reasoning_efforts=("low",)), request)
    assert not capture_logprobs_eligible(
        replace(profile, logprobs_reasoning_efforts=("low",)),
        request.model_copy(update={"reasoning_effort": "none"}),
    )


def test_capture_eligibility_honors_the_callers_explicit_false() -> None:
    """A cleared ``logprobs: false`` on the rung request still declines via the caller's."""
    rung_request = _chat_request()
    caller = rung_request.model_copy(update={"logprobs": False})
    assert capture_logprobs_eligible(_profile(), rung_request)
    assert not capture_logprobs_eligible(_profile(), rung_request, caller=caller)


def test_an_unqualified_effort_names_the_efforts_that_serve_probabilities() -> None:
    """A caller learns which reasoning_effort to send instead of a generic refusal."""
    request = _chat_request().model_copy(update={"logprobs": True, "top_logprobs": 5})
    profile = replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="max",
        supported_reasoning_efforts=("low", "high", "max"),
        logprobs_reasoning_efforts=("low", "high", "max"),
    )
    require_chat_logprobs((profile,), request)
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs(
            (profile,),
            request.model_copy(
                update={
                    "reasoning_effort": "medium",
                    "reasoning_effort_parameter": "reasoning_effort",
                }
            ),
        )
    assert caught.value.param == "reasoning_effort"
    assert "'medium'" in str(caught.value)
    assert "low, high, max" in str(caught.value)
    assert "remove logprobs and top_logprobs" in str(caught.value)
    assert "thinking on/off control set to agree" in str(caught.value)
    # No qualified effort at all keeps the route refusal on the probability field,
    # whether or not the caller sent an effort: changing it cannot admit the request.
    unqualified = replace(profile, logprobs_reasoning_efforts=())
    for effort in (None, "medium"):
        with pytest.raises(ProviderParameterError) as caught:
            require_chat_logprobs(
                (unqualified,), request.model_copy(update={"reasoning_effort": effort})
            )
        assert caught.value.param == "top_logprobs"
        assert "cannot preserve" in str(caught.value)


def test_a_nested_chat_effort_is_refused_under_its_own_spelling() -> None:
    """A Chat caller who sent ``reasoning.effort`` is pointed at that field, not the flat one."""
    profile = replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="max",
        supported_reasoning_efforts=("low", "medium", "max"),
        logprobs_reasoning_efforts=("low", "max"),
    )
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True,
    }
    nested = decode_chat({**base, "reasoning": {"effort": "medium"}}).request
    assert nested.reasoning_effort == "medium"
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((profile,), nested)
    assert caught.value.param == "reasoning.effort"
    flat = decode_chat({**base, "reasoning_effort": "medium"}).request
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((profile,), flat)
    assert caught.value.param == "reasoning_effort"
    translated = decode_chat({**base, "output_config": {"effort": "medium"}}).request
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((profile,), translated)
    assert caught.value.param == "output_config.effort"


def _reasoning_profile(efforts: tuple[str, ...], default: str = "max") -> GatewayWireProfile:
    """Build a probability-capable reasoning rung qualifying ``efforts``."""
    return replace(
        _profile(),
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort=default,
        supported_reasoning_efforts=("low", "medium", "high", "max"),
        logprobs_reasoning_efforts=efforts,
    )


def test_a_suggested_effort_admits_the_request_on_the_refusing_rung() -> None:
    """Narrowing reports the first rung's efforts, and sending one keeps that rung."""
    request = _chat_request().model_copy(
        update={
            "logprobs": True,
            "reasoning_effort": "medium",
            "reasoning_effort_parameter": "reasoning_effort",
        }
    )
    route = (_reasoning_profile(("low", "max")), _reasoning_profile(("high",)))
    with pytest.raises(ProviderParameterError) as caught:
        compatible_generation_parameter_profile_indexes(route, request)
    assert caught.value.param == "reasoning_effort"
    assert "one of: low, max (" in str(caught.value)
    retried = request.model_copy(update={"reasoning_effort": "low"})
    assert compatible_generation_parameter_profile_indexes(route, retried) == (0,)


def test_suggested_efforts_are_limited_to_the_rungs_effort_ladder() -> None:
    """A qualified effort the rung cannot carry is never suggested, since the retry would fail."""
    request = _chat_request().model_copy(
        update={
            "logprobs": True,
            "reasoning_effort": "medium",
            "reasoning_effort_parameter": "reasoning_effort",
        }
    )
    rung = replace(
        _reasoning_profile(("low", "ultra"), default="low"),
        supported_reasoning_efforts=("low", "medium"),
    )
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((rung,), request)
    assert "one of: low (" in str(caught.value)
    assert "ultra" not in str(caught.value)
    # No qualified effort on the ladder: the probability field is named.
    unreachable = replace(rung, logprobs_reasoning_efforts=("ultra",))
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((unreachable,), request)
    assert caught.value.param == "logprobs"
    assert "cannot preserve" in str(caught.value)


def test_an_effort_the_caller_did_not_name_keeps_the_probability_field() -> None:
    """A disable control or a gateway-resolved effort is not reported as a caller field."""
    profile = replace(
        _reasoning_profile(("low", "max")),
        supported_reasoning_efforts=("none", "low", "medium", "max"),
    )
    disabled = decode_chat(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "reasoning": {"enabled": False},
        }
    ).request
    assert disabled.reasoning_effort == "none"
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((profile,), disabled)
    assert caught.value.param == "logprobs"
    assert "resolves to ('none')" in str(caught.value)
    assert "in place of the other thinking controls" in str(caught.value)
    # A rung request whose effort the gateway resolved carries no caller field.
    shaped = _chat_request().model_copy(update={"logprobs": True, "reasoning_effort": "medium"})
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((profile,), shaped)
    assert caught.value.param == "logprobs"
    assert "reasoning_effort 'medium'" not in str(caught.value)


def test_an_incapable_later_rung_refuses_before_any_effort_advice() -> None:
    """Changing effort cannot admit a route whose later rung returns no probabilities."""
    request = _chat_request().model_copy(
        update={
            "logprobs": True,
            "reasoning_effort": "medium",
            "reasoning_effort_parameter": "reasoning_effort",
        }
    )
    route = (_reasoning_profile(("low", "max")), replace(_profile(), supports_logprobs=False))
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs(route, request)
    assert caught.value.param == "logprobs"
    assert "cannot preserve" in str(caught.value)


def test_an_omitted_effort_is_named_as_the_model_default() -> None:
    """A caller who sent no effort is not told it sent the default."""
    request = _chat_request().model_copy(update={"logprobs": True})
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((_reasoning_profile(("low",), default="max"),), request)
    assert caught.value.param == "logprobs"
    assert "the model's default reasoning effort 'max'" in str(caught.value)
    # A level-less on switch (enable_thinking=true) leaves no caller effort;
    # the advice must not let a kept switch contradict the suggested effort.
    assert "(with any thinking on/off control set to agree)" in str(caught.value)


def test_the_nested_effort_refusal_message_names_the_nested_field() -> None:
    """The message tells a ``reasoning.effort`` caller to change that field."""
    request = decode_chat(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "reasoning": {"effort": "medium"},
        }
    ).request
    with pytest.raises(ProviderParameterError) as caught:
        require_chat_logprobs((_reasoning_profile(("low", "max")),), request)
    assert "reasoning.effort 'medium'" in str(caught.value)
    assert "Use reasoning.effort one of: low, max" in str(caught.value)
