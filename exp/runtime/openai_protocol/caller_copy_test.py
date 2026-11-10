"""Tests for the caller wording of durable gateway failure sentences."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.openai_protocol.caller_copy import (
    CALLER_COPY_EXAMPLES,
    CALLER_COPY_PASSTHROUGH,
    caller_copy,
)
from exp.runtime.openai_protocol.errors import public_failure_error

_INTERNAL_WORDS = ("rung", "lane", "deployment", "alias", "exact-model", "native engine")


@pytest.mark.parametrize(("failure_class", "durable"), CALLER_COPY_EXAMPLES)
def test_every_known_sentence_is_reworded_without_internal_words(
    failure_class: GatewayFailureClass, durable: str
) -> None:
    text = caller_copy(failure_class, durable)
    assert text != durable
    assert text[0].isupper() and text.endswith(".")
    for word in _INTERNAL_WORDS:
        assert word not in text.replace("/v1/models/{model}", ""), (word, text)
    assert "\u2014" not in text and "\u2013" not in text and "\u2212" not in text


def test_the_public_error_carries_the_caller_wording_and_keeps_the_shape() -> None:
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
            safe_message="all exact-model deployments are unavailable",
        )
    )
    assert error.status_code == 502
    assert error.detail.code == "all_routes_failed"
    assert error.detail.type == "api_error"
    assert error.detail.message == (
        "No provider for this model can take the request right now. Retry after a short "
        "delay; if it keeps failing, choose another model."
    )


def test_the_throttle_window_names_its_wait_in_seconds() -> None:
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message=(
                "all exact-model deployments are inside a provider throttle window; retry in 30s"
            ),
            retry_after_seconds=30,
        )
    )
    assert error.detail.message == (
        "Every provider serving this model is rate limiting requests right now. "
        "Retry in 30 seconds."
    )
    assert error.retry_after_seconds == 30


def test_a_refusal_keeps_its_category_and_a_bare_one_names_none() -> None:
    assert caller_copy(
        GatewayFailureClass.REFUSAL, "provider refused the request: content policy"
    ) == (
        "The model's provider declined this request (content policy). Rephrase the request "
        "or choose another model."
    )
    assert caller_copy(GatewayFailureClass.REFUSAL, "provider refused the request") == (
        "The model's provider declined to answer this request. Rephrase the request or "
        "choose another model."
    )


def test_a_relayed_provider_detail_keeps_the_durable_head() -> None:
    error = public_failure_error(
        GatewayFailure(
            failure_class=GatewayFailureClass.INVALID_REQUEST,
            safe_message=(
                "provider rejected the request; verify the request fields against the model "
                "alias capabilities"
            ),
            provider_detail="max_tokens is too large",
        )
    )
    assert error.detail.message == "provider rejected the request: max_tokens is too large"


@pytest.mark.parametrize(("failure_class", "message"), CALLER_COPY_PASSTHROUGH)
def test_specific_or_unknown_sentences_pass_through(
    failure_class: GatewayFailureClass, message: str
) -> None:
    assert caller_copy(failure_class, message) == message


def test_a_signed_throttle_wait_is_not_a_wait() -> None:
    """Only plain ASCII digits name the wait, exactly as the native plane parses it."""
    text = caller_copy(
        GatewayFailureClass.THROTTLED,
        "all exact-model deployments are inside a provider throttle window; retry in +27s",
    )
    assert text.endswith("when the response carries one).")


def test_a_refusal_remediation_clause_is_not_a_category() -> None:
    text = caller_copy(
        GatewayFailureClass.REFUSAL,
        "provider refused the request; revise the request content and retry",
    )
    assert text == (
        "The model's provider declined to answer this request. Rephrase the request "
        "or choose another model."
    )
