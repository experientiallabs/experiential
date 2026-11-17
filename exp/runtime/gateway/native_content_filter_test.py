"""Tests for the rungs a provider content-filter refusal may advance to."""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_content_filter import other_providers

_PROVIDERS = ("azure_openai", "openai", "azure_openai", "huawei")


def _failure(
    *, failure_class: GatewayFailureClass = GatewayFailureClass.REFUSAL, flagged: bool = True
) -> GatewayFailure:
    """Build one refusal as the data plane classifies it."""
    return GatewayFailure(
        failure_class=failure_class,
        safe_message="provider refused the request: content policy",
        failover_eligible=True,
        provider_content_filter=flagged,
    )


def test_only_rungs_of_another_provider_remain() -> None:
    """The same provider's filter refuses again, so its rungs are dropped."""
    assert other_providers((1, 2, 3), _failure(), 0, _PROVIDERS, refusal_failover=False) == (1, 3)


def test_every_other_failure_keeps_its_depths() -> None:
    """An unflagged refusal, a guardrail verdict, and an opted-in alias walk as before."""
    depths = (1, 2, 3)
    assert (
        other_providers(depths, _failure(flagged=False), 0, _PROVIDERS, refusal_failover=False)
        == depths
    )
    guardrail = _failure(failure_class=GatewayFailureClass.GUARDRAIL)
    assert other_providers(depths, guardrail, 0, _PROVIDERS, refusal_failover=False) == depths
    assert other_providers(depths, _failure(), 0, _PROVIDERS, refusal_failover=True) == depths
    assert other_providers(depths, _failure(), 0, (), refusal_failover=False) == depths


def test_a_rung_that_opted_into_the_refusal_keeps_its_opt_in() -> None:
    """A same-provider ``failover_only_on`` rung matched for the refusal stays."""
    rules = (None, None, frozenset({"refusal:content_policy"}), None)
    kept = other_providers((1, 2, 3), _failure(), 0, _PROVIDERS, False, rules)
    assert kept == (1, 2, 3)
