"""Request-class throttles act only on self-hosted rungs and are inert by default."""

from __future__ import annotations

import json

import pytest

from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.class_throttles import (
    class_concurrency_cap,
    is_self_hosted_rung,
    output_pace,
    rung_saturation,
    started_attempt_payload,
)
from exp.runtime.gateway.native_execution import rung_load_key
from exp.runtime.gateway.native_rungs_test import _AUTHORIZATION, _deployment
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed

_PACED = _AUTHORIZATION.model_copy(update={"output_tokens_per_second": 40.0})


def _rung(
    deployment_id: str = "vllm",
    *,
    self_hosted: bool = True,
    bound: int | None = 4,
) -> ExactModelDeployment:
    """A rung whose dispatch policy marks it self-hosted (``upstream_priority``) or not."""
    base = _deployment(deployment_id, "experiential_cloud" if self_hosted else "openrouter")
    policy = GatewayRungDispatchPolicy(upstream_priority=self_hosted, concurrency_bound=bound)
    return base.model_copy(update={"gateway": base.gateway.model_copy(update={"dispatch": policy})})


def _hold(loads: RungLoadRegistry, rung: ExactModelDeployment, count: int) -> None:
    """Hold ``count`` in-flight reservations on ``rung``."""
    for index in range(count):
        ticket = loads.reserve(
            rung_load_key(rung),
            organization_id=f"org-{index}",
            weight=1,
            bound=None,
            fair_share=False,
        )
        assert isinstance(ticket, str)


def test_only_an_upstream_priority_rung_is_self_hosted() -> None:
    """The predicate is the operator's vLLM opt-in, the one the priority field already uses."""
    assert is_self_hosted_rung(_rung())
    assert not is_self_hosted_rung(_rung(self_hosted=False))
    assert not is_self_hosted_rung(_deployment("plain", "openrouter"))


def test_nothing_is_paced_or_capped_by_default() -> None:
    """An authorization with no class throttle leaves every rung, self-hosted or not, alone."""
    loads = RungLoadRegistry()
    for rung in (_rung(), _rung(self_hosted=False), _deployment("plain", "openrouter")):
        assert output_pace(_AUTHORIZATION, rung, loads) is None
        assert class_concurrency_cap(_AUTHORIZATION, rung) is None


@pytest.mark.parametrize("provider", ["openrouter", "anthropic", "customer-endpoint"])
def test_third_party_byok_and_tenant_rungs_are_never_paced_or_capped(provider: str) -> None:
    """A full class throttle is a hard no-op on any rung that is not ours."""
    authorization = _AUTHORIZATION.model_copy(
        update={
            "priority_class": 2,
            "output_tokens_per_second": 10.0,
            "self_hosted_class_concurrency": 1,
        }
    )
    rung = _deployment("theirs", provider)
    assert output_pace(authorization, rung, RungLoadRegistry()) is None
    assert class_concurrency_cap(authorization, rung) is None


def test_unconditional_pace_applies_on_a_self_hosted_rung() -> None:
    """With no saturation threshold the pace applies whatever the load."""
    assert output_pace(_PACED, _rung(), RungLoadRegistry()) == 40.0


def test_saturation_threshold_paces_only_under_contention() -> None:
    """At 0.75 of a bound of 4: unpaced at 2 in flight, paced from 3."""
    authorization = _PACED.model_copy(update={"output_pace_saturation": 0.75})
    rung = _rung(bound=4)
    loads = RungLoadRegistry()
    _hold(loads, rung, 2)
    assert rung_saturation(loads, rung) == 0.5
    assert output_pace(authorization, rung, loads) is None
    _hold(loads, rung, 1)
    assert output_pace(authorization, rung, loads) == 40.0


def test_threshold_pace_needs_a_bound_to_read_saturation() -> None:
    """An unbounded rung has no saturation signal, so a threshold pace never applies there."""
    authorization = _PACED.model_copy(update={"output_pace_saturation": 0.5})
    rung = _rung(bound=None)
    assert rung_saturation(RungLoadRegistry(), rung) is None
    assert output_pace(authorization, rung, RungLoadRegistry()) is None
    # The worker's default lane share is a bound too.
    loads = RungLoadRegistry(default_bound=2)
    _hold(loads, rung, 1)
    assert output_pace(authorization, rung, loads) == 40.0


def test_unpaced_start_reply_is_byte_identical() -> None:
    """No pace serializes exactly the reply every attempt sent before."""
    assert started_attempt_payload("attempt-1", 2, None) == json.dumps(
        {"attempt_id": "attempt-1", "route_depth": 2}, separators=(",", ":")
    )
    assert json.loads(started_attempt_payload("attempt-1", 0, 12.5)) == {
        "attempt_id": "attempt-1",
        "route_depth": 0,
        "output_tokens_per_second": 12.5,
    }


def test_class_cap_sheds_only_its_class_and_frees_on_release() -> None:
    """A capped class sheds as a hit ceiling (never force-admitted); other classes pass."""
    authorization = _AUTHORIZATION.model_copy(
        update={"priority_class": 1, "self_hosted_class_concurrency": 1}
    )
    rung = _rung()
    cap = class_concurrency_cap(authorization, rung)
    assert cap == (1, 1)
    loads = RungLoadRegistry()
    key = rung_load_key(rung)

    def reserve(class_cap: tuple[int, int] | None, *, force: bool = False) -> str | RungShed:
        return loads.reserve(
            key,
            organization_id="org",
            weight=1,
            bound=8,
            fair_share=False,
            class_cap=class_cap,
            force=force,
        )

    first = reserve(cap)
    assert isinstance(first, str)
    shed = reserve(cap, force=True)
    assert shed == RungShed("queue_bound", overflow_ceiling=True)
    assert isinstance(reserve((6, 1)), str)
    assert isinstance(reserve(None), str)
    loads.release_ticket(first)
    assert isinstance(reserve(cap), str)


def test_class_cap_needs_a_class() -> None:
    """A cap without a request class names no class to count, so it is not applied."""
    authorization = _AUTHORIZATION.model_copy(update={"self_hosted_class_concurrency": 1})
    assert class_concurrency_cap(authorization, _rung()) is None
