"""Request-class throttles act only on self-hosted rungs and are inert by default."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from exp.common.models.catalog import BillingSource
from exp.common.models.catalog_prices import GatewayServiceTierPrices, GatewayTokenPrices
from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.class_throttles import (
    FAST_PRIORITY_CLASS,
    FAST_UPSTREAM_PRIORITY,
    TOP_PRIORITY_CLASS,
    admission_priority_class,
    class_concurrency_cap,
    fast_on_rung,
    fast_request,
    fast_reserve_slots,
    is_self_hosted_rung,
    output_pace,
    require_fast_path,
    rung_saturation,
    started_attempt_payload,
    tier_authorization,
    tier_settlement,
    upstream_priority,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_execution import InflightRequest, rung_load_key
from exp.runtime.gateway.native_rungs_test import _AUTHORIZATION, _deployment
from exp.runtime.gateway.native_service_tiers import settlement_kwarg
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.service_tiers import GatewayServiceTierAdmission
from exp.runtime.models.providers.errors import ProviderCapabilityError

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
    return base.model_copy(
        update={
            "billing_source": BillingSource.HOST_MANAGED,
            "gateway": base.gateway.model_copy(update={"dispatch": policy}),
        }
    )


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


def test_a_customer_billed_rung_is_never_self_hosted() -> None:
    """A customer's own key or endpoint stays a hard no-op even if a policy marks it."""
    customer = _rung().model_copy(update={"billing_source": BillingSource.CUSTOMER_MANAGED})
    assert not is_self_hosted_rung(customer)


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


_FAST = _AUTHORIZATION.model_copy(update={"fast": True})
_STANDARD = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=1_000, output_nano_usd_per_million_tokens=2_000
)
_CARDED = _STANDARD.model_copy(
    update={
        "priority": GatewayServiceTierPrices(
            input_nano_usd_per_million_tokens=2_000, output_nano_usd_per_million_tokens=4_000
        )
    }
)


def _carded(rung: ExactModelDeployment) -> ExactModelDeployment:
    """The rung with a priority card (2x), the price every fast lever comes with."""
    return rung.model_copy(update={"gateway": rung.gateway.model_copy(update={"prices": _CARDED})})


def test_fast_is_off_by_default_and_orders_above_every_class() -> None:
    """No fast flag: the admission class is the host's class; :fast sits above the top class."""
    assert _AUTHORIZATION.fast is False
    assert admission_priority_class(_AUTHORIZATION) is None
    assert admission_priority_class(_AUTHORIZATION.model_copy(update={"priority_class": 8})) == 8
    assert admission_priority_class(_FAST) == FAST_PRIORITY_CLASS == TOP_PRIORITY_CLASS + 1


def test_upstream_priority_puts_fast_ahead_only_with_the_priority_card() -> None:
    """vLLM runs the lowest value first: fast -1 on a carded rung, else the class or level map."""
    by_level = {2: 0, 1: 1, 0: 2}
    assert upstream_priority(_FAST, _carded(_rung()), by_level) == FAST_UPSTREAM_PRIORITY == -1
    # Without the card a fast request gets no lever at the standard price.
    assert upstream_priority(_FAST, _rung(), by_level) == 2
    classed = _AUTHORIZATION.model_copy(update={"priority_class": 8})
    assert upstream_priority(classed, _carded(_rung()), by_level) == 0
    assert upstream_priority(_AUTHORIZATION, _carded(_rung()), by_level) == 2


def test_fast_is_exempt_from_the_class_pace_and_cap() -> None:
    """A :fast request on our carded GPU is never paced or class-capped.

    On an uncarded GPU it is billed at standard, so it keeps every throttle.
    """
    throttled = _FAST.model_copy(
        update={
            "priority_class": 1,
            "output_tokens_per_second": 10.0,
            "self_hosted_class_concurrency": 1,
        }
    )
    assert output_pace(throttled, _carded(_rung()), RungLoadRegistry()) is None
    assert class_concurrency_cap(throttled, _carded(_rung())) is None
    assert output_pace(throttled, _rung(), RungLoadRegistry()) == 10.0
    assert class_concurrency_cap(throttled, _rung()) == (1, 1)


def test_fast_resource_levers_apply_only_on_our_carded_gpus() -> None:
    """fast_on_rung needs a self-hosted rung AND its priority card."""
    assert fast_on_rung(_FAST, _carded(_rung()))
    assert not fast_on_rung(_FAST, _rung())
    assert not fast_on_rung(_FAST, _carded(_rung(self_hosted=False)))
    assert not fast_on_rung(_AUTHORIZATION, _carded(_rung()))


def test_fast_requests_the_priority_tier_and_refuses_without_one() -> None:
    """:fast sets service_tier=priority; admission that dropped it refuses, never bills standard."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
    )
    assert fast_request(request, _AUTHORIZATION) is request
    assert fast_request(request, _FAST).service_tier == "priority"
    require_fast_path(request, _AUTHORIZATION)
    require_fast_path(fast_request(request, _FAST), _FAST)
    with pytest.raises(ProviderCapabilityError) as refused:
        require_fast_path(request, _FAST)
    assert refused.value.capability == "service_tier"


def _reserve(
    loads: RungLoadRegistry,
    *,
    fast: bool = False,
    force: bool = False,
    ceiling: float | None = None,
) -> str | RungShed:
    return loads.reserve(
        rung_load_key(_rung()),
        organization_id="org",
        weight=1,
        bound=4,
        fair_share=False,
        fast=fast,
        force=force,
        overflow_ceiling=ceiling,
    )


def test_fast_splits_the_bound_while_in_flight() -> None:
    """Fast holds the upper half (2 of 4); while it is held others hold the lower half (2)."""
    loads = RungLoadRegistry()
    assert isinstance(_reserve(loads), str)
    fast_ticket = _reserve(loads, fast=True)
    assert isinstance(fast_ticket, str)
    assert isinstance(_reserve(loads), str)
    # Two ordinary held: the lower half is full, even for a forced priority overflow.
    shed = _reserve(loads, force=True, ceiling=6.0)
    assert shed == RungShed("queue_bound", overflow_ceiling=True)
    assert isinstance(_reserve(loads, fast=True), str)
    # The fast half is full too: a third fast request sheds.
    assert _reserve(loads, fast=True) == RungShed("queue_bound", overflow_ceiling=True)
    loads.release_ticket(fast_ticket)
    assert isinstance(_reserve(loads, fast=True), str)


def test_fast_found_a_full_rung_and_ordinary_traffic_returns_when_it_leaves() -> None:
    """A fast request admits on a full rung; others wait for the lower half, then regain all."""
    loads = RungLoadRegistry()
    tickets = [_reserve(loads) for _ in range(4)]
    assert all(isinstance(ticket, str) for ticket in tickets)
    fast_ticket = _reserve(loads, fast=True)
    assert isinstance(fast_ticket, str)
    loads.release_ticket(str(tickets.pop()))
    assert isinstance(_reserve(loads), RungShed)  # 3 ordinary held >= the lower half (2)
    loads.bind(fast_ticket, "attempt-fast")
    loads.release_attempt("attempt-fast")
    assert isinstance(_reserve(loads), str)  # no fast left: the whole bound again


def _reserved(rung: ExactModelDeployment, fraction: float | None = 0.25) -> ExactModelDeployment:
    """The carded rung with an always-on :fast reserve of ``fraction``."""
    policy = rung.gateway.dispatch
    assert policy is not None
    return rung.model_copy(
        update={
            "gateway": rung.gateway.model_copy(
                update={
                    "dispatch": policy.model_copy(update={"fast_reserve_fraction": fraction}),
                    "prices": _CARDED,
                }
            )
        }
    )


def test_fast_reserve_slots_need_our_carded_gpu_and_a_bound_of_two() -> None:
    """ceil(fraction * bound), at least one, never the whole bound; else no reserve."""
    assert fast_reserve_slots(_reserved(_rung(bound=48))) == 12
    assert fast_reserve_slots(_reserved(_rung(bound=4))) == 1
    assert fast_reserve_slots(_reserved(_rung(bound=4), 0.9)) == 3
    assert fast_reserve_slots(_reserved(_rung(bound=2), 0.75)) == 1
    assert fast_reserve_slots(_reserved(_rung(bound=1))) is None
    assert fast_reserve_slots(_reserved(_rung(bound=48), None)) is None
    assert fast_reserve_slots(_reserved(_rung(bound=48, self_hosted=False))) is None
    uncarded = _reserved(_rung(bound=48))
    uncarded = uncarded.model_copy(
        update={"gateway": uncarded.gateway.model_copy(update={"prices": _STANDARD})}
    )
    assert fast_reserve_slots(uncarded) is None


def test_the_reserve_requires_a_bound_to_reserve_from() -> None:
    """An unbounded rung has nothing to keep a share of, so authoring it fails closed."""
    with pytest.raises(ValueError, match="fast_reserve_fraction requires a concurrency_bound"):
        GatewayRungDispatchPolicy(fast_reserve_fraction=0.25)


def _reserve_kept(
    loads: RungLoadRegistry, *, fast: bool = False, force: bool = False
) -> str | RungShed:
    return loads.reserve(
        rung_load_key(_rung()),
        organization_id="org",
        weight=1,
        bound=4,
        fair_share=False,
        fast=fast,
        force=force,
        overflow_ceiling=6.0 if force else None,
        fast_reserve=1,
    )


def test_the_reserve_holds_others_below_it_with_no_fast_in_flight() -> None:
    """Others hold 3 of 4 always, even a forced priority overflow; fast fills to the bound."""
    loads = RungLoadRegistry()
    ordinary = [_reserve_kept(loads) for _ in range(3)]
    assert all(isinstance(ticket, str) for ticket in ordinary)
    assert _reserve_kept(loads) == RungShed("queue_bound", overflow_ceiling=True)
    assert _reserve_kept(loads, force=True) == RungShed("queue_bound", overflow_ceiling=True)
    assert isinstance(_reserve_kept(loads, fast=True), str)
    # The bound is full: fast never passes it (no split overflow).
    assert _reserve_kept(loads, fast=True) == RungShed("queue_bound", overflow_ceiling=True)


def test_fast_may_use_the_whole_bound_and_others_return_as_it_drains() -> None:
    """An idle rung admits fast up to the bound; others then wait for room below the reserve."""
    loads = RungLoadRegistry()
    fast = [_reserve_kept(loads, fast=True) for _ in range(4)]
    assert all(isinstance(ticket, str) for ticket in fast)
    assert _reserve_kept(loads, fast=True) == RungShed("queue_bound", overflow_ceiling=True)
    assert isinstance(_reserve_kept(loads), RungShed)  # the bound is full
    loads.release_ticket(str(fast[0]))
    assert isinstance(_reserve_kept(loads), str)


def test_self_served_tier_settles_confirmed_only_on_our_gpu() -> None:
    """Our server reports no tier: missing evidence confirms the admitted card there only."""
    admission = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=_STANDARD,
        requested_prices=_CARDED.for_service_tier("priority"),
    )
    served = settlement_kwarg(admission, {}, self_served=True)["service_tier"]
    assert served.resolution == "confirmed"
    assert served.prices == admission.requested_prices
    # A third-party rung's missing evidence still holds billing, exactly as before.
    held = settlement_kwarg(admission, {})["service_tier"]
    assert held.resolution == "missing"
    assert held.prices is None
    # Explicit evidence always wins over the self-served default.
    downgraded = settlement_kwarg(
        admission,
        {"service_tier": {"served": "default", "resolution": "confirmed"}},
        self_served=True,
    )["service_tier"]
    assert downgraded.prices == admission.standard_prices


def _entry(authorization: AuthorizationSnapshot, rung: ExactModelDeployment) -> InflightRequest:
    """The two fields tier_settlement reads, on one reserved attempt at depth 0."""
    admission = GatewayServiceTierAdmission(
        requested="priority",
        standard_prices=_STANDARD,
        requested_prices=_CARDED.for_service_tier("priority"),
    )
    return cast(
        "InflightRequest",
        SimpleNamespace(
            authorization=authorization,
            route=SimpleNamespace(deployments=(rung,)),
            attempt_service_tiers={"attempt-1": admission},
            attempt_depths={"attempt-1": 0},
        ),
    )


def test_tier_settlement_self_serves_only_a_fast_request_on_our_carded_gpu() -> None:
    """A non-fast entry keeps the evidence rule; so does a third-party rung.

    An admitted explicit priority tier reaches here as fast (tier_authorization).
    """
    assert tier_settlement(_entry(_FAST, _carded(_rung())), "attempt-1")["self_served"]
    assert not tier_settlement(_entry(_AUTHORIZATION, _carded(_rung())), "attempt-1")["self_served"]
    assert not tier_settlement(_entry(_FAST, _carded(_rung(self_hosted=False))), "attempt-1")[
        "self_served"
    ]
    untiered = tier_settlement(_entry(_FAST, _carded(_rung())), "attempt-unknown")
    assert untiered == {"service_tier": None, "self_served": False}


def test_an_admitted_explicit_priority_tier_is_fast() -> None:
    """``service_tier`` priority (or fast, its alias) left standing by admission is ``:fast``.

    Without it, our GPU (no tier evidence) would hold the request at the card
    and never settle; with it, the request runs fast and settles at the card.
    """
    plain = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
    )
    assert tier_authorization(_AUTHORIZATION, plain) is _AUTHORIZATION
    flex = plain.model_copy(update={"service_tier": "flex"})
    assert tier_authorization(_AUTHORIZATION, flex) is _AUTHORIZATION
    # The wire models normalize ``fast`` to ``priority`` (service_tiers.ServiceTier).
    tiered = plain.model_copy(update={"service_tier": "priority"})
    assert tier_authorization(_AUTHORIZATION, tiered).fast
    assert tier_authorization(_FAST, plain) is _FAST
