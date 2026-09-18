"""Focused tests for SQLite platform row conversion and replay checks."""

from pathlib import Path

import pytest

from exp.common.models.catalog import GatewayLongContextTier, GatewayTokenPrices
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.platform import AttemptReservationRequest
from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform


@pytest.mark.parametrize("long_context", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "cache_creation_input_nano_usd_per_million_tokens",
        "cache_creation_1h_input_nano_usd_per_million_tokens",
    ],
)
def test_reservation_replay_rejects_changed_frozen_cache_write_rate(
    tmp_path: Path, long_context: bool, field: str
) -> None:
    """Changing only one write price must not replay an existing reservation."""
    clock = FakeLedgerClock()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    authority = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=_request("hello"),
        deadline_monotonic=clock.monotonic() + 30,
    )
    ledger.accept_request(authorization=authority)
    prices = GatewayTokenPrices(
        cache_creation_input_nano_usd_per_million_tokens=3750,
        cache_creation_1h_input_nano_usd_per_million_tokens=6000,
        long_context=GatewayLongContextTier(
            input_threshold_tokens=200000,
            cache_creation_input_nano_usd_per_million_tokens=7500,
            cache_creation_1h_input_nano_usd_per_million_tokens=12000,
        ),
    )
    deployment = _deployment()
    deployment = deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": prices})}
    )
    request = AttemptReservationRequest(
        organization_id="org-one",
        snapshot=_execution(authority),
        deployment=deployment,
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=10000,
    )
    platform = SQLiteGatewayPlatform(tmp_path / "gateway.db", attempts=ledger)
    first = platform.reserve_attempt(request)
    assert platform.reserve_attempt(request) == first
    if long_context:
        assert prices.long_context is not None
        changed = prices.model_copy(
            update={"long_context": prices.long_context.model_copy(update={field: 1})}
        )
    else:
        changed = prices.model_copy(update={field: 1})
    altered = deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": changed})}
    )
    with pytest.raises(ValueError, match="differs from durable accounting input"):
        platform.reserve_attempt(request.model_copy(update={"deployment": altered}))
    assert platform.reserve_attempt(request) == first
