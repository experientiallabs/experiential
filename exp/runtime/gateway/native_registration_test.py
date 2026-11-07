"""Initial admission handoff preserves quantity bounds and exact lane authority."""

from __future__ import annotations

import json
import time
from pathlib import Path

from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _request,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_registration import registration_json
from exp.runtime.gateway.routing import GatewayRoute


def test_registration_carries_physical_bounds_without_resolving_prices(tmp_path: Path) -> None:
    """Physical dispatch receives the original card and each wire's output ceiling."""
    deployment = _deployment()
    clock = FakeLedgerClock()
    store, _ledger, key = _authority_fixture(tmp_path, clock)
    request = _request("hello").model_copy(update={"maximum_output_tokens": 50})
    authorization = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=clock.monotonic() + 60
    )
    snapshot = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=deployment.exact_model_id,
        pool_id="pool",
        deployment_ids=(deployment.deployment_id,),
    )
    entry = InflightRequest(
        authorization=authorization,
        route=GatewayRoute(snapshot=snapshot, deployment=deployment, route_reason="direct"),
        request=request,
        deadline_monotonic=time.monotonic() + 60,
        reserved_output_tokens_by_depth=(32,),
    )
    data = json.loads(registration_json(entry, bind_sticky=True))
    assert data["lanes"][0]["prices"] == deployment.gateway.prices.model_dump(mode="json")
    assert data["lanes"][0]["ceiling"]["output"] == 32
    assert data["lanes"][0]["output_ceiling"] == 32
    assert data["lanes"][0]["key"] == [
        authorization.catalog_sha256,
        deployment.deployment_id,
        deployment.connection_sha256,
    ]
    assert 0 < data["remaining_seconds"] <= 60
