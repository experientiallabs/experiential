"""Native admission recovery retains exact content-free evidence across storage failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.sqlite.native import connect
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_handoff_test import _prepared
from exp.runtime.gateway.native_request_settlement import RequestSettlements


@pytest.mark.parametrize("searches", [0, 1, 3])
@pytest.mark.parametrize("cancelled", [False, True])
def test_request_terminal_and_certificate_commit_through_rust(
    tmp_path: Path, searches: int, cancelled: bool
) -> None:
    """Only an unsent failed request without paid prework receives a no-effects certificate."""
    factory, entry, ledger = _prepared(tmp_path)
    owner = RequestSettlements(factory)
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.CANCELLED
        if cancelled
        else GatewayFailureClass.INVALID_REQUEST,
        safe_message="synthetic admission failure",
    )
    try:
        assert owner.finish(
            entry.authorization, failure, certify_no_effects=True, web_search_requests=searches
        )
        assert not owner.pending and not owner.outstanding
        with connect(ledger.database_path) as connection:
            assert connection.execute(
                "SELECT terminal_state,web_search_requests,failed_without_effects "
                "FROM gateway_requests"
            ).fetchall() == [
                (
                    "cancelled" if cancelled else "failed",
                    searches,
                    int(not cancelled and not searches),
                )
            ]
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (0,)
    finally:
        assert owner.close()


@pytest.mark.parametrize("searches", [0, 2])
def test_storage_failure_keeps_original_admission_until_shutdown_acknowledges_it(
    tmp_path: Path, searches: int
) -> None:
    """A native retry survives failed close and cannot replace the original financial evidence."""
    factory, entry, ledger = _prepared(tmp_path)
    owner = RequestSettlements(factory)
    failure = GatewayFailure(
        failure_class=GatewayFailureClass.INVALID_REQUEST, safe_message="initial refusal"
    )
    with connect(ledger.database_path) as connection:
        connection.execute(
            "CREATE TRIGGER unavailable BEFORE UPDATE ON gateway_requests "
            "BEGIN SELECT RAISE(ABORT,'fixture unavailable'); END"
        )
    try:
        assert not owner.finish(
            entry.authorization, failure, certify_no_effects=True, web_search_requests=searches
        )
        assert owner.pending and owner.outstanding
        assert owner.fenced(entry.authorization.organization_id)
        assert owner.fenced_organizations() == frozenset({entry.authorization.organization_id})
        assert not owner.close(0.01)
    finally:
        with connect(ledger.database_path) as connection:
            connection.execute("DROP TRIGGER unavailable")
    assert owner.close(2)
    assert not owner.pending and not owner.outstanding
    with connect(ledger.database_path) as connection:
        assert connection.execute(
            "SELECT terminal_state,web_search_requests,failed_without_effects FROM gateway_requests"
        ).fetchall() == [("failed", searches, int(not searches))]
