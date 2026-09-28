# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for local gateway lifecycle component readiness."""

from typing import TYPE_CHECKING, cast

from exp.runtime.gateway.lifecycle_components import LocalGatewayComponents

if TYPE_CHECKING:
    from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger
    from exp.runtime.gateway.interfaces import GatewayControlStore
    from exp.runtime.gateway.ledger import SQLiteAttemptLedger
    from exp.runtime.gateway.lifecycle import _AliasAuthorityReloader
    from exp.runtime.gateway.management import GatewayManagement
    from exp.runtime.gateway.routing import CatalogRouteResolver, SelectionWorkerPool


class _Manager:
    organization_id = "org-local"


class _Writer:
    closed = False


def test_component_readiness_tracks_the_shared_writer() -> None:
    # The property uses only `closed`; lifecycle_test covers full component assembly.
    writer = _Writer()
    components = LocalGatewayComponents(
        manager=cast("GatewayManagement", _Manager()),
        store=cast("GatewayControlStore", object()),
        ledger=cast("SQLiteAttemptLedger", object()),
        write_ledger=cast("GroupCommitAttemptLedger", writer),
        routes=cast("CatalogRouteResolver", object()),
        reloader=cast("_AliasAuthorityReloader", object()),
        selection_workers=cast("SelectionWorkerPool", object()),
        reconciled_expired_requests=0,
        reconciled_unknown_attempts=0,
    )

    assert components.organization_id == "org-local"
    assert components.accounting_healthy
    writer.closed = True
    assert not components.accounting_healthy
