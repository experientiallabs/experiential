"""Durable hosted attempt authority tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from wmo.common.models import BillingSource
from wmo.common.project import ProjectStage
from wmo.optimize.router.attempt_authority import (
    FileHostedAttemptAuthorityStore,
    HostedAttemptAuthorityError,
    HostedProviderHazard,
)
from wmo.optimize.router.spend import (
    ProviderSpendComponent,
    ProviderSpendEntry,
    ProviderSpendStatus,
)


def test_attempt_binding_is_write_once_for_project_and_exact_ceiling(tmp_path: Path) -> None:
    """One random authority cannot be replayed for another Project or authorization."""
    store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = store.create()

    first = store.bind(
        authority,
        project_id="project-a",
        ceiling_usd=Decimal("25.000000"),
    )
    replay = store.bind(
        authority,
        project_id="project-a",
        ceiling_usd=Decimal("25.000000"),
    )

    assert replay == first
    with pytest.raises(HostedAttemptAuthorityError, match="another Project"):
        store.bind(
            authority,
            project_id="project-b",
            ceiling_usd=Decimal("25.000000"),
        )
    with pytest.raises(HostedAttemptAuthorityError, match="another Project"):
        store.bind(
            authority,
            project_id="project-a",
            ceiling_usd=Decimal("25.000001"),
        )


def test_attempt_rejects_one_microunit_over_large_exact_ceiling(tmp_path: Path) -> None:
    """A reservation one numeric(20,6) unit over authorization fails closed."""
    store = FileHostedAttemptAuthorityStore(tmp_path / "authority")
    authority = store.create()
    store.bind(
        authority,
        project_id="project-a",
        ceiling_usd=Decimal("99999999999998.000000"),
    )

    with pytest.raises(HostedAttemptAuthorityError, match="exceeds"):
        store.begin(
            HostedProviderHazard(
                project_id="project-a",
                attempt_id=authority.attempt_id,
                authority_sha256=authority.authority_sha256,
                stage=ProjectStage.BUILDING_WORLD_MODEL,
                reservations=(
                    ProviderSpendEntry(
                        operation_id="provider-reservation-a",
                        component=ProviderSpendComponent.RETRIEVAL_EMBEDDING,
                        billing_source=BillingSource.HOST_MANAGED,
                        status=ProviderSpendStatus.RESERVED,
                        operation_count=1,
                        amount_usd=Decimal("99999999999998.000001"),
                    ),
                ),
            )
        )

    assert store.unresolved(authority) is None
