"""Tests for transactional SQLite gateway authority."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.runtime.gateway.contracts import (
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ProjectTarget,
)
from wmo.runtime.gateway.sqlite.store import (
    AliasNotGrantedError,
    InvalidVirtualKeyError,
    OperationConflictError,
    OperationReplayUnavailableError,
    SQLiteGatewayStore,
)

_DIGEST = "a" * 64


class FakeClock:
    """Controllable wall and monotonic clock for authority tests."""

    def __init__(self) -> None:
        """Initialize a fixed UTC instant and monotonic value."""
        self.wall = datetime(2026, 8, 18, 20, 0, tzinfo=UTC)
        self.monotonic_value = 100.0

    def now(self) -> datetime:
        """Return the controlled wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return the controlled monotonic time."""
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        """Advance both clocks by the same duration.

        Args:
            seconds: Positive elapsed seconds.
        """
        self.wall += timedelta(seconds=seconds)
        self.monotonic_value += seconds


def _request(content: str = "content-canary") -> GatewayRequest:
    """Create one canonical request with a canary that must not persist."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content=content),),
    )


def _configured_store(tmp_path: Path) -> tuple[SQLiteGatewayStore, FakeClock, str]:
    """Create explicit organization, identity, snapshot, alias, grant, and key state."""
    clock = FakeClock()
    store = SQLiteGatewayStore(tmp_path / "gateway.db", clock=clock)
    store.create_organization(
        organization_id="org-one", slug="one", display_name="Organization One"
    )
    store.create_identity(
        organization_id="org-one", identity_id="identity-one", display_name="Identity One"
    )
    store.register_catalog_snapshot(
        organization_id="org-one", snapshot_ref="snapshot-one", catalog_sha256=_DIGEST
    )
    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-coding",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-coding"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_DIGEST,
    )
    store.grant_alias(
        organization_id="org-one", identity_id="identity-one", alias_id="alias-coding"
    )
    issued = store.issue_virtual_key(
        organization_id="org-one", identity_id="identity-one", key_id="key-one"
    )
    return store, clock, issued.raw_key


def test_key_derived_authority_is_deny_by_default_and_revocation_is_immediate(
    tmp_path: Path,
) -> None:
    """A key derives authority, grants gate aliases, and revocation affects the next lookup."""
    store, clock, raw_key = _configured_store(tmp_path)

    snapshot = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=_request(),
        deadline_monotonic=clock.monotonic() + 30,
    )

    assert snapshot.organization_id == "org-one"
    assert snapshot.identity_id == "identity-one"
    assert snapshot.alias_revision_id == "revision-one"
    assert isinstance(snapshot.target, DirectTarget)
    assert store.granted_aliases(raw_key=raw_key) == ("coding",)

    store.revoke_alias_grant(
        organization_id="org-one", identity_id="identity-one", alias_id="alias-coding"
    )
    with pytest.raises(AliasNotGrantedError, match="not granted"):
        store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=_request(),
            deadline_monotonic=clock.monotonic() + 30,
        )

    store.grant_alias(
        organization_id="org-one", identity_id="identity-one", alias_id="alias-coding"
    )
    assert store.revoke_virtual_key(organization_id="org-one", key_id="key-one")
    with pytest.raises(InvalidVirtualKeyError, match="invalid"):
        store.granted_aliases(raw_key=raw_key)


def test_expiry_identity_disable_and_pepper_rotation_fail_closed(tmp_path: Path) -> None:
    """Expiry and identity state deny old keys while pepper rotation preserves fingerprints."""
    clock = FakeClock()
    store = SQLiteGatewayStore(tmp_path / "gateway.db", clock=clock)
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.create_identity(organization_id="org-one", identity_id="identity-one", display_name="One")
    expiring = store.issue_virtual_key(
        organization_id="org-one",
        identity_id="identity-one",
        key_id="key-expiring",
        expires_at=clock.now() + timedelta(seconds=5),
    )
    assert store.rotate_fingerprint_pepper() == 2
    assert store.granted_aliases(raw_key=expiring.raw_key) == ()
    clock.advance(5)
    with pytest.raises(InvalidVirtualKeyError):
        store.granted_aliases(raw_key=expiring.raw_key)

    active = store.issue_virtual_key(
        organization_id="org-one", identity_id="identity-one", key_id="key-active"
    )
    assert store.disable_identity(organization_id="org-one", identity_id="identity-one")
    with pytest.raises(InvalidVirtualKeyError):
        store.granted_aliases(raw_key=active.raw_key)


def test_operation_receipts_are_atomic_and_one_time_key_replay_stays_secret(
    tmp_path: Path,
) -> None:
    """Mutation retries converge while raw one-time key material never replays."""
    store = SQLiteGatewayStore(tmp_path / "gateway.db")
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    assert (
        store.create_identity(
            organization_id="org-one",
            identity_id="identity-one",
            display_name="Identity",
            operation_id="operation-identity",
        )
        == "identity-one"
    )
    assert (
        store.create_identity(
            organization_id="org-one",
            identity_id="identity-one",
            display_name="Identity",
            operation_id="operation-identity",
        )
        == "identity-one"
    )
    with pytest.raises(OperationConflictError, match="different input"):
        store.create_identity(
            organization_id="org-one",
            identity_id="identity-one",
            display_name="Changed",
            operation_id="operation-identity",
        )

    issued = store.issue_virtual_key(
        organization_id="org-one",
        identity_id="identity-one",
        key_id="key-one",
        operation_id="operation-key",
    )
    with pytest.raises(OperationReplayUnavailableError, match="cannot be revealed"):
        store.issue_virtual_key(
            organization_id="org-one",
            identity_id="identity-one",
            key_id="key-one",
            operation_id="operation-key",
        )

    durable = (tmp_path / "gateway.db").read_bytes()
    wal = tmp_path / "gateway.db-wal"
    if wal.exists():
        durable += wal.read_bytes()
    assert issued.raw_key.encode() not in durable


def test_project_activation_binding_is_unique_per_tenant_and_revisions_are_immutable(
    tmp_path: Path,
) -> None:
    """Database constraints prevent two active aliases for one project activation."""
    store = SQLiteGatewayStore(tmp_path / "gateway.db")
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.register_catalog_snapshot(
        organization_id="org-one", snapshot_ref="snapshot-one", catalog_sha256=_DIGEST
    )
    target = ProjectTarget(
        project_ref="project-one", activation_ref="activation-one", catalog_sha256=_DIGEST
    )
    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-one",
        alias_name="first",
        revision_id="revision-one",
        target=target,
        snapshot_ref="snapshot-one",
        catalog_sha256=_DIGEST,
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.activate_alias_revision(
            organization_id="org-one",
            alias_id="alias-two",
            alias_name="second",
            revision_id="revision-two",
            target=target,
            snapshot_ref="snapshot-one",
            catalog_sha256=_DIGEST,
        )

    store.activate_alias_revision(
        organization_id="org-one",
        alias_id="alias-one",
        alias_name="first",
        revision_id="revision-three",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_DIGEST,
    )
    connection = sqlite3.connect(tmp_path / "gateway.db")
    try:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM alias_revisions WHERE alias_id = 'alias-one'"
            ).fetchone()[0]
            == 2
        )
    finally:
        connection.close()


def test_cross_tenant_grant_is_rejected_by_composite_foreign_keys(tmp_path: Path) -> None:
    """An identity cannot be granted another tenant's alias through mismatched IDs."""
    store = SQLiteGatewayStore(tmp_path / "gateway.db")
    store.create_organization(organization_id="org-one", slug="one", display_name="One")
    store.create_organization(organization_id="org-two", slug="two", display_name="Two")
    store.create_identity(
        organization_id="org-one", identity_id="identity-one", display_name="Identity"
    )
    store.register_catalog_snapshot(
        organization_id="org-two", snapshot_ref="snapshot-two", catalog_sha256=_DIGEST
    )
    store.activate_alias_revision(
        organization_id="org-two",
        alias_id="alias-two",
        alias_name="two",
        revision_id="revision-two",
        target=DirectTarget(pool_id="pool-two"),
        snapshot_ref="snapshot-two",
        catalog_sha256=_DIGEST,
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.grant_alias(
            organization_id="org-one", identity_id="identity-one", alias_id="alias-two"
        )
