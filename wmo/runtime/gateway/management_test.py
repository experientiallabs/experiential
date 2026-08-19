"""Tests for content-free gateway management status reads."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.gateway.sqlite.store import SQLiteGatewayStore


class _FixedClock:
    """Injectable clock pinned to one aware instant."""

    def __init__(self, wall: datetime) -> None:
        """Bind one fixed wall-clock instant.

        Args:
            wall: Timezone-aware instant returned by every read.
        """
        self.wall = wall

    def now(self) -> datetime:
        """Return the fixed wall time."""
        return self.wall

    def monotonic(self) -> float:
        """Return a fixed monotonic time."""
        return 1_000.0


def test_status_counts_active_keys_by_utc_expiry_in_any_host_timezone(tmp_path: Path) -> None:
    """An expired key stays out of active_keys even on a negative-offset host."""
    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    try:
        manager = GatewayManagement(tmp_path)
        manager.initialize()
        manager.create_identity(identity_id="identity-one", display_name="Identity")
        real_now = datetime.now(UTC)
        backdated = SQLiteGatewayStore(
            manager.database_path,
            clock=_FixedClock(real_now - timedelta(hours=4)),
        )
        backdated.issue_virtual_key(
            organization_id=manager.organization_id,
            identity_id="identity-one",
            key_id="expired-key",
            expires_at=real_now - timedelta(hours=2),
        )
        manager.issue_key(
            identity_id="identity-one",
            key_id="active-key",
            expires_at=real_now + timedelta(hours=2),
        )

        assert manager.status().active_keys == 1
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()
