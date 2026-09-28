# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for the local SQLite request-authority primitives."""

import sqlite3
from datetime import UTC, datetime

import pytest

from exp.runtime.gateway.auth import PepperKey
from exp.runtime.gateway.sqlite.request_authority import authenticate_sqlite_key
from exp.runtime.gateway.sqlite.store import InvalidVirtualKeyError


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return 0.0


def _unexpected_pepper(_version: int) -> PepperKey:
    raise AssertionError("malformed credentials must fail before pepper lookup")


def test_key_authentication_rejects_malformed_credentials_before_query() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(InvalidVirtualKeyError, match="virtual key is invalid"):
            authenticate_sqlite_key(
                connection,
                "not-a-key",
                clock=_Clock(),
                pepper_key=_unexpected_pepper,
                last_used_refresh_seconds=60.0,
                invalid_key_error=InvalidVirtualKeyError,
            )
    finally:
        connection.close()
