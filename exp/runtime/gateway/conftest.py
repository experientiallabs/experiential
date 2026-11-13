"""Explicit initial-admission observation for gateway boundary tests."""

from collections.abc import Callable, Iterator

import pytest

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.tests.native_admission_fixture_test import AdmissionRecords
from exp.runtime.gateway.tests.native_classifier_fixture_test import close_classifier_fixtures


@pytest.fixture(autouse=True)
def native_classifier_lifetime() -> Iterator[None]:
    """Close every loopback classifier model started by this test's native gateway."""
    try:
        yield
    finally:
        close_classifier_fixtures()


@pytest.fixture
def admission_records(monkeypatch: pytest.MonkeyPatch) -> AdmissionRecords:
    """Observe immutable admission inputs while every financial operation remains native."""
    records = AdmissionRecords()
    register = NativeAttemptAccounting.register

    def record(
        self: NativeAttemptAccounting,
        entry: InflightRequest,
        *,
        prepare: Callable[[], None] | None = None,
    ) -> None:
        """Call the real ownership handoff before exposing its initial input to the test."""
        register(self, entry, prepare=prepare)
        records.remember(self, entry)

    monkeypatch.setattr(NativeAttemptAccounting, "register", record)
    return records
