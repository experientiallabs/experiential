# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for the native accounting health mixin wiring."""

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_health import NativeAccountingHealthMixin


def test_attempt_accounting_uses_the_health_mixin_implementations() -> None:
    assert NativeAttemptAccounting._record_health is NativeAccountingHealthMixin._record_health
    assert (
        NativeAttemptAccounting._record_cache_fraction
        is NativeAccountingHealthMixin._record_cache_fraction
    )
