"""Native ownership is selected explicitly; a factory failure never changes ledgers."""

import time

import pytest

from exp.runtime.gateway.native_accounting_test import (
    _deployment,
    _RecordingLedger,
    _request,
    _route,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_handoff import (
    NativeRequestAccounting,
    accounting_drained,
    uses_native,
)


def _entry() -> InflightRequest:
    route = _route((_deployment("lane", connection_sha256="b" * 64),))
    return InflightRequest(
        authorization=route.snapshot.authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 60,
    )


def test_explicit_synchronous_selection_keeps_its_own_request_owner() -> None:
    accounting = NativeRequestAccounting(
        _RecordingLedger(), native_request_factory=lambda _: None, background_recovery=False
    )
    entry = _entry()
    accounting.register(entry)
    assert accounting.entry(entry.authorization.request_id) is entry
    assert not uses_native(accounting, entry.authorization.request_id)
    assert not accounting_drained(accounting)
    accounting.abandon('{"request_id":"request-one"}')
    assert accounting_drained(accounting)
    assert accounting.close()


def test_native_factory_failure_does_not_register_a_synchronous_request() -> None:
    def fail(_: InflightRequest) -> None:
        raise ValueError("owner unavailable")

    accounting = NativeRequestAccounting(
        _RecordingLedger(), native_request_factory=fail, background_recovery=False
    )
    entry = _entry()
    with pytest.raises(ValueError, match="owner unavailable"):
        accounting.register(entry)
    assert accounting.entry(entry.authorization.request_id) is None
    assert accounting_drained(accounting)
    assert accounting.close()


@pytest.mark.parametrize("fail_registration", [False, True])
def test_host_memos_release_only_after_native_registration(
    monkeypatch: pytest.MonkeyPatch, fail_registration: bool
) -> None:
    """Failed ownership transfer leaves the original acceptance available for cleanup."""
    from typing import cast
    from unittest.mock import Mock

    from exp_gateway_native import NativeCallbacks

    from exp.runtime.gateway.native_handoff import NativeRequestHandoff

    native = Mock()
    native.close.return_value = True
    if fail_registration:
        native.register.side_effect = ValueError("bad registration")
    release = Mock()
    target = NativeRequestHandoff(cast("NativeCallbacks", object()), release)
    accounting = NativeRequestAccounting(
        _RecordingLedger(), native_request_factory=lambda _: target, background_recovery=False
    )
    monkeypatch.setattr(accounting, "native", native)
    monkeypatch.setattr(
        "exp.runtime.gateway.native_handoff.registration_json", lambda *_a, **_k: "{}"
    )
    try:
        if fail_registration:
            with pytest.raises(ValueError, match="bad registration"):
                accounting.register(_entry())
            release.assert_not_called()
        else:
            accounting.register(_entry())
            native.register.assert_called_once()
            release.assert_called_once_with()
    finally:
        assert accounting.close()
