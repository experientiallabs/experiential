"""Tests for the tool-search round callback on the native control plane."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_tool_search import NativeToolSearchMixin, _Registry


class _Accounting:
    def __init__(self, entry: InflightRequest | None) -> None:
        self._entry = entry

    def entry(self, request_id: str) -> InflightRequest | None:
        return self._entry if request_id == "req" else None


class _Plane(NativeToolSearchMixin):
    def __init__(self, entry: InflightRequest | None) -> None:
        self._accounting: _Registry = _Accounting(entry)


def test_unknown_request_or_missing_state_is_a_protocol_error() -> None:
    plane = _Plane(None)
    with pytest.raises(NativeBridgeError):
        plane.tool_search_round(json.dumps({"request_id": "nope", "route_depth": 0, "calls": []}))

    entry = cast(
        "InflightRequest",
        SimpleNamespace(tool_search=None, resolved_wires=None, public_request=None),
    )
    plane = _Plane(entry)
    with pytest.raises(NativeBridgeError):
        plane.tool_search_round(json.dumps({"request_id": "req", "route_depth": 0, "calls": []}))
