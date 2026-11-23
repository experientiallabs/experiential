"""Tests for the video bridge methods' disabled lane and argument handling.

The served lifecycle (create through settlement, ownership, delete) is covered
end to end in ``exp/runtime/gateway/tests/native_videos_test.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_videos import NativeVideosMixin
from exp.runtime.gateway.video.providers import (
    VEO_DIALECT,
    XAI_VIDEOS_DIALECT,
    VeoVideoClient,
    XaiVideoClient,
    default_http_factory,
    public_http_factory,
)


class _Ledger:
    """Expose the required terminal signature without permitting fixture accounting."""

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Reject any write from these argument-validation-only tests."""
        raise AssertionError("argument validation must not reach accounting")


class _Plane(NativeVideosMixin):
    """A bare plane exposing only what the video methods read."""

    def __init__(self, video_jobs: object | None, *, public_upstreams_only: bool = False) -> None:
        """Bind a components stub with an optional video job store."""
        self._public_upstreams_only = public_upstreams_only
        self._components = cast(Any, SimpleNamespace(video_jobs=video_jobs))
        self._write_ledger = cast(Any, _Ledger())
        self._guardrails = None
        self._bind_videos(start_poller=False)


def _status(error: NativeBridgeError) -> tuple[int, str | None]:
    """Read the status and param of one bridge error."""
    public = json.loads(error.public_error_json)
    return public["status_code"], public["param"]


@pytest.mark.parametrize(
    "method", ["video_create", "video_retrieve", "video_list", "video_delete", "video_content"]
)
def test_every_route_answers_404_without_a_job_store(method: str) -> None:
    """A composition with no video job store never builds an engine."""
    plane = _Plane(None)
    assert plane._videos is None
    with pytest.raises(NativeBridgeError) as refused:
        getattr(plane, method)(json.dumps({"bearer_key": "k", "video_id": "video_x"}))
    assert _status(refused.value)[0] == 404


def test_a_store_builds_the_engine() -> None:
    """Installing a job store enables the lane."""
    assert _Plane(cast(Any, object()))._videos is not None


def test_create_requires_an_object_body() -> None:
    """Fields that are not an object answer 400 before any authority call."""
    plane = _Plane(cast(Any, object()))
    with pytest.raises(NativeBridgeError) as refused:
        plane.video_create(json.dumps({"bearer_key": "k", "fields": []}))
    assert _status(refused.value)[0] == 400


def test_a_missing_bearer_key_is_named() -> None:
    """An absent key argument is a 400 on its name."""
    plane = _Plane(cast(Any, object()))
    with pytest.raises(NativeBridgeError) as refused:
        plane.video_create(json.dumps({"fields": {}}))
    assert _status(refused.value) == (400, "bearer_key")


@pytest.mark.parametrize(
    ("public_only", "factory"), [(False, default_http_factory), (True, public_http_factory)]
)
def test_the_plane_egress_policy_selects_the_video_job_client(
    public_only: bool, factory: object
) -> None:
    """A public-only plane submits and polls video jobs through the public-only client."""
    plane = _Plane(object(), public_upstreams_only=public_only)
    assert plane.public_upstreams_only is public_only
    assert plane._videos is not None
    clients = plane._videos._clients
    assert {VEO_DIALECT, XAI_VIDEOS_DIALECT} <= set(clients)
    for client in clients.values():
        assert isinstance(client, VeoVideoClient | XaiVideoClient)
        assert client._http is factory
