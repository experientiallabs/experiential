"""Tests for the normalized context connector types."""

from __future__ import annotations

import httpx
import pytest

from wmh.connect.types import (
    ConnectError,
    ConnectorAuth,
    ContextItem,
    ItemKind,
    PullQuery,
    transport_errors,
)


def test_item_kind_covers_the_normalized_content_kinds() -> None:
    assert ItemKind.PULL_REQUEST == "pull_request"
    assert {kind.value for kind in ItemKind} == {
        "document",
        "page",
        "issue",
        "pull_request",
        "message",
        "thread",
        "email",
        "event",
        "file",
    }


def test_context_item_defaults_and_json_round_trip() -> None:
    item = ContextItem(id="1", source="github", kind=ItemKind.ISSUE, title="Bug", body="It broke")
    assert item.url is None
    assert item.created_at is None and item.updated_at is None
    assert item.metadata == {}
    assert ContextItem.model_validate_json(item.model_dump_json()) == item


def test_pull_query_defaults_to_a_limit_of_100() -> None:
    query = PullQuery()
    assert query.limit == 100
    assert query.target is None and query.query is None
    assert query.since is None and query.until is None


def test_connector_auth_defaults() -> None:
    auth = ConnectorAuth(kind="token", access_token="tok")
    assert auth.refresh_token is None and auth.expires_at is None
    assert auth.scopes == [] and auth.account is None and auth.extra == {}


def test_connect_error_is_a_raisable_exception() -> None:
    with pytest.raises(ConnectError, match="broken"):
        raise ConnectError("broken; do the thing")


def test_transport_errors_maps_httpx_failures_to_actionable_connect_errors() -> None:
    request = httpx.Request("GET", "https://api.example.com/things")
    with pytest.raises(
        ConnectError,
        match=r"could not reach api\.example\.com \(connection timed out\); "
        r"check your network connection and retry",
    ):
        with transport_errors("api.example.com"):
            raise httpx.ConnectTimeout("connection timed out", request=request)


def test_transport_errors_passes_non_httpx_exceptions_through() -> None:
    with pytest.raises(ValueError, match="not transport"):
        with transport_errors("api.example.com"):
            raise ValueError("not transport")
