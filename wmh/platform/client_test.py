"""Tests for the platform HTTP client (httpx mock transport, no network)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import httpx
import pytest

from wmh.platform.client import PlatformClient, PlatformError, fetch_cli_config

API_URL = "https://api.test"

_WHOAMI = {
    "actor": {"kind": "api_key", "id": "api-key:org-1"},
    "orgs": [{"id": "org-1", "slug": "acme", "name": "Acme"}],
    "projects": [{"id": "proj-1", "org_id": "org-1", "slug": "alpha", "name": "Alpha"}],
}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> PlatformClient:
    return PlatformClient(API_URL, "xpl_secret", transport=httpx.MockTransport(handler))


def test_whoami_parses_and_sends_bearer_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xpl_secret"
        assert request.url.path == "/api/whoami"
        return httpx.Response(200, json=_WHOAMI)

    with _client(handler) as client:
        identity = client.whoami()

    assert identity.actor.kind == "api_key"
    assert identity.projects[0].slug == "alpha"


def test_error_payloads_become_platform_errors_with_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Project not found: p"})

    with (
        _client(handler) as client,
        pytest.raises(PlatformError, match="Project not found") as info,
    ):
        client.whoami()
    assert info.value.status_code == 404


def test_401_error_suggests_logging_in() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    with _client(handler) as client, pytest.raises(PlatformError, match="wmh login"):
        client.whoami()


def test_push_model_bundle_sends_multipart_file_and_meta() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.read()
        return httpx.Response(201, json={"id": "wm-1", "name": "tau-bench", "status": "ready"})

    with _client(handler) as client:
        pushed = client.push_model_bundle(
            "proj-1", "tau-bench", b"bundle-bytes", {"serve_provider": "anthropic"}
        )

    assert pushed.status == "ready"
    assert "multipart/form-data" in str(seen["content_type"])
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b"bundle-bytes" in body
    assert b'"serve_provider": "anthropic"' in body


def test_download_model_bundle_verifies_declared_digest() -> None:
    content = b"bundle-bytes"

    def good(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=content, headers={"X-Bundle-Sha256": hashlib.sha256(content).hexdigest()}
        )

    with _client(good) as client:
        assert client.download_model_bundle("proj-1", "tau-bench") == content

    def corrupted(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"X-Bundle-Sha256": "0" * 64})

    with _client(corrupted) as client, pytest.raises(PlatformError, match="digest mismatch"):
        client.download_model_bundle("proj-1", "tau-bench")


def test_harness_round_trip_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            payload = json.loads(request.read())
            assert payload["doc_hash"] == "a" * 32
            return httpx.Response(
                201,
                json={"name": "agent", "version": 3, "doc_hash": "a" * 32, "created": True},
            )
        if request.url.path.endswith("/versions/3"):
            return httpx.Response(
                200, json={"version": 3, "doc": {"name": "agent"}, "doc_hash": "a" * 32}
            )
        return httpx.Response(
            200,
            json={
                "harness": {"id": "h-1", "name": "agent", "latest_version": 3},
                "versions": [{"version": 3, "doc_hash": "a" * 32}],
            },
        )

    with _client(handler) as client:
        pushed = client.push_harness_version("proj-1", "agent", {"name": "agent"}, "a" * 32)
        assert pushed.version == 3
        assert pushed.created

        harness, versions = client.get_harness("proj-1", "agent")
        assert harness.latest_version == 3
        assert versions[0].doc_hash == "a" * 32

        doc = client.get_harness_version("proj-1", "agent", 3)
        assert doc.doc == {"name": "agent"}


def test_fetch_cli_config_reads_the_api_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://platform.test/api/cli/config"
        return httpx.Response(200, json={"apiUrl": "https://api.test/"})

    api_url = fetch_cli_config("https://platform.test/", transport=httpx.MockTransport(handler))
    assert api_url == "https://api.test"


def test_fetch_cli_config_maps_failures() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(PlatformError, match="discovery failed"):
        fetch_cli_config("https://platform.test", transport=httpx.MockTransport(handler))
