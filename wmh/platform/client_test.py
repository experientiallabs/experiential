"""Tests for the platform HTTP client (httpx mock transport, no network)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from wmh.platform.client import PlatformClient, PlatformError, fetch_cli_config

API_URL = "https://api.test"

_WHOAMI = {
    "actor": {"kind": "api_key", "id": "api-key:org-1"},
    "orgs": [{"id": "org-1", "slug": "acme", "name": "Acme"}],
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
    assert identity.orgs[0].slug == "acme"


def test_error_payloads_become_platform_errors_with_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Organization not found: o"})

    with (
        _client(handler) as client,
        pytest.raises(PlatformError, match="Organization not found") as info,
    ):
        client.whoami()
    assert info.value.status_code == 404


def test_401_error_suggests_logging_in() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    with _client(handler) as client, pytest.raises(PlatformError, match="wmh login"):
        client.whoami()


def test_push_model_bundle_runs_ticket_put_finalize(tmp_path: Path) -> None:
    bundle_path = tmp_path / "tau-bench.tar.gz"
    bundle_path.write_bytes(b"bundle-bytes")
    digest = hashlib.sha256(b"bundle-bytes").hexdigest()
    seen: dict[str, object] = {}
    finalize_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/orgs/org-1/world-models/tau-bench/bundle/uploads":
            return httpx.Response(
                201,
                json={
                    "upload_url": "https://storage.test/upload/staging/cli/abc.tar.gz?token=t",
                    "token": "t",
                    "staging_path": "staging/cli/abc.tar.gz",
                },
            )
        if request.url.host == "storage.test":
            seen["put_body"] = request.read()
            seen["put_method"] = request.method
            return httpx.Response(200, json={"Key": "abc"})
        assert request.url.path == "/api/orgs/org-1/world-models/tau-bench/bundle"
        finalize_body.update(json.loads(request.read()))
        return httpx.Response(201, json={"id": "wm-1", "name": "tau-bench", "status": "ready"})

    with _client(handler) as client:
        pushed = client.push_model_bundle(
            "org-1",
            "tau-bench",
            bundle_path,
            digest,
            len(b"bundle-bytes"),
            {"serve_provider": "anthropic"},
        )

    assert pushed.status == "ready"
    assert seen["put_method"] == "PUT"
    assert seen["put_body"] == b"bundle-bytes"
    assert finalize_body["staging_path"] == "staging/cli/abc.tar.gz"
    assert finalize_body["sha256"] == digest
    assert finalize_body["meta"] == {"serve_provider": "anthropic"}


def test_push_model_bundle_surfaces_storage_upload_failure(tmp_path: Path) -> None:
    bundle_path = tmp_path / "tau-bench.tar.gz"
    bundle_path.write_bytes(b"bundle-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bundle/uploads"):
            return httpx.Response(
                201,
                json={
                    "upload_url": "https://storage.test/upload/x?token=t",
                    "token": "t",
                    "staging_path": "staging/cli/x.tar.gz",
                },
            )
        return httpx.Response(413, text="Payload too large")

    with (
        _client(handler) as client,
        pytest.raises(PlatformError, match="upload to storage failed"),
    ):
        client.push_model_bundle("org-1", "tau-bench", bundle_path, "0" * 64, 12, {})


def _download_handler(
    content: bytes, declared_sha256: str
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=content)
        assert request.url.path == "/api/orgs/org-1/world-models/tau-bench/bundle"
        return httpx.Response(
            200,
            json={
                "url": "https://storage.test/signed/bundle.tar.gz?token=t",
                "sha256": declared_sha256,
                "byte_size": len(content),
                "artifact_id": "artifact-1",
                "expires_in": 600,
            },
        )

    return handler


def test_download_model_bundle_streams_and_verifies_digest(tmp_path: Path) -> None:
    content = b"bundle-bytes"
    dest = tmp_path / "tau-bench.tar.gz"

    handler = _download_handler(content, hashlib.sha256(content).hexdigest())
    with _client(handler) as client:
        digest = client.download_model_bundle("org-1", "tau-bench", dest)

    assert dest.read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()


def test_download_model_bundle_rejects_digest_mismatch(tmp_path: Path) -> None:
    dest = tmp_path / "tau-bench.tar.gz"
    handler = _download_handler(b"bundle-bytes", "0" * 64)
    with _client(handler) as client, pytest.raises(PlatformError, match="digest mismatch"):
        client.download_model_bundle("org-1", "tau-bench", dest)
    assert not dest.exists()


def test_harness_round_trip_payloads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == "/api/orgs/org-1/harnesses/agent/versions"
            payload = json.loads(request.read())
            assert payload["doc_hash"] == "a" * 32
            return httpx.Response(
                201,
                json={"name": "agent", "version": 3, "doc_hash": "a" * 32, "created": True},
            )
        if request.url.path == "/api/orgs/org-1/harnesses/agent/versions/3":
            return httpx.Response(
                200, json={"version": 3, "doc": {"name": "agent"}, "doc_hash": "a" * 32}
            )
        assert request.url.path == "/api/orgs/org-1/harnesses/agent"
        return httpx.Response(
            200,
            json={
                "harness": {"id": "h-1", "name": "agent", "latest_version": 3},
                "versions": [{"version": 3, "doc_hash": "a" * 32}],
            },
        )

    with _client(handler) as client:
        pushed = client.push_harness_version("org-1", "agent", {"name": "agent"}, "a" * 32)
        assert pushed.version == 3
        assert pushed.created

        harness, versions = client.get_harness("org-1", "agent")
        assert harness.latest_version == 3
        assert versions[0].doc_hash == "a" * 32

        doc = client.get_harness_version("org-1", "agent", 3)
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
