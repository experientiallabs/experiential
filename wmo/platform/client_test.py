"""Tests for the platform HTTP client (httpx mock transport, no network)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from wmo.core.types import Action, ActionKind
from wmo.platform.client import (
    PlatformClient,
    PlatformError,
    PlatformUnreachable,
    fetch_cli_config,
)

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

    with _client(handler) as client, pytest.raises(PlatformError, match="wmo login"):
        client.whoami()


def test_unified_run_target_and_world_model_session_payloads() -> None:
    """The run client resolves once, then uses the hosted world-model session API."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/run-targets/wm-1":
            return httpx.Response(
                200,
                json={
                    "id": "wm-1",
                    "kind": "world_model",
                    "org_id": "org-1",
                    "name": "tau",
                    "status": "ready",
                },
            )
        if request.url.path == "/api/world-models/wm-1/sessions":
            assert json.loads(request.read()) == {"task": "book a flight"}
            return httpx.Response(
                201,
                json={"id": "sess-1", "world_model_id": "wm-1", "status": "active"},
            )
        assert request.url.path == "/api/sessions/sess-1/step"
        body = json.loads(request.read())
        assert body["action"] == {
            "kind": "tool_call",
            "name": "search",
            "arguments": {"q": "SFO"},
            "content": None,
        }
        return httpx.Response(200, json={"observation": {"content": "three flights"}})

    with _client(handler) as client:
        target = client.resolve_run_target("wm-1")
        session = client.create_world_model_session(target.id, task="book a flight")
        observation = client.step_world_model_session(
            session.id,
            Action(kind=ActionKind.TOOL_CALL, name="search", arguments={"q": "SFO"}),
        )

    assert target.kind == "world_model"
    assert observation.content == "three flights"
    assert seen == [
        "/api/run-targets/wm-1",
        "/api/world-models/wm-1/sessions",
        "/api/sessions/sess-1/step",
    ]


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


def test_fetch_cli_config_maps_an_unreachable_host() -> None:
    """A refused connection is a platform verdict, not an httpx traceback."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    with pytest.raises(PlatformUnreachable, match="cannot reach") as info:
        fetch_cli_config("https://platform.test", transport=httpx.MockTransport(handler))
    assert "wmo login --url" in str(info.value)
    assert info.value.status_code is None


def test_fetch_cli_config_maps_a_non_json_200() -> None:
    """A login-walled preview answers 200 with HTML; that is 'not a platform'."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html>Authentication Required</html>")

    with pytest.raises(PlatformError, match="not JSON") as info:
        fetch_cli_config("https://platform.test", transport=httpx.MockTransport(handler))
    assert not isinstance(info.value, PlatformUnreachable)
    assert "text/html" in str(info.value)


def test_whoami_maps_a_non_json_200() -> None:
    """`login --api-url`/`status` reach whoami first: a captive portal is not a crash."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html>Sign in</html>")

    with _client(handler) as client, pytest.raises(PlatformError, match="not JSON"):
        client.whoami()


def test_a_json_array_body_is_a_platform_error() -> None:
    """Valid JSON that is not an object would crash `.get`/`model_validate` the same way."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with _client(handler) as client, pytest.raises(PlatformError, match="not an object"):
        client.list_world_models("org-1")


def test_an_unreachable_storage_host_hides_the_signed_url(tmp_path: Path) -> None:
    """Signed upload URLs carry their capability in the query; errors land in CI logs."""
    bundle_path = tmp_path / "tau-bench.tar.gz"
    bundle_path.write_bytes(b"bundle-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            raise httpx.ConnectError("[Errno 61] Connection refused", request=request)
        return httpx.Response(
            201,
            json={
                "upload_url": "https://storage.test/upload/x?token=SIGNATURE",
                "token": "SIGNATURE",
                "staging_path": "staging/cli/x.tar.gz",
            },
        )

    with _client(handler) as client, pytest.raises(PlatformUnreachable) as info:
        client.push_model_bundle("org-1", "tau-bench", bundle_path, "0" * 64, 12, {})

    message = str(info.value)
    assert "SIGNATURE" not in message
    # The hop that failed still has to be named, host and path intact.
    reached = message.removeprefix("cannot reach ").split(": ", 1)[0]
    assert reached == "https://storage.test/upload/x"
