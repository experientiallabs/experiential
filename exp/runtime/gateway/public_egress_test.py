"""Tests for the public-only destination policy of the gateway's Python HTTP clients."""

from __future__ import annotations

import ipaddress
import ssl
import threading
from collections.abc import Iterable, Sequence

import httpcore
import httpx
import pytest

from exp.runtime.gateway import public_egress
from exp.runtime.gateway.public_egress import (
    PublicOnlyNetworkBackend,
    PublicOnlyTransport,
    is_public_address,
    public_only_client,
)

_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"


class _Stream(httpcore.NetworkStream):
    """A socket stand-in that records TLS and request bytes and answers one 200."""

    def __init__(self, record: _Backend) -> None:
        self._record = record
        self._pending = _RESPONSE

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk, self._pending = self._pending[:max_bytes], self._pending[max_bytes:]
        return chunk

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._record.written += buffer

    def close(self) -> None:
        return None

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        self._record.server_hostnames.append(server_hostname)
        return self

    def get_extra_info(self, info: str) -> object:
        return None


class _Backend(httpcore.NetworkBackend):
    """An inner backend that records every connect and never touches the network."""

    def __init__(self) -> None:
        self.connected: list[tuple[str, int]] = []
        self.server_hostnames: list[str | None] = []
        self.written = b""

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        self.connected.append((host, port))
        return _Stream(self)


def _client(answers: Sequence[str]) -> tuple[httpx.Client, _Backend, list[str]]:
    """Build a public-only client over a fixed resolution and a recording backend."""
    inner = _Backend()
    asked: list[str] = []

    def resolver(host: str, port: int) -> Sequence[str]:
        asked.append(host)
        return answers

    client = httpx.Client(
        trust_env=False, transport=PublicOnlyTransport(resolver=resolver, inner=inner)
    )
    return client, inner, asked


@pytest.mark.parametrize(
    "answers",
    [
        ["169.254.169.254"],
        ["127.0.0.1"],
        ["10.1.2.3"],
        ["::ffff:127.0.0.1"],
        ["::1"],
        ["2002:a00:1::1"],
        ["2001:0:4136:e378:8000:63bf:3fff:fdd2"],
        ["fd00::1"],
        ["fe80::1%1"],
        [],
    ],
)
def test_a_non_public_resolution_is_refused_before_any_byte_is_sent(answers: list[str]) -> None:
    """A private, loopback, link-local, mapped or transition answer never opens a socket."""
    client, inner, asked = _client(answers)
    with client, pytest.raises(httpx.ConnectError, match="not allowed"):
        client.post("https://videos.example.com/v1/jobs", content=b"{}")
    assert asked == ["videos.example.com"]
    assert inner.connected == []
    assert inner.written == b""


def test_a_mixed_public_and_private_resolution_is_refused_whole() -> None:
    """One non-public answer refuses every address, the public ones included."""
    client, inner, _ = _client(["93.184.216.34", "10.0.0.5"])
    with client, pytest.raises(httpx.ConnectError):
        client.get("https://videos.example.com/v1/jobs/1")
    assert inner.connected == []


@pytest.mark.parametrize(
    "url",
    [
        "http://videos.example.com/v1/jobs",
        "https://user:secret@videos.example.com/v1/jobs",
        "https://169.254.169.254/latest/meta-data",
        "https://[::ffff:7f00:1]/v1/jobs",
    ],
)
def test_a_forbidden_url_is_refused_before_resolution_or_connect(url: str) -> None:
    """Plain HTTP, embedded credentials and non-public literals never reach a socket."""
    client, inner, _ = _client(["93.184.216.34"])
    with client, pytest.raises(httpx.ConnectError):
        client.get(url)
    assert inner.connected == []


def test_a_public_resolution_connects_to_the_vetted_address_with_the_original_name() -> None:
    """The socket opens to the checked address while TLS and Host keep the hostname."""
    client, inner, _ = _client(["93.184.216.34", "2606:2800:220:1::1"])
    with client:
        response = client.get("https://videos.example.com/v1/jobs/1")
    assert response.status_code == 200
    assert inner.connected == [("93.184.216.34", 443)]
    assert inner.server_hostnames == ["videos.example.com"]
    assert b"Host: videos.example.com" in inner.written


def test_a_refused_address_falls_through_to_the_next_vetted_one() -> None:
    """A vetted address that does not accept moves to the next vetted address only."""

    class _FirstRefuses(_Backend):
        def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
        ) -> httpcore.NetworkStream:
            if not self.connected:
                self.connected.append((host, port))
                raise httpcore.ConnectError("refused")
            return super().connect_tcp(host, port)

    inner = _FirstRefuses()
    backend = PublicOnlyNetworkBackend(
        resolver=lambda _host, _port: ["93.184.216.34", "93.184.216.35"], inner=inner
    )
    backend.connect_tcp("videos.example.com", 443)
    assert inner.connected == [("93.184.216.34", 443), ("93.184.216.35", 443)]


def test_the_client_ignores_environment_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambient proxy cannot move resolution outside the policy."""
    monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.1:3128")
    client = public_only_client(timeout=httpx.Timeout(5.0))
    with client:
        assert client.trust_env is False
        assert client.follow_redirects is False
        assert client._mounts == {}
        assert isinstance(client._transport, PublicOnlyTransport)


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("8.8.8.8", True),
        ("100.64.0.1", False),
        ("192.0.0.8", False),
        ("198.18.0.1", False),
        ("224.0.0.1", False),
        ("255.255.255.255", False),
        ("2606:4700::1111", True),
        ("2001:db8::1", False),
        ("64:ff9b::a00:1", False),
        ("3fff::1", False),
        ("ff02::1", False),
    ],
)
def test_the_address_policy_matches_the_native_ranges(address: str, public: bool) -> None:
    """Each range the native transport excludes is excluded here as well."""
    assert is_public_address(ipaddress.ip_address(address)) is public


def test_the_system_resolver_keeps_distinct_answers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate getaddrinfo rows (one per socket family entry) collapse to one address."""
    rows = [
        (2, 1, 6, "", ("93.184.216.34", 443)),
        (2, 1, 6, "", ("93.184.216.34", 443)),
        (10, 1, 6, "", ("2606:2800:220:1::1", 443, 0, 0)),
    ]
    monkeypatch.setattr(public_egress.socket, "getaddrinfo", lambda *_args, **_kwargs: rows)
    assert public_egress.system_resolver("videos.example.com", 443) == [
        "93.184.216.34",
        "2606:2800:220:1::1",
    ]


def test_the_policy_is_one_way_once_required() -> None:
    """Required once, every later provider client is public-only; nothing turns it back off."""
    with public_egress.provider_client(timeout=httpx.Timeout(5.0)) as client:
        assert not isinstance(client._transport, PublicOnlyTransport)
        assert client.trust_env is False
    assert public_egress.public_upstreams_only() is False
    public_egress.require_public_upstreams()
    public_egress.require_public_upstreams()
    assert public_egress.public_upstreams_only() is True
    with public_egress.provider_client(timeout=httpx.Timeout(5.0)) as client:
        assert isinstance(client._transport, PublicOnlyTransport)


def test_a_poller_thread_sees_the_policy_a_later_serve_requires() -> None:
    """A client built on another thread after the requirement (a video poller) is public-only."""
    seen: list[bool] = []
    public_egress.require_public_upstreams()

    def poll() -> None:
        with public_egress.provider_client(timeout=httpx.Timeout(5.0)) as client:
            seen.append(isinstance(client._transport, PublicOnlyTransport))

    thread = threading.Thread(target=poll)
    thread.start()
    thread.join()
    assert seen == [True]


def test_tls_trust_honors_operator_ca_overrides_but_not_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSL_CERT_FILE still selects the trust store; proxy variables are still ignored."""
    calls: list[bool] = []
    real = httpx.create_ssl_context

    def record(*, trust_env: bool = True) -> ssl.SSLContext:
        calls.append(trust_env)
        return real(trust_env=trust_env)

    monkeypatch.setattr(public_egress.httpx, "create_ssl_context", record)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    with public_egress.provider_client(timeout=httpx.Timeout(5.0)) as client:
        assert client.trust_env is False
    public_egress.require_public_upstreams()
    with public_egress.provider_client(timeout=httpx.Timeout(5.0)) as client:
        assert client.trust_env is False
    assert calls and all(calls)
