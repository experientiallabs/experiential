"""Public-only destination policy for the gateway's own Python HTTP clients.

The native data plane enforces ``public_upstreams_only`` inside its Rust
transport. A Python client that the gateway itself points at a provider URL
(the asynchronous video job submit and poll) holds the same rule here, at
connect time:

- Only ``https`` URLs without credentials or a fragment are sent.
- Every A/AAAA answer for the host is checked, and one non-public answer
  refuses the whole resolution, so a mixed answer cannot reach the private
  address on a later attempt.
- The socket is opened to the vetted address itself, never re-resolved, so a
  DNS change between the check and the connect cannot redirect it. TLS still
  verifies, and HTTP still addresses, the original hostname.
- Environment proxies are ignored and redirects stay off.

The policy is one process-wide flag that can only be turned ON
(:func:`require_public_upstreams`): ``serve_native_gateway`` turns it on from
its own ``public_upstreams_only``, and a hosted embedder may turn it on earlier,
before constructing a control plane whose video poller starts at once.
:func:`provider_client` reads it on every call, so every control plane in the
process (execution-bound ones included, and pollers that outlive a serve)
holds the same rule as the native transport. A process that never turns it on
keeps the standalone default that permits local and private servers.

The address ranges mirror the native policy exactly: private, loopback,
link-local, shared, reserved, documentation, benchmarking and multicast IPv4
ranges, and every IPv6 address outside global unicast ``2000::/3`` or inside
its transition and documentation prefixes. IPv4-mapped, NAT64, Teredo and 6to4
forms are therefore refused outright rather than unwrapped, so an embedded
private IPv4 destination can never cross the boundary.
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import threading
from collections.abc import Callable, Iterable, Sequence

import httpcore
import httpx

Resolver = Callable[[str, int], Sequence[str]]
"""Resolve ``(host, port)`` to the address strings a connection may use."""

_BLOCKED_V4 = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)
_GLOBAL_UNICAST_V6 = ipaddress.IPv6Network("2000::/3")
_BLOCKED_V6 = tuple(
    ipaddress.IPv6Network(network)
    for network in ("2001::/23", "2001:db8::/32", "2002::/16", "3fff::/20")
)

REFUSAL = "provider endpoint is not allowed; configure a public HTTPS endpoint"
"""The one message every refusal carries; it names no resolved address."""


def is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether one address is a public destination under the native policy.

    Args:
        address: A parsed IPv4 or IPv6 address.

    Returns:
        True only for global unicast addresses outside every excluded range.
    """
    if isinstance(address, ipaddress.IPv4Address):
        return not any(address in network for network in _BLOCKED_V4)
    return address in _GLOBAL_UNICAST_V6 and not any(address in network for network in _BLOCKED_V6)


def system_resolver(host: str, port: int) -> list[str]:
    """Resolve one host through the operating system, keeping answer order.

    Args:
        host: The hostname or address literal.
        port: The destination port.

    Returns:
        The distinct address strings, in resolver order.
    """
    answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in answers))


def vetted_addresses(host: str, port: int, resolver: Resolver) -> list[str]:
    """Resolve one host and return its addresses only if every one is public.

    Args:
        host: The URL host, an address literal possibly in brackets.
        port: The destination port.
        resolver: The resolution seam.

    Returns:
        The vetted addresses to connect to, in resolver order.

    Raises:
        httpcore.ConnectError: The resolution failed, was empty, or named any
            non-public or unparseable address.
    """
    literal = host.strip("[]")
    try:
        # An address literal is its own answer and never reaches the resolver.
        answers: Sequence[str] = [str(ipaddress.ip_address(literal))]
    except ValueError:
        answers = []
    try:
        answers = answers or resolver(literal, port)
    except OSError as exc:
        raise httpcore.ConnectError(REFUSAL) from exc
    vetted: list[str] = []
    for answer in answers:
        try:
            address = ipaddress.ip_address(answer)
        except ValueError as exc:
            raise httpcore.ConnectError(REFUSAL) from exc
        if address.version == 6 and address.scope_id is not None:
            raise httpcore.ConnectError(REFUSAL)
        if not is_public_address(address):
            raise httpcore.ConnectError(REFUSAL)
        vetted.append(str(address))
    if not vetted:
        raise httpcore.ConnectError(REFUSAL)
    return vetted


class PublicOnlyNetworkBackend(httpcore.NetworkBackend):
    """Open TCP connections only to vetted public addresses.

    httpcore starts TLS on the returned stream with the request's own
    hostname, so certificate verification and SNI are unchanged by connecting
    to the address.
    """

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        inner: httpcore.NetworkBackend | None = None,
    ) -> None:
        """Bind the resolution and socket seams.

        Args:
            resolver: Host resolution; the system resolver by default.
            inner: The backend that opens the socket to a vetted address.
        """
        self._resolver = resolver
        self._inner = inner if inner is not None else httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        """Connect to the first reachable vetted address of ``host``.

        Raises:
            httpcore.ConnectError: The destination is not public, or no vetted
                address accepted the connection.
        """
        options = None if socket_options is None else list(socket_options)
        failure: Exception | None = None
        for address in vetted_addresses(host, port, self._resolver):
            try:
                return self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                failure = exc
        assert failure is not None  # vetted_addresses never returns an empty list.
        raise failure

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        """Refuse every Unix socket: a public destination is always TCP.

        Raises:
            httpcore.ConnectError: Always.
        """
        del path, timeout, socket_options
        raise httpcore.ConnectError(REFUSAL)

    def sleep(self, seconds: float) -> None:
        """Delegate retry backoff to the inner backend."""
        self._inner.sleep(seconds)


class PublicOnlyTransport(httpx.HTTPTransport):
    """An httpx transport that sends only to public HTTPS destinations."""

    def __init__(
        self,
        *,
        resolver: Resolver = system_resolver,
        inner: httpcore.NetworkBackend | None = None,
    ) -> None:
        """Build a proxy-free pool over the public-only network backend.

        Args:
            resolver: Host resolution; the system resolver by default.
            inner: The backend that opens a socket to a vetted address; the
                synchronous system backend by default.
        """
        super().__init__(trust_env=False)
        self._pool = httpcore.ConnectionPool(
            ssl_context=_trust_store(),
            network_backend=PublicOnlyNetworkBackend(resolver=resolver, inner=inner),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Refuse a URL the policy forbids before any socket is opened.

        Raises:
            httpx.ConnectError: The URL is not plain ``https``, or its
                destination does not resolve only to public addresses.
        """
        url = request.url
        if url.scheme != "https" or url.userinfo or url.fragment or url.port == 0:
            raise httpx.ConnectError(REFUSAL, request=request)
        return super().handle_request(request)


def public_only_client(
    *, timeout: httpx.Timeout, resolver: Resolver = system_resolver
) -> httpx.Client:
    """Build one synchronous client bound to the public-only policy.

    Args:
        timeout: The client's request bounds.
        resolver: Host resolution; the system resolver by default.

    Returns:
        A client with redirects off, environment proxies ignored, and every
        connection checked by :class:`PublicOnlyTransport`.
    """
    return httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        transport=PublicOnlyTransport(resolver=resolver),
    )


def _trust_store() -> ssl.SSLContext:
    """TLS verification that honors an operator's ``SSL_CERT_FILE`` / ``SSL_CERT_DIR``.

    ``trust_env=False`` on the clients only keeps environment PROXIES out; the
    certificate trust an operator configured for a private CA still applies.
    """
    return httpx.create_ssl_context(trust_env=True)


class _ProcessPolicy:
    """The process's public-only flag: it can be turned on, never back off.

    One-way on purpose. Control planes, their background video pollers and
    every execution-bound plane in the process read this one flag, and their
    lifetimes do not nest inside one ``serve_native_gateway`` call: a poller
    starts at construction and runs until ``close()``, and two serves may
    overlap. A flag that a finishing scope restored could reopen private egress
    under a hosted server that is still running, so a hosted process stays
    public-only until it exits.
    """

    def __init__(self) -> None:
        """Start permissive, the standalone default."""
        self._lock = threading.Lock()
        self._public_only = False

    def get(self) -> bool:
        """Return the flag under the lock (a plain read of the current value)."""
        with self._lock:
            return self._public_only

    def require(self) -> None:
        """Turn the flag on for the rest of the process; idempotent, never reversed."""
        with self._lock:
            self._public_only = True


_POLICY = _ProcessPolicy()


def public_upstreams_only() -> bool:
    """Return whether this process requires public-only egress for Python provider clients."""
    return _POLICY.get()


def require_public_upstreams() -> None:
    """Hold every Python provider client in this process to the public-only policy.

    Hosted embedders call this before constructing a control plane, so a video
    poller that starts at construction never runs a permissive client.
    ``serve_native_gateway(public_upstreams_only=True)`` also calls it. It is
    never undone: see :class:`_ProcessPolicy`.
    """
    _POLICY.require()


def provider_client(*, timeout: httpx.Timeout) -> httpx.Client:
    """Build one provider client under the policy in force right now.

    Args:
        timeout: The client's request bounds.

    Returns:
        :func:`public_only_client` while the host serves public-only, otherwise
        a client that may reach local or private servers. Both ignore
        environment proxies and never follow redirects.
    """
    if public_upstreams_only():
        return public_only_client(timeout=timeout)
    return httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        verify=_trust_store(),
    )
