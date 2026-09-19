"""Upstream DNS resolution bypasses hosts and respects finite authoritative TTLs."""

import asyncio

import dns.message
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.resolver
import dns.rrset
import pytest

from exp.runtime.capture.resolver import UpstreamResolver


def _answer(host: str, kind: str, address: str, ttl: int) -> dns.resolver.Answer:
    """Build a real DNS answer containing one synthetic record and TTL."""
    response = dns.message.make_response(dns.message.make_query(host, kind))
    response.answer.append(dns.rrset.from_text(host + ".", ttl, "IN", kind, address))
    response = dns.message.from_wire(response.to_wire())
    assert isinstance(response, dns.message.QueryMessage)
    return dns.resolver.Answer(
        dns.name.from_text(host), dns.rdatatype.from_text(kind), dns.rdataclass.IN, response
    )


def test_dns_uses_wire_resolver_and_refreshes_zero_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh expired DNS answers instead of retaining a fixed upstream address."""
    calls: list[str] = []

    async def query(host: str, kind: str, *, search: bool) -> dns.resolver.Answer:
        """Return one synthetic DNS answer without consulting hosts or the network."""
        assert not search
        calls.append(kind)
        return _answer(host, kind, "1.1.1.1" if kind == "A" else "2606:4700:4700::1111", 0)

    resolver = UpstreamResolver()
    monkeypatch.setattr(resolver._resolver, "resolve", query)

    async def run() -> None:
        """Run the asynchronous synthetic networking scenario to completion."""
        assert await resolver.resolve("api.example") == "1.1.1.1"
        assert await resolver.resolve("api.example") == "1.1.1.1"

    asyncio.run(run())
    assert calls == ["A", "AAAA", "A", "AAAA"]


def test_loopback_and_private_dns_answers_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject redirected loopback answers before they can create a routing loop."""

    async def query(host: str, kind: str, *, search: bool) -> dns.resolver.Answer:
        """Return one synthetic DNS answer without consulting hosts or the network."""
        return _answer(host, kind, "127.0.0.1" if kind == "A" else "::1", 100)

    resolver = UpstreamResolver()
    monkeypatch.setattr(resolver._resolver, "resolve", query)
    with pytest.raises(ValueError, match="public upstream"):
        asyncio.run(resolver.resolve("api.example"))
