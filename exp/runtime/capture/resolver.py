"""TTL-bound upstream DNS lookups that never consult the overridden hosts file."""

from __future__ import annotations

import asyncio
import ipaddress
import time
from dataclasses import dataclass

import dns.asyncresolver
import dns.exception
import dns.resolver


@dataclass(frozen=True)
class _Resolution:
    """One public upstream address and its monotonic expiration."""

    address: str
    expires: float


class UpstreamResolver:
    """Resolve directly through configured DNS servers instead of libc hosts lookup."""

    def __init__(self, nameservers: tuple[str, ...] | None = None) -> None:
        """Keep the user's DNS configuration, allowing an explicit server override."""
        self._resolver = dns.asyncresolver.Resolver(configure=True)
        if nameservers:
            self._resolver.nameservers = list(nameservers)
        self._resolver.timeout = 2.0
        self._resolver.lifetime = 4.0
        self._cache: dict[str, _Resolution] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def prime(self, domains: tuple[str, ...]) -> None:
        """Verify all upstreams before the caller enables local DNS redirection."""
        await asyncio.gather(*(self.resolve(domain) for domain in domains))

    async def resolve(self, host: str) -> str:
        """Return a current public address, rejecting loops and unsafe DNS answers.

        DNS queries use dnspython's wire resolver and do not consult `/etc/hosts`.
        Cache entries expire at the authoritative TTL, capped at five minutes.
        A failed refresh never reuses an expired address indefinitely.
        """
        host = host.lower().rstrip(".")
        async with self._locks.setdefault(host, asyncio.Lock()):
            cached = self._cache.get(host)
            if cached is not None and cached.expires > time.monotonic():
                return cached.address
            answers = await asyncio.gather(
                self._resolver.resolve(host, "A", search=False),
                self._resolver.resolve(host, "AAAA", search=False),
                return_exceptions=True,
            )
            for answer in answers:
                if not isinstance(answer, dns.resolver.Answer) or answer.rrset is None:
                    continue
                for record in answer:
                    address = record.to_text()
                    try:
                        parsed = ipaddress.ip_address(address)
                    except ValueError:
                        continue
                    if parsed.is_global:
                        self._cache[host] = _Resolution(
                            address, time.monotonic() + min(max(answer.rrset.ttl, 0), 300)
                        )
                        return address
            raise ValueError(f"cannot resolve a public upstream address for {host}")
