"""Web-search backends the control plane calls at admission.

The engine ships one production backend (Exa, the same engine OpenRouter
defaults to for models without native search) behind a small protocol so a
host can supply another vendor or a fixture. Credentials are environment
references only, mirroring provider secrets: the value is read at call time
and never retained on the backend object.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Final, Protocol
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

import httpx

from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult

_logger = logging.getLogger(__name__)

EXA_API_KEY_ENV: Final = "EXA_API_KEY"
"""Environment variable naming the Exa credential; absent means no backend."""

EXA_SEARCH_URL: Final = "https://api.exa.ai/search"
EXA_SEARCH_URL_ENV: Final = "EXA_SEARCH_URL"
"""Optional override of the Exa endpoint (a proxy, or a loopback fixture in tests)."""
_MAXIMUM_RESPONSE_BYTES: Final = 262_144
_SNIPPET_CHARACTERS: Final = 400
_MAX_CONNECTIONS: Final = 16

_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = WeakKeyDictionary()
_clients_lock = threading.Lock()


class WebSearchBackendError(RuntimeError):
    """The search vendor failed, timed out, or answered outside its contract."""


class WebSearchBackend(Protocol):
    """One search vendor the gateway can query before dispatch."""

    @property
    def name(self) -> str:
        """Stable vendor label for disclosures and logs."""
        ...

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        allowed_domains: Sequence[str],
        blocked_domains: Sequence[str],
        timeout_seconds: float,
    ) -> tuple[GatewayWebSearchResult, ...]:
        """Return up to ``max_results`` ranked hits for ``query``.

        Raises:
            WebSearchBackendError: The vendor did not answer usably in time.
        """
        ...


class StaticWebSearchBackend:
    """Fixture backend returning canned results; records every query."""

    name = "static"

    def __init__(self, results: Sequence[GatewayWebSearchResult]) -> None:
        """Serve ``results`` (truncated to the requested count) for any query.

        Args:
            results: Hits returned in order for every search.
        """
        self._results = tuple(results)
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        allowed_domains: Sequence[str],
        blocked_domains: Sequence[str],
        timeout_seconds: float,
    ) -> tuple[GatewayWebSearchResult, ...]:
        """Record the query and return the canned hits.

        Args:
            query: The derived search query.
            max_results: Requested result ceiling.
            allowed_domains: Ignored by the fixture.
            blocked_domains: Ignored by the fixture.
            timeout_seconds: Ignored by the fixture.

        Returns:
            The first ``max_results`` canned hits.
        """
        del allowed_domains, blocked_domains, timeout_seconds
        self.queries.append(query)
        return self._results[:max_results]


class FailingWebSearchBackend:
    """Fixture backend whose every search fails, for disclosure tests."""

    name = "failing"

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        allowed_domains: Sequence[str],
        blocked_domains: Sequence[str],
        timeout_seconds: float,
    ) -> tuple[GatewayWebSearchResult, ...]:
        """Always raise.

        Raises:
            WebSearchBackendError: Unconditionally.
        """
        del query, max_results, allowed_domains, blocked_domains, timeout_seconds
        raise WebSearchBackendError("fixture search backend is unavailable")


def _shared_client() -> httpx.AsyncClient:
    """Return one pooled client per event loop (never the provider pool)."""
    loop = asyncio.get_running_loop()
    with _clients_lock:
        client = _clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=_MAX_CONNECTIONS),
                follow_redirects=False,
                trust_env=False,
            )
            _clients[loop] = client
        return client


class ExaWebSearchBackend:
    """Exa ``/search`` with highlights, keyed by an environment reference."""

    name = "exa"

    def __init__(
        self,
        *,
        api_key_env: str = EXA_API_KEY_ENV,
        url: str = EXA_SEARCH_URL,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Bind the credential reference; the value is read per call.

        Args:
            api_key_env: Environment variable holding the Exa API key.
            url: Search endpoint (overridden by tests).
            environ: Environment mapping, defaulting to the process environment.
        """
        self._api_key_env = api_key_env
        self._url = url
        self._environ = environ

    def _api_key(self) -> str:
        environ = os.environ if self._environ is None else self._environ
        value = environ.get(self._api_key_env, "")
        if not value:
            raise WebSearchBackendError(f"{self._api_key_env} is not set")
        return value

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        allowed_domains: Sequence[str],
        blocked_domains: Sequence[str],
        timeout_seconds: float,
    ) -> tuple[GatewayWebSearchResult, ...]:
        """Run one Exa search and normalize its hits.

        Args:
            query: Natural-language query.
            max_results: Result ceiling forwarded as ``numResults``.
            allowed_domains: Forwarded as ``includeDomains`` when non-empty.
            blocked_domains: Forwarded as ``excludeDomains`` when non-empty.
            timeout_seconds: Whole-call budget.

        Returns:
            Ranked hits with a bounded highlight as the snippet.

        Raises:
            WebSearchBackendError: Transport failure, non-2xx, oversized or
                malformed body, or timeout.
        """
        body: dict[str, object] = {
            "query": query,
            "type": "auto",
            "numResults": max_results,
            "contents": {
                "highlights": {"maxCharacters": _SNIPPET_CHARACTERS, "highlightsPerUrl": 1}
            },
        }
        if allowed_domains:
            body["includeDomains"] = list(allowed_domains)
        if blocked_domains:
            body["excludeDomains"] = list(blocked_domains)
        headers = {"x-api-key": self._api_key(), "content-type": "application/json"}
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await _shared_client().post(
                    self._url, json=body, headers=headers, timeout=timeout_seconds
                )
                raw = await _read_bounded(response)
        except WebSearchBackendError:
            raise
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            raise WebSearchBackendError(f"exa search failed: {type(exc).__name__}") from exc
        if response.status_code // 100 != 2:
            raise WebSearchBackendError(f"exa search answered HTTP {response.status_code}")
        return _parse_exa(raw, max_results)

    def __repr__(self) -> str:
        """Never show the credential (only its environment name)."""
        return f"ExaWebSearchBackend(api_key_env={self._api_key_env!r})"


async def _read_bounded(response: httpx.Response) -> bytes:
    """Read the body up to the size bound, refusing anything larger."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAXIMUM_RESPONSE_BYTES:
            raise WebSearchBackendError("exa search body exceeded the gateway limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_exa(raw: bytes, max_results: int) -> tuple[GatewayWebSearchResult, ...]:
    """Normalize Exa's ``results`` array; malformed entries are skipped."""
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise WebSearchBackendError("exa search body is not JSON") from exc
    entries = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise WebSearchBackendError("exa search body has no results array")
    hits: list[GatewayWebSearchResult] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        title = entry.get("title")
        highlights = entry.get("highlights")
        snippet = ""
        if isinstance(highlights, list) and highlights and isinstance(highlights[0], str):
            snippet = highlights[0]
        published = entry.get("publishedDate")
        try:
            hits.append(
                GatewayWebSearchResult(
                    url=url[:2048],
                    title=(title if isinstance(title, str) else "")[:512],
                    snippet=snippet[:4000],
                    published_at=published[:64] if isinstance(published, str) else None,
                )
            )
        except ValueError:
            continue
        if len(hits) >= max_results:
            break
    return tuple(hits)


_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})


def validate_search_url(url: str) -> str:
    """Accept an https search endpoint, or plain http only to an exact loopback host.

    The credential rides every search request, so the endpoint must not be
    steerable to an attacker host: the URL is parsed and the host compared
    exactly (a prefix test would admit ``127.0.0.1.evil.example``).

    Args:
        url: Candidate endpoint.

    Returns:
        The validated URL.

    Raises:
        ValueError: The scheme, host, credentials, or fragment are unacceptable.
    """
    parsed = urlparse(url)
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise ValueError(f"{EXA_SEARCH_URL_ENV} must be a bare https URL")
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return url
    raise ValueError(f"{EXA_SEARCH_URL_ENV} must be an https URL (or an exact loopback fixture)")


def default_web_search_backend(
    environ: Mapping[str, str] | None = None,
) -> WebSearchBackend | None:
    """Return the Exa backend when its credential is present, else ``None``.

    Args:
        environ: Environment mapping, defaulting to the process environment.

    Returns:
        A backend, or ``None`` so requests asking for search are disclosed-dropped.
    """
    mapping = os.environ if environ is None else environ
    if not mapping.get(EXA_API_KEY_ENV):
        return None
    url = validate_search_url(mapping.get(EXA_SEARCH_URL_ENV) or EXA_SEARCH_URL)
    _logger.info("gateway web search backend: exa")
    return ExaWebSearchBackend(url=url, environ=environ)
