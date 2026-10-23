"""Web-search backends the control plane calls at admission.

The engine ships two production backends behind a small protocol so a host
can supply another vendor or a fixture: Exa (the same engine OpenRouter
defaults to for models without native search), bound by its key, and
Keenable, bound only when the operator names it or sets its key. Credentials
are environment references only, mirroring provider secrets: the value is
read at call time and never retained on the backend object.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from itertools import zip_longest
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

WEB_SEARCH_BACKEND_ENV: Final = "EXP_WEB_SEARCH_BACKEND"
"""Optional explicit backend choice, ``exa`` or ``keenable``; unset binds by key."""

KEENABLE_API_KEY_ENV: Final = "KEENABLE_API_KEY"
"""Optional Keenable credential: lifts the keyless rate limit and selects Keenable without Exa."""

KEENABLE_SEARCH_URL: Final = "https://api.keenable.ai/v1/search"
KEENABLE_PUBLIC_SEARCH_URL: Final = "https://api.keenable.ai/v1/search/public"
KEENABLE_SEARCH_URL_ENV: Final = "KEENABLE_SEARCH_URL"
"""Optional override of the Keenable endpoint, keyed or keyless (a proxy or a loopback fixture)."""
KEENABLE_APP_TITLE: Final = "Experiential"
"""App label Keenable requires on keyless calls; sent on every call."""

_BACKEND_CHOICES: Final = ("exa", "keenable")
_MAXIMUM_RESPONSE_BYTES: Final = 262_144
_SNIPPET_CHARACTERS: Final = 400
_FILTERED_FETCH_RESULTS: Final = 20
_MAXIMUM_SITE_SEARCHES: Final = 5
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
        raw = await _post_search("exa", self._url, body, headers, timeout_seconds)
        return _parse_exa(raw, max_results)

    def __repr__(self) -> str:
        """Never show the credential (only its environment name)."""
        return f"ExaWebSearchBackend(api_key_env={self._api_key_env!r})"


class KeenableWebSearchBackend:
    """Keenable ``/v1/search``, keyless unless its environment key is set."""

    name = "keenable"

    def __init__(
        self,
        *,
        api_key_env: str = KEENABLE_API_KEY_ENV,
        url: str = KEENABLE_SEARCH_URL,
        public_url: str = KEENABLE_PUBLIC_SEARCH_URL,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Bind the optional credential reference; the value is read per call.

        Args:
            api_key_env: Environment variable that may hold a Keenable API key.
            url: Keyed search endpoint, used when the key is set.
            public_url: Keyless search endpoint, used when it is not.
            environ: Environment mapping, defaulting to the process environment.
        """
        self._api_key_env = api_key_env
        self._url = url
        self._public_url = public_url
        self._environ = environ

    def _api_key(self) -> str:
        """Return the optional key, or an empty string for a keyless call."""
        environ = os.environ if self._environ is None else self._environ
        return environ.get(self._api_key_env, "")

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        allowed_domains: Sequence[str],
        blocked_domains: Sequence[str],
        timeout_seconds: float,
    ) -> tuple[GatewayWebSearchResult, ...]:
        """Run one Keenable search and normalize its hits.

        Keenable takes one ``site`` filter per call, so each allowed domain (up
        to five) gets its own concurrent call and the ranked hits are
        interleaved; blocked domains are removed from a larger page of hits by
        host name.

        Args:
            query: Natural-language query.
            max_results: Result ceiling, forwarded as ``max_results`` unless
                blocked domains are removed locally.
            allowed_domains: Hosts (and their subdomains) a hit must belong to.
            blocked_domains: Hosts (and their subdomains) a hit must not belong to.
            timeout_seconds: Whole-call budget.

        Returns:
            Ranked hits with the bounded page text as the snippet.

        Raises:
            WebSearchBackendError: Transport failure, non-2xx, oversized or
                malformed body, or timeout.
        """
        allowed = tuple(dict.fromkeys(_filter_host(domain) for domain in allowed_domains))
        blocked = tuple(_filter_host(domain) for domain in blocked_domains)
        fetch = max(max_results, _FILTERED_FETCH_RESULTS) if blocked else max_results
        headers = {"x-keenable-title": KEENABLE_APP_TITLE, "content-type": "application/json"}
        url = self._public_url
        api_key = self._api_key()
        if api_key:
            headers["x-api-key"] = api_key
            url = self._url

        async def search_site(site: str | None) -> tuple[GatewayWebSearchResult, ...]:
            """Run one call, restricted to ``site`` when given."""
            body: dict[str, object] = {"query": query, "max_results": fetch}
            if site is not None:
                body["site"] = site
            raw = await _post_search("keenable", url, body, headers, timeout_seconds)
            return _parse_keenable(raw, max_results, allowed=allowed, blocked=blocked)

        sites = allowed[:_MAXIMUM_SITE_SEARCHES] or (None,)
        pages = await asyncio.gather(*(search_site(site) for site in sites), return_exceptions=True)
        ranked: list[tuple[GatewayWebSearchResult, ...]] = []
        for page in pages:
            if isinstance(page, BaseException):
                raise page
            ranked.append(page)
        return _interleave(ranked, max_results)

    def __repr__(self) -> str:
        """Never show the credential (only its environment name)."""
        return f"KeenableWebSearchBackend(api_key_env={self._api_key_env!r})"


async def _post_search(
    vendor: str,
    url: str,
    body: Mapping[str, object],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> bytes:
    """POST one search body and return the bounded body of a 2xx answer.

    Args:
        vendor: Backend name used in error messages.
        url: Search endpoint.
        body: JSON request body.
        headers: Request headers, credential included when the vendor needs one.
        timeout_seconds: Whole-call budget.

    Returns:
        The raw response body.

    Raises:
        WebSearchBackendError: Transport failure, non-2xx, oversized body, or timeout.
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await _shared_client().post(
                url, json=body, headers=headers, timeout=timeout_seconds
            )
            raw = await _read_bounded(response, vendor)
    except WebSearchBackendError:
        raise
    except (httpx.HTTPError, TimeoutError, OSError) as exc:
        raise WebSearchBackendError(f"{vendor} search failed: {type(exc).__name__}") from exc
    if response.status_code // 100 != 2:
        raise WebSearchBackendError(f"{vendor} search answered HTTP {response.status_code}")
    return raw


async def _read_bounded(response: httpx.Response, vendor: str) -> bytes:
    """Read the body up to the size bound, refusing anything larger."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAXIMUM_RESPONSE_BYTES:
            raise WebSearchBackendError(f"{vendor} search body exceeded the gateway limit")
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


def _interleave(
    pages: Sequence[Sequence[GatewayWebSearchResult]], max_results: int
) -> tuple[GatewayWebSearchResult, ...]:
    """Merge ranked pages rank by rank, dropping repeated URLs, up to ``max_results``."""
    seen: set[str] = set()
    hits: list[GatewayWebSearchResult] = []
    for rank in zip_longest(*pages):
        for hit in rank:
            if hit is not None and hit.url not in seen and len(hits) < max_results:
                seen.add(hit.url)
                hits.append(hit)
    return tuple(hits)


def _filter_host(domain: str) -> str:
    """Reduce a caller's domain filter to a bare lowercase host name."""
    host = domain.strip().lower().removeprefix("https://").removeprefix("http://")
    return host.split("/", 1)[0].rstrip(".")


def _host_within(host: str, domains: Sequence[str]) -> bool:
    """Return whether ``host`` is one of ``domains`` or a subdomain of one."""
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _parse_keenable(
    raw: bytes,
    max_results: int,
    *,
    allowed: Sequence[str],
    blocked: Sequence[str],
) -> tuple[GatewayWebSearchResult, ...]:
    """Normalize Keenable's ``results`` array; malformed or filtered entries are skipped.

    Args:
        raw: Response body.
        max_results: Most hits returned.
        allowed: Bare hosts a hit must belong to; empty admits every host.
        blocked: Bare hosts a hit must not belong to.

    Returns:
        Ranked hits whose snippet is the page text with whitespace collapsed and bounded.

    Raises:
        WebSearchBackendError: The body is not JSON or has no results array.
    """
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise WebSearchBackendError("keenable search body is not JSON") from exc
    entries = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise WebSearchBackendError("keenable search body has no results array")
    hits: list[GatewayWebSearchResult] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        try:
            host = (urlparse(url).hostname or "").rstrip(".")
        except ValueError:
            continue
        if (allowed and not _host_within(host, allowed)) or _host_within(host, blocked):
            continue
        title = entry.get("title")
        snippet = entry.get("snippet")
        text = " ".join(snippet.split()) if isinstance(snippet, str) else ""
        try:
            hits.append(
                GatewayWebSearchResult(
                    url=url[:2048],
                    title=(title if isinstance(title, str) else "")[:512],
                    snippet=text[:_SNIPPET_CHARACTERS],
                )
            )
        except ValueError:
            continue
        if len(hits) >= max_results:
            break
    return tuple(hits)


_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "::1", "localhost"})


def validate_search_url(url: str, *, env_name: str = EXA_SEARCH_URL_ENV) -> str:
    """Accept an https search endpoint, or plain http only to an exact loopback host.

    The credential rides every search request, so the endpoint must not be
    steerable to an attacker host: the URL is parsed and the host compared
    exactly (a prefix test would admit ``127.0.0.1.evil.example``).

    Args:
        url: Candidate endpoint.
        env_name: Environment variable the URL came from, named in errors.

    Returns:
        The validated URL.

    Raises:
        ValueError: The scheme, host, credentials, or fragment are unacceptable.
    """
    parsed = urlparse(url)
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise ValueError(f"{env_name} must be a bare https URL")
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return url
    raise ValueError(f"{env_name} must be an https URL (or an exact loopback fixture)")


def default_web_search_backend(
    environ: Mapping[str, str] | None = None,
) -> WebSearchBackend | None:
    """Return the backend the environment selects, else ``None``.

    ``EXP_WEB_SEARCH_BACKEND`` names the backend explicitly (``exa`` or
    ``keenable``). Unset, Exa binds when ``EXA_API_KEY`` is present, then
    Keenable when ``KEENABLE_API_KEY`` is present; with neither there is no
    backend. Keenable named without a key searches keyless.

    Args:
        environ: Environment mapping, defaulting to the process environment.

    Returns:
        A backend, or ``None`` so requests asking for search are disclosed-dropped.

    Raises:
        ValueError: The backend name is unknown, or an endpoint override is unsafe.
    """
    mapping = os.environ if environ is None else environ
    choice = mapping.get(WEB_SEARCH_BACKEND_ENV, "").strip().lower()
    if choice and choice not in _BACKEND_CHOICES:
        raise ValueError(
            f"{WEB_SEARCH_BACKEND_ENV} must be one of {', '.join(_BACKEND_CHOICES)} "
            f"(or unset to bind by key), not {choice!r}"
        )
    if not choice:
        if mapping.get(EXA_API_KEY_ENV):
            choice = "exa"
        elif mapping.get(KEENABLE_API_KEY_ENV):
            choice = "keenable"
        else:
            return None
    if choice == "keenable":
        override = mapping.get(KEENABLE_SEARCH_URL_ENV)
        url = validate_search_url(override, env_name=KEENABLE_SEARCH_URL_ENV) if override else None
        _logger.info("gateway web search backend: keenable")
        if url is None:
            return KeenableWebSearchBackend(environ=environ)
        return KeenableWebSearchBackend(url=url, public_url=url, environ=environ)
    if not mapping.get(EXA_API_KEY_ENV):
        _logger.warning("gateway web search: exa is selected without its key; searches drop")
        return None
    url = validate_search_url(mapping.get(EXA_SEARCH_URL_ENV) or EXA_SEARCH_URL)
    _logger.info("gateway web search backend: exa")
    return ExaWebSearchBackend(url=url, environ=environ)
