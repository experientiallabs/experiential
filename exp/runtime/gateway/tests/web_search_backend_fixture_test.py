"""Fixture web-search backends shared by the search planning and admission tests."""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.web_search.backend import WebSearchBackendError
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


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
