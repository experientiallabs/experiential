"""Canonical web-search request and result contracts.

One request may ask the gateway to search the web before the model answers.
Callers spell it five ways (OpenAI Chat ``web_search_options``, the
OpenRouter ``plugins: [{"id": "web"}]`` object and ``:online`` model suffix,
the OpenAI Responses ``web_search`` tool, the Anthropic ``web_search_*``
server tool); every spelling normalizes to :class:`GatewayWebSearch` so the
planner, the replay identity, and the provider payloads see one shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

MAXIMUM_WEB_SEARCH_RESULTS = 10
"""Most results one search may inject; matches the OpenRouter plugin ceiling."""

DEFAULT_WEB_SEARCH_RESULTS = 5
"""Results injected when the caller names no count (the OpenRouter default)."""

WebSearchDeclaredAs = Literal[
    "web_search_options",
    "plugin",
    "model_suffix",
    "responses_tool",
    "messages_server_tool",
]
"""The public spelling the caller used; drives native pass-through decisions."""

_CONTEXT_SIZE_RESULTS: dict[str, int] = {"low": 3, "medium": 5, "high": 8}


def results_for_context_size(size: str | None) -> int:
    """Map OpenAI's ``search_context_size`` onto a result count.

    Args:
        size: ``low``, ``medium``, ``high``, or ``None`` for the default.

    Returns:
        The number of results the search injects.
    """
    if size is None:
        return DEFAULT_WEB_SEARCH_RESULTS
    return _CONTEXT_SIZE_RESULTS.get(size, DEFAULT_WEB_SEARCH_RESULTS)


class GatewayWebSearchResult(ContractModel):
    """One ranked search hit the gateway injects and may cite."""

    url: str = Field(min_length=1, max_length=2048)
    title: str = Field(default="", max_length=512)
    snippet: str = Field(default="", max_length=4000)
    published_at: str | None = Field(default=None, max_length=64)


class GatewayWebSearch(ContractModel):
    """The caller's normalized request for a pre-answer web search.

    Present on a request only when the caller asked for it. It changes the
    answer for the same body, so it joins replay identity through
    :func:`exp.runtime.gateway.replay_identity.canonical_request_sha256`
    (a reused operation key with a different search object is a conflict).
    The fetched results never join identity: they are derived at admission.
    """

    declared_as: WebSearchDeclaredAs
    max_results: int = Field(
        default=DEFAULT_WEB_SEARCH_RESULTS, ge=1, le=MAXIMUM_WEB_SEARCH_RESULTS
    )
    search_context_size: Literal["low", "medium", "high"] | None = None
    allowed_domains: tuple[str, ...] = Field(default=(), max_length=100)
    blocked_domains: tuple[str, ...] = Field(default=(), max_length=100)
    user_location: JsonObject | None = None
    """Verbatim caller location hint; carried for parity, not used to filter."""
    search_prompt: str | None = Field(default=None, max_length=2000)
    """OpenRouter's override for the instruction that frames the results."""
    max_uses: int | None = Field(default=None, ge=1)
    """Anthropic's per-turn search cap; the gateway runs at most one search."""

    @model_validator(mode="after")
    def _require_one_domain_filter(self) -> GatewayWebSearch:
        """Reject a search that both allows and blocks domains (provider rule)."""
        if self.allowed_domains and self.blocked_domains:
            raise ValueError("allowed_domains and blocked_domains are mutually exclusive")
        for domain in (*self.allowed_domains, *self.blocked_domains):
            if not domain or len(domain) > 253 or " " in domain:
                raise ValueError("web search domain filters must be host names")
        return self
