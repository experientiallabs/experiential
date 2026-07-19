# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Host-answered connector tools for a live agent session.

A live session's tools are answered by the HOST (the CLI process, or the platform runner), never
by the sandbox: the `ToolExecutor` runs where credentials live, so the agent can reach a service
without the token ever entering the runner. This module packages the GitHub connector as one such
tool, `github_search`: a `ToolSpec` object the caller appends to the session's tool list, a cheap
availability check, and a fetch that resolves the token host-side, pulls via `wmh.connect`, and
renders the result into one capped observation.

Placement note: this lives with the harness (not `wmh.connect`) because the shape it produces is
harness machinery, a `ToolSpec` plus a `ToolOutcome`. `wmh.connect` stays a standalone context
library with no dependency on the harness; the arrow points harness -> connect, never back.
"""

from __future__ import annotations

from wmh.connect import (
    ConnectError,
    ConnectorAuth,
    PullQuery,
    get_connector,
    load_connector_auth,
    token_env_vars,
)
from wmh.connect.types import ContextItem
from wmh.core.types import JsonObject, JsonValue
from wmh.harness.live_session import ToolOutcome
from wmh.harness.tools import ToolSpec

_CONNECTOR = "github"

# One github_search observation is capped so a large repo pull cannot blow up the agent's context.
# The connector does not cap individual item bodies, so this cap governs the whole rendering: whole
# items are dropped from the tail, and a single item larger than the cap is hard-truncated.
_OBSERVATION_CAP_CHARS = 24_000
# The most items one search may pull, regardless of the model-supplied `limit`: a live agent is
# reading, not archiving, so a tight cap keeps one tool call cheap and the observation bounded.
_MAX_LIMIT = 30
_DEFAULT_LIMIT = 10

GITHUB_SEARCH = ToolSpec(
    name="github_search",
    description=(
        "Search a GitHub repository's issues, pull requests, and README, newest first. "
        "Answered by the host from its configured GitHub credential; the repo must be one that "
        "credential can see."
    ),
    arguments={
        "target": "the repository as 'owner/repo' (required)",
        "query": "optional GitHub issue-search text (e.g. 'label:bug crash'); omit for the "
        "newest issues and pull requests",
        "since": "optional ISO-8601 lower bound on item update time (e.g. '2026-01-01')",
        "limit": f"optional max items to return (1-{_MAX_LIMIT}, default {_DEFAULT_LIMIT})",
    },
)


def github_search_available() -> bool:
    """Whether a GitHub credential resolves host-side, so the tool would not always error.

    True when a token resolves from the environment (`WMH_GITHUB_TOKEN`) or from the stored
    connector credential file. The tool is offered to an agent only when this holds.
    """
    try:
        return load_connector_auth(_CONNECTOR) is not None
    except ConnectError:
        # A corrupt stored credential still means "configured"; the fetch surfaces the real
        # error rather than silently hiding a tool the user meant to have.
        return True


def _coerce_limit(raw: JsonValue | None) -> int:
    """Clamp a model-supplied limit into `[1, _MAX_LIMIT]`, defaulting when absent or invalid."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return _DEFAULT_LIMIT
    value = int(raw)
    if value < 1:
        return _DEFAULT_LIMIT
    return min(value, _MAX_LIMIT)


def _opt_arg(raw: JsonValue | None) -> str | None:
    """A tool argument as a non-empty stripped string, else None."""
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    return stripped or None


def _render_items(items: list[ContextItem]) -> str:
    """Render pulled items into one capped observation, newest first, truncation made visible.

    Each item becomes a compact block (title, a kind/state/date/url fact line, then its body).
    When the assembled text would exceed the observation cap, whole items are dropped from the
    tail and a final "... n more items omitted" line keeps the truncation loud, never silent.
    """
    if not items:
        return "no matching issues, pull requests, or README found"
    blocks = [_render_item(item) for item in items]
    full = "\n\n".join(blocks)
    if len(full) <= _OBSERVATION_CAP_CHARS:
        return full
    for kept in range(len(blocks) - 1, 0, -1):
        omitted = len(blocks) - kept
        candidate = "\n\n".join([*blocks[:kept], f"... {omitted} more items omitted"])
        if len(candidate) <= _OBSERVATION_CAP_CHARS:
            return candidate
    # Even the first item alone is over budget; return it hard-truncated with a loud marker,
    # reserving room for that marker so the whole observation still fits the cap.
    marker = f"\n... item truncated and {len(blocks) - 1} more items omitted"
    head = blocks[0][: max(0, _OBSERVATION_CAP_CHARS - len(marker))]
    return f"{head}{marker}"


def _render_item(item: ContextItem) -> str:
    """One item block: its title, a kind/state/date/url fact line, then its body."""
    facts: list[str] = [item.kind.value]
    state = item.metadata.get("state")
    if isinstance(state, str) and state:
        facts.append(state)
    if item.updated_at:
        facts.append(f"updated {item.updated_at}")
    if item.url:
        facts.append(item.url)
    block = f"### {item.title}\n{' | '.join(facts)}"
    body = item.body.strip()
    if body:
        block += f"\n\n{body}"
    return block


def github_search_fetch(args: JsonObject) -> ToolOutcome:
    """Answer one `github_search` call host-side, returning the rendered observation.

    Resolves the GitHub token from the host environment / stored credential, builds a capped
    `PullQuery`, pulls via `wmh.connect.get_connector("github")`, and renders the items into one
    capped observation. A missing credential or a `ConnectError` becomes a clean `is_error`
    `ToolOutcome`; connector internals never leak to the agent.

    Args:
        args: The model-supplied tool arguments (`target`, optional `query`/`since`/`limit`).

    Returns:
        A `ToolOutcome`: the rendered observation on success, or an actionable error on failure.
    """
    target = _opt_arg(args.get("target"))
    if target is None:
        return ToolOutcome(
            content="github_search needs a 'target' repository as 'owner/repo'", is_error=True
        )
    auth = _resolve_auth()
    if auth is None:
        hint = " or ".join(f"${var}" for var in token_env_vars(_CONNECTOR))
        return ToolOutcome(
            content=(
                f"github_search is not configured: set {hint} on the machine running the "
                "CLI before starting the session"
            ),
            is_error=True,
        )
    query = PullQuery(
        target=target,
        query=_opt_arg(args.get("query")),
        since=_opt_arg(args.get("since")),
        limit=_coerce_limit(args.get("limit")),
    )
    try:
        items = get_connector(_CONNECTOR).pull(auth, query)
    except ConnectError as error:
        return ToolOutcome(content=f"github_search failed: {error}", is_error=True)
    return ToolOutcome(content=_render_items(items))


def _resolve_auth() -> ConnectorAuth | None:
    """The host-side GitHub credential, or None when a corrupt stored one still cannot be used."""
    try:
        return load_connector_auth(_CONNECTOR)
    except ConnectError:
        return None
