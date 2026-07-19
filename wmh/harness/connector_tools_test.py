"""Tests for the host-answered github_search connector tool.

No network: the connector's `pull` is faked (or raised through) and credentials are resolved from
an isolated env/store, so every path (render, no-token, ConnectError, cap) is exercised offline.
"""

from __future__ import annotations

import pytest

from wmh.connect.credentials import ENV_CONNECTORS_PATH
from wmh.connect.types import ConnectError, ConnectorAuth, ContextItem, ItemKind, PullQuery
from wmh.core.types import JsonObject
from wmh.harness import connector_tools as mod


@pytest.fixture(autouse=True)
def _isolated_creds(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate credential resolution: empty tmp store, no GitHub env token by default."""
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path) + "/connectors.toml")  # type: ignore[operator]
    monkeypatch.delenv("WMH_GITHUB_TOKEN", raising=False)


class _FakeConnector:
    """A stand-in github connector recording the pull it received and returning fixed items."""

    name = "github"
    label = "GitHub"

    def __init__(self, items: list[ContextItem]) -> None:
        self._items = items
        self.received: PullQuery | None = None

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Record the query and return the fixed items (no network)."""
        _ = auth
        self.received = query
        return self._items


class _RaisingConnector:
    """A github connector whose pull raises a ConnectError, to test error mapping."""

    name = "github"
    label = "GitHub"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Always raise a ConnectError (with a token-shaped detail we assert is NOT leaked raw)."""
        _ = (auth, query)
        raise ConnectError("github rejected the stored credential during the pull (HTTP 401)")


def _item(number: int, *, body: str = "body text", state: str = "open") -> ContextItem:
    """One issue-kind ContextItem for rendering tests."""
    return ContextItem(
        id=f"octocat/hello#{number}",
        source="github",
        kind=ItemKind.ISSUE,
        title=f"#{number} a bug",
        body=body,
        url=f"https://github.com/octocat/hello/issues/{number}",
        updated_at="2026-07-01T00:00:00Z",
        metadata={"state": state},
    )


def _use_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a GitHub credential resolve host-side via the env token."""
    monkeypatch.setenv("WMH_GITHUB_TOKEN", "gho_test")


# -- availability ----------------------------------------------------------------------------------


def test_available_is_false_without_any_credential() -> None:
    assert mod.github_search_available() is False


def test_available_is_true_with_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)
    assert mod.github_search_available() is True


# -- fetch: success --------------------------------------------------------------------------------


def test_fetch_renders_items_into_one_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)
    fake = _FakeConnector([_item(1), _item(2)])
    monkeypatch.setattr(mod, "get_connector", lambda _name: fake)

    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is False
    assert "#1 a bug" in outcome.content
    assert "#2 a bug" in outcome.content
    assert "issue | open" in outcome.content
    assert "https://github.com/octocat/hello/issues/1" in outcome.content
    assert fake.received is not None
    assert fake.received.target == "octocat/hello"


def test_fetch_passes_query_and_since_and_caps_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)
    fake = _FakeConnector([_item(1)])
    monkeypatch.setattr(mod, "get_connector", lambda _name: fake)

    args: JsonObject = {
        "target": "octocat/hello",
        "query": "label:bug",
        "since": "2026-01-01",
        "limit": 999,
    }
    mod.github_search_fetch(args)

    assert fake.received is not None
    assert fake.received.query == "label:bug"
    assert fake.received.since == "2026-01-01"
    assert fake.received.limit == mod._MAX_LIMIT


def test_fetch_defaults_the_limit_when_missing_or_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_token(monkeypatch)
    fake = _FakeConnector([_item(1)])
    monkeypatch.setattr(mod, "get_connector", lambda _name: fake)

    mod.github_search_fetch({"target": "octocat/hello", "limit": "not-a-number"})

    assert fake.received is not None
    assert fake.received.limit == mod._DEFAULT_LIMIT


def test_fetch_reports_no_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)
    monkeypatch.setattr(mod, "get_connector", lambda _name: _FakeConnector([]))

    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is False
    assert "no matching" in outcome.content


def test_fetch_caps_the_observation_and_makes_truncation_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_token(monkeypatch)
    big = "x" * 20_000
    items = [_item(n, body=big) for n in range(5)]
    monkeypatch.setattr(mod, "get_connector", lambda _name: _FakeConnector(items))

    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is False
    assert len(outcome.content) <= mod._OBSERVATION_CAP_CHARS
    assert "items omitted" in outcome.content


def test_fetch_caps_a_single_oversized_item(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)
    # One item whose body alone exceeds the cap, so the hard-truncate branch runs.
    big = "x" * (mod._OBSERVATION_CAP_CHARS * 2)
    monkeypatch.setattr(mod, "get_connector", lambda _name: _FakeConnector([_item(1, body=big)]))

    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is False
    assert len(outcome.content) <= mod._OBSERVATION_CAP_CHARS
    assert "item truncated" in outcome.content


# -- fetch: errors ---------------------------------------------------------------------------------


def test_fetch_without_a_target_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)

    outcome = mod.github_search_fetch({})

    assert outcome.is_error is True
    assert "target" in outcome.content


def test_fetch_without_a_credential_is_a_clean_error() -> None:
    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is True
    assert "WMH_GITHUB_TOKEN" in outcome.content
    assert "not configured" in outcome.content


def test_fetch_with_a_corrupt_credential_surfaces_the_real_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-unusable credential reports the invalid-credential error, not 'not configured'.

    This is what github_search_available promises when it offers the tool despite a ConnectError.
    """

    def _raise(_connector: str) -> object:
        raise ConnectError("stored github credential is malformed")

    monkeypatch.setattr(mod, "load_connector_auth", _raise)

    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is True
    assert "credential is invalid" in outcome.content
    assert "not configured" not in outcome.content


def test_fetch_maps_connect_error_to_a_clean_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_token(monkeypatch)
    monkeypatch.setattr(mod, "get_connector", lambda _name: _RaisingConnector())

    outcome = mod.github_search_fetch({"target": "octocat/hello"})

    assert outcome.is_error is True
    assert outcome.content.startswith("github_search failed:")
    # The env token itself must never appear in the observation.
    assert "gho_test" not in outcome.content
