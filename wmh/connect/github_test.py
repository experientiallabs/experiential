"""Tests for the GitHub context connector: recorded-fixture payloads through MockTransport."""

from __future__ import annotations

import base64
from collections.abc import Callable
from urllib.parse import parse_qsl

import httpx
import pytest

from wmh.connect.connector import ConnectUI, get_connector
from wmh.connect.github import GitHubConnector
from wmh.connect.types import ConnectError, ConnectorAuth, ItemKind, PullQuery

_AUTH = ConnectorAuth(kind="oauth", access_token="gho_test")

# Row shapes mirror the documented `GET /repos/{owner}/{repo}/issues` response (the combined
# issues + pull requests listing; PR rows carry a "pull_request" key).
_ISSUE_ROW = {
    "id": 1,
    "number": 1347,
    "state": "open",
    "title": "Found a bug",
    "body": "I'm having a problem with this.",
    "user": {"login": "octocat", "id": 1},
    "labels": [{"id": 208045946, "name": "bug", "color": "f29513"}],
    "comments": 12,
    "html_url": "https://github.com/octocat/Hello-World/issues/1347",
    "created_at": "2011-04-22T13:33:48Z",
    "updated_at": "2011-04-25T10:00:00Z",
}

_PR_ROW = {
    "id": 2,
    "number": 1400,
    "state": "closed",
    "title": "Fix the bug",
    "body": "Closes #1347.",
    "user": {"login": "hubber", "id": 2},
    "labels": [],
    "comments": 3,
    "pull_request": {
        "url": "https://api.github.com/repos/octocat/Hello-World/pulls/1400",
        "html_url": "https://github.com/octocat/Hello-World/pull/1400",
    },
    "html_url": "https://github.com/octocat/Hello-World/pull/1400",
    "created_at": "2011-05-01T00:00:00Z",
    "updated_at": "2011-05-02T00:00:00Z",
}

_README_TEXT = "# Hello-World\n\nMy first repository on GitHub!\n"
_README_B64 = base64.b64encode(_README_TEXT.encode()).decode()

# GitHub wraps base64 content in newlines; the connector must decode it anyway.
_README = {
    "type": "file",
    "encoding": "base64",
    "name": "README.md",
    "path": "README.md",
    "content": "\n".join(_README_B64[i : i + 40] for i in range(0, len(_README_B64), 40)),
    "html_url": "https://github.com/octocat/Hello-World/blob/master/README.md",
}


def _connector(handler: Callable[[httpx.Request], httpx.Response]) -> GitHubConnector:
    return GitHubConnector(transport=httpx.MockTransport(handler))


def _unused(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP call: {request.url}")


def test_registers_itself_on_import() -> None:
    connector = get_connector("github")
    assert isinstance(connector, GitHubConnector)
    assert connector.label == "GitHub"


def test_connect_runs_the_device_flow_and_stamps_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMH_GITHUB_CLIENT_ID", "cid_test")
    monkeypatch.delenv("WMH_GITHUB_CLIENT_SECRET", raising=False)
    device_requests: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://github.com/login/device/code":
            device_requests.append(dict(parse_qsl(request.content.decode())))
            return httpx.Response(
                200,
                json={
                    "device_code": "dev123",
                    "user_code": "WDJB-MJHT",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 899,
                    "interval": 0,
                },
            )
        if url == "https://github.com/login/oauth/access_token":
            body = dict(parse_qsl(request.content.decode()))
            assert body["device_code"] == "dev123"
            return httpx.Response(
                200,
                json={"access_token": "gho_new", "token_type": "bearer", "scope": "repo,read:org"},
            )
        assert url == "https://api.github.com/user"
        assert request.headers["Authorization"] == "Bearer gho_new"
        return httpx.Response(200, json={"login": "octocat", "name": "The Octocat"})

    presented: list[tuple[str, str]] = []
    infos: list[str] = []
    ui = ConnectUI(
        open_url=lambda url: pytest.fail(f"the device flow must not open {url}"),
        present_code=lambda uri, code: presented.append((uri, code)),
        prompt_secret=lambda label: pytest.fail(f"the device flow must not prompt for {label}"),
        info=infos.append,
    )

    auth = _connector(handler).connect(ui)

    # GitHub Apps carry permissions in the registration: the device request sends no scope.
    assert device_requests == [{"client_id": "cid_test"}]
    assert presented == [("https://github.com/login/device", "WDJB-MJHT")]
    assert auth.kind == "oauth"
    assert auth.access_token == "gho_new"
    assert auth.scopes == ["repo", "read:org"]
    assert auth.account == "octocat (The Octocat)"
    assert any("octocat" in message for message in infos)
    assert any("Install App" in message for message in infos)


def test_verify_returns_login_and_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.github.com/user"
        assert request.headers["Authorization"] == "Bearer gho_test"
        assert request.headers["Accept"] == "application/vnd.github+json"
        assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
        return httpx.Response(200, json={"login": "octocat", "name": "The Octocat"})

    assert _connector(handler).verify(_AUTH) == "octocat (The Octocat)"


def test_verify_falls_back_to_the_bare_login() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"login": "octocat", "name": None})

    assert _connector(handler).verify(_AUTH) == "octocat"


def test_verify_maps_401_to_a_reconnect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        _connector(handler).verify(_AUTH)


@pytest.mark.parametrize("target", [None, "", "just-a-name", "too/many/parts", "owner/ "])
def test_pull_requires_an_owner_repo_target(target: str | None) -> None:
    with pytest.raises(ConnectError, match="owner/repo"):
        _connector(_unused).pull(_AUTH, PullQuery(target=target))


def test_pull_normalizes_issues_pull_requests_and_the_readme() -> None:
    seen_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/octocat/Hello-World/issues":
            seen_params.update(dict(request.url.params))
            return httpx.Response(200, json=[_PR_ROW, _ISSUE_ROW])
        assert request.url.path == "/repos/octocat/Hello-World/readme"
        return httpx.Response(200, json=_README)

    items = _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World"))

    assert seen_params["state"] == "all"
    assert seen_params["sort"] == "updated"
    assert seen_params["direction"] == "desc"
    assert seen_params["per_page"] == "100"
    assert "since" not in seen_params

    kinds = [item.kind for item in items]
    assert kinds == [ItemKind.PULL_REQUEST, ItemKind.ISSUE, ItemKind.DOCUMENT]
    pr, issue, readme = items

    assert pr.id == "octocat/Hello-World#1400"
    assert pr.title == "#1400 Fix the bug"
    assert pr.url == "https://github.com/octocat/Hello-World/pull/1400"
    assert pr.metadata["state"] == "closed"

    assert issue.id == "octocat/Hello-World#1347"
    assert issue.source == "github"
    assert issue.title == "#1347 Found a bug"
    assert issue.body == "I'm having a problem with this."
    assert issue.url == "https://github.com/octocat/Hello-World/issues/1347"
    assert issue.created_at == "2011-04-22T13:33:48Z"
    assert issue.updated_at == "2011-04-25T10:00:00Z"
    assert issue.metadata == {
        "state": "open",
        "labels": ["bug"],
        "author": "octocat",
        "comments": 12,
    }

    assert readme.id == "octocat/Hello-World:README.md"
    assert readme.source == "github"
    assert readme.title == "README.md"
    assert readme.body == _README_TEXT
    assert readme.url == "https://github.com/octocat/Hello-World/blob/master/README.md"


def test_pull_paginates_via_the_link_header_and_caps_at_the_limit() -> None:
    calls: list[str] = []
    page_two = "https://api.github.com/repositories/1296269/issues?page=2&per_page=3"
    page_last = "https://api.github.com/repositories/1296269/issues?page=9&per_page=3"

    def _row(number: int) -> dict[str, object]:
        return {**_ISSUE_ROW, "id": number, "number": number, "title": f"Issue {number}"}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/repos/octocat/Hello-World/issues":
            params = dict(request.url.params)
            assert params["since"] == "2026-01-01T00:00:00Z"
            assert params["per_page"] == "3"
            link = f'<{page_two}>; rel="next", <{page_last}>; rel="last"'
            return httpx.Response(200, json=[_row(5), _row(4)], headers={"Link": link})
        assert str(request.url) == page_two
        return httpx.Response(200, json=[_row(3), _row(2)])

    items = _connector(handler).pull(
        _AUTH,
        PullQuery(target="octocat/Hello-World", since="2026-01-01T00:00:00Z", limit=3),
    )

    assert [item.id.rpartition("#")[2] for item in items] == ["5", "4", "3"]
    # The limit was reached after page two: no third page and no README fetch.
    assert len(calls) == 2


def test_pull_stops_at_the_limit_without_following_the_next_link() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octocat/Hello-World/issues"
        link = '<https://api.github.com/repositories/1296269/issues?page=2>; rel="next"'
        return httpx.Response(200, json=[_PR_ROW, _ISSUE_ROW], headers={"Link": link})

    items = _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World", limit=2))
    assert len(items) == 2


def test_pull_skips_a_missing_readme() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/octocat/Hello-World/issues":
            return httpx.Response(200, json=[_ISSUE_ROW])
        assert request.url.path == "/repos/octocat/Hello-World/readme"
        return httpx.Response(404, json={"message": "Not Found"})

    items = _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World"))
    assert [item.kind for item in items] == [ItemKind.ISSUE]


def test_pull_routes_query_through_the_search_api() -> None:
    searches: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/issues":
            searches.append(dict(request.url.params))
            payload = {"total_count": 1, "incomplete_results": False, "items": [_ISSUE_ROW]}
            return httpx.Response(200, json=payload)
        assert request.url.path == "/repos/octocat/Hello-World/readme"
        return httpx.Response(404, json={"message": "Not Found"})

    items = _connector(handler).pull(
        _AUTH,
        PullQuery(target="octocat/Hello-World", query="label:bug crash", since="2026-01-01"),
    )

    assert searches == [
        {
            "q": "repo:octocat/Hello-World label:bug crash updated:>=2026-01-01",
            "sort": "updated",
            "order": "desc",
            "per_page": "100",
        }
    ]
    assert [item.kind for item in items] == [ItemKind.ISSUE]
    assert items[0].id == "octocat/Hello-World#1347"


def test_pull_maps_401_to_a_reconnect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World"))


def test_transport_failures_become_actionable_connect_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ConnectError, match=r"api\.github\.com.*network"):
        _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World"))


def test_pull_surfaces_the_rate_limit_reset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1789000000"},
        )

    with pytest.raises(ConnectError, match="X-RateLimit-Reset 1789000000"):
        _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World"))


def test_pull_names_the_missing_repo_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(ConnectError, match="octocat/Hello-World"):
        _connector(handler).pull(_AUTH, PullQuery(target="octocat/Hello-World"))
