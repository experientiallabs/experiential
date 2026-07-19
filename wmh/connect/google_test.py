"""Tests for the Google context connectors (Calendar, Drive, Gmail).

Every HTTP interaction goes through httpx.MockTransport with fixture payloads shaped from
Google's documented API responses; nothing touches the network or a real clock.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl

import httpx
import pytest

import wmh.connect.google as google_mod
from wmh.connect.connector import ConnectUI, get_connector
from wmh.connect.credentials import ENV_CONNECTORS_PATH, load_connector_auth, token_env_var
from wmh.connect.google import (
    CALENDAR_SCOPE,
    DRIVE_SCOPE,
    GMAIL_SCOPE,
    GmailConnector,
    GoogleCalendarConnector,
    GoogleDriveConnector,
    _GoogleConnector,
)
from wmh.connect.oauth import OAuthApp
from wmh.connect.types import ConnectError, ConnectorAuth, ItemKind, PullQuery
from wmh.core.types import JsonObject


def _auth() -> ConnectorAuth:
    return ConnectorAuth(kind="oauth", access_token="ya29.token")


def _b64url(text: str) -> str:
    """Base64url without padding, the way Gmail encodes message part bodies."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _ui() -> ConnectUI:
    return ConnectUI(
        open_url=lambda _url: None,
        present_code=lambda _uri, _code: None,
        prompt_secret=lambda _label: "",
        info=lambda _message: None,
    )


# -- registration -------------------------------------------------------------------------------


def test_connectors_register_on_import() -> None:
    for name, label in (
        ("google-calendar", "Google Calendar"),
        ("google-drive", "Google Drive"),
        ("gmail", "Gmail"),
    ):
        connector = get_connector(name)
        assert connector.name == name
        assert connector.label == label


# -- verify -------------------------------------------------------------------------------------


def test_calendar_verify_returns_primary_calendar_summary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer ya29.token"
        assert request.url.host == "www.googleapis.com"
        assert request.url.path == "/calendar/v3/calendars/primary"
        return httpx.Response(200, json={"id": "primary", "summary": "kion@example.com"})

    connector = GoogleCalendarConnector(transport=httpx.MockTransport(handler))
    assert connector.verify(_auth()) == "kion@example.com"


def test_drive_verify_returns_display_name_and_email() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/drive/v3/about"
        assert request.url.params["fields"] == "user"
        return httpx.Response(
            200, json={"user": {"displayName": "Kion F", "emailAddress": "kion@example.com"}}
        )

    connector = GoogleDriveConnector(transport=httpx.MockTransport(handler))
    assert connector.verify(_auth()) == "Kion F (kion@example.com)"


def test_gmail_verify_returns_email_address() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "gmail.googleapis.com"
        assert request.url.path == "/gmail/v1/users/me/profile"
        return httpx.Response(200, json={"emailAddress": "kion@example.com", "messagesTotal": 42})

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    assert connector.verify(_auth()) == "kion@example.com"


@pytest.mark.parametrize(
    "connector_cls", [GoogleCalendarConnector, GoogleDriveConnector, GmailConnector]
)
def test_verify_401_tells_the_user_to_reconnect(connector_cls: type[_GoogleConnector]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 401, "message": "Invalid Credentials"}})

    connector = connector_cls(transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        connector.verify(_auth())


def test_transport_failures_become_actionable_connect_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out", request=request)

    connector = GoogleDriveConnector(transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectError, match=r"www\.googleapis\.com.*network"):
        connector.pull(_auth(), PullQuery(limit=5))


# -- calendar pull ------------------------------------------------------------------------------

_EVENT_PLANNING: JsonObject = {
    "id": "evt1",
    "status": "confirmed",
    "summary": "Sprint planning",
    "description": "Plan the sprint.",
    "location": "Room 4",
    "htmlLink": "https://www.google.com/calendar/event?eid=evt1",
    "created": "2026-06-30T08:00:00.000Z",
    "updated": "2026-07-01T09:00:00.000Z",
    "start": {"dateTime": "2026-07-02T10:00:00+02:00"},
    "end": {"dateTime": "2026-07-02T11:00:00+02:00"},
    "attendees": [
        {"displayName": "Ada Lovelace", "email": "ada@example.com"},
        {"email": "bob@example.com"},
    ],
}

_EVENT_ALL_DAY: JsonObject = {
    "id": "evt2",
    "htmlLink": "https://www.google.com/calendar/event?eid=evt2",
    "start": {"date": "2026-07-10"},
    "end": {"date": "2026-07-11"},
}


def test_calendar_pull_paginates_and_normalizes_events() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer ya29.token"
        assert request.url.path == "/calendar/v3/calendars/primary/events"
        if request.url.params.get("pageToken") == "page2":
            return httpx.Response(200, json={"items": [_EVENT_ALL_DAY]})
        return httpx.Response(200, json={"items": [_EVENT_PLANNING], "nextPageToken": "page2"})

    connector = GoogleCalendarConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(since="2026-07-01", limit=50))

    assert len(seen) == 2
    first = seen[0].url.params
    assert first["singleEvents"] == "true"
    assert first["orderBy"] == "startTime"
    assert first["timeMin"] == "2026-07-01T00:00:00+00:00"
    assert first.get("timeMax")  # defaulted window upper bound
    assert first["maxResults"] == "50"
    assert "pageToken" not in first
    assert seen[1].url.params["pageToken"] == "page2"
    assert seen[1].url.params["maxResults"] == "49"

    assert [item.id for item in items] == ["evt1", "evt2"]
    planning = items[0]
    assert planning.source == "google-calendar"
    assert planning.kind is ItemKind.EVENT
    assert planning.title == "Sprint planning"
    assert planning.url == "https://www.google.com/calendar/event?eid=evt1"
    assert planning.created_at == "2026-06-30T08:00:00.000Z"
    assert planning.updated_at == "2026-07-01T09:00:00.000Z"
    assert planning.body == (
        "When: 2026-07-02T10:00:00+02:00 to 2026-07-02T11:00:00+02:00\n"
        "Location: Room 4\n"
        "Attendees: Ada Lovelace <ada@example.com>, bob@example.com\n"
        "\n"
        "Plan the sprint."
    )
    assert planning.metadata["calendar"] == "primary"
    all_day = items[1]
    assert all_day.title == "(no title)"
    assert all_day.body == "When: 2026-07-10 to 2026-07-11"


def test_calendar_pull_stops_at_limit_without_fetching_more_pages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/calendar/v3/calendars/team@example.com/events"
        return httpx.Response(200, json={"items": [_EVENT_PLANNING], "nextPageToken": "page2"})

    connector = GoogleCalendarConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(target="team@example.com", limit=1))

    assert len(items) == 1
    assert len(seen) == 1
    assert seen[0].url.params["maxResults"] == "1"


# -- drive pull ---------------------------------------------------------------------------------

_DRIVE_LISTING: JsonObject = {
    "files": [
        {
            "id": "doc1",
            "name": "Roadmap",
            "mimeType": "application/vnd.google-apps.document",
            "createdTime": "2026-05-01T12:00:00.000Z",
            "modifiedTime": "2026-07-01T12:00:00.000Z",
            "webViewLink": "https://docs.google.com/document/d/doc1/edit",
        },
        {
            "id": "sheet1",
            "name": "Budget",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "createdTime": "2026-05-02T00:00:00.000Z",
            "modifiedTime": "2026-06-30T00:00:00.000Z",
            "webViewLink": "https://docs.google.com/spreadsheets/d/sheet1/edit",
        },
        {
            "id": "txt1",
            "name": "notes.txt",
            "mimeType": "text/plain",
            "createdTime": "2026-06-01T00:00:00.000Z",
            "modifiedTime": "2026-06-02T00:00:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/txt1/view",
            "size": "18",
        },
        {
            "id": "bin1",
            "name": "logo.png",
            "mimeType": "image/png",
            "createdTime": "2026-04-01T00:00:00.000Z",
            "modifiedTime": "2026-04-02T00:00:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/bin1/view",
            "size": "2048",
        },
    ]
}


def test_drive_pull_exports_docs_downloads_text_and_skips_binaries() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/drive/v3/files":
            params = request.url.params
            assert params["q"] == "trashed=false and fullText contains 'road\\'map'"
            assert params["orderBy"] == "modifiedTime desc"
            assert params["pageSize"] == "100"
            assert (
                "files(id,name,mimeType,modifiedTime,createdTime,webViewLink,size)"
                in (params["fields"])
            )
            return httpx.Response(200, json=_DRIVE_LISTING)
        if path == "/drive/v3/files/doc1/export":
            assert request.url.params["mimeType"] == "text/plain"
            return httpx.Response(200, text="Exported doc body.")
        if path == "/drive/v3/files/sheet1/export":
            # Sheets have no text/plain export; the connector must ask for CSV.
            assert request.url.params["mimeType"] == "text/csv"
            return httpx.Response(200, text="month,total\nJune,42")
        if path == "/drive/v3/files/txt1":
            assert request.url.params["alt"] == "media"
            return httpx.Response(200, text="plain text contents")
        raise AssertionError(f"unexpected request: {request.url}")

    connector = GoogleDriveConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(query="road'map"))

    assert [request.url.path for request in seen] == [
        "/drive/v3/files",
        "/drive/v3/files/doc1/export",
        "/drive/v3/files/sheet1/export",
        "/drive/v3/files/txt1",
    ]
    assert [item.id for item in items] == ["doc1", "sheet1", "txt1", "bin1"]

    doc = items[0]
    assert doc.source == "google-drive"
    assert doc.kind is ItemKind.DOCUMENT
    assert doc.title == "Roadmap"
    assert doc.body == "Exported doc body."
    assert doc.url == "https://docs.google.com/document/d/doc1/edit"
    assert doc.created_at == "2026-05-01T12:00:00.000Z"
    assert doc.updated_at == "2026-07-01T12:00:00.000Z"
    assert doc.metadata["mimeType"] == "application/vnd.google-apps.document"

    sheet = items[1]
    assert sheet.kind is ItemKind.DOCUMENT
    assert sheet.body == "month,total\nJune,42"
    assert sheet.metadata["mimeType"] == "application/vnd.google-apps.spreadsheet"
    assert "fetch_error" not in sheet.metadata

    text = items[2]
    assert text.kind is ItemKind.DOCUMENT
    assert text.body == "plain text contents"
    assert text.metadata["size"] == "18"

    binary = items[3]
    assert binary.kind is ItemKind.FILE
    assert binary.body == ""
    assert binary.metadata["mimeType"] == "image/png"


def test_drive_pull_scopes_query_to_folder_and_since() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/drive/v3/files"
        return httpx.Response(200, json={"files": []})

    connector = GoogleDriveConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(target="folder123", since="2026-06-01"))

    assert items == []
    assert seen[0].url.params["q"] == (
        "trashed=false and 'folder123' in parents and modifiedTime > '2026-06-01T00:00:00+00:00'"
    )


def test_drive_pull_caps_exported_content() -> None:
    big = "x" * 200_050

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/drive/v3/files":
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "id": "doc1",
                            "name": "Huge",
                            "mimeType": "application/vnd.google-apps.document",
                        }
                    ]
                },
            )
        return httpx.Response(200, text=big)

    connector = GoogleDriveConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(limit=1))

    body = items[0].body
    assert len(body) < len(big)
    assert body.startswith("xxxx")
    assert body.endswith("[content truncated at 200000 characters]")


def test_drive_pull_respects_limit_across_pages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "files": [{"id": f"bin{len(seen)}", "name": "a.png", "mimeType": "image/png"}],
                "nextPageToken": "more",
            },
        )

    connector = GoogleDriveConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(limit=2))

    assert [item.id for item in items] == ["bin1", "bin2"]
    assert len(seen) == 2
    assert seen[1].url.params["pageToken"] == "more"
    assert seen[1].url.params["pageSize"] == "1"


# -- gmail pull ---------------------------------------------------------------------------------

_INTERNAL_DATE_MS = 1751623200000

_GMAIL_MULTIPART: JsonObject = {
    "id": "m1",
    "threadId": "t1",
    "labelIds": ["INBOX", "IMPORTANT"],
    "internalDate": str(_INTERNAL_DATE_MS),
    "payload": {
        "mimeType": "multipart/mixed",
        "headers": [
            {"name": "Subject", "value": "Deploy failed"},
            {"name": "From", "value": "Ops Bot <ops@example.com>"},
            {"name": "To", "value": "kion@example.com"},
            {"name": "Date", "value": "Fri, 4 Jul 2025 10:00:00 +0000"},
        ],
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    # html first: the connector must still prefer the text/plain part
                    {
                        "mimeType": "text/html",
                        "body": {"data": _b64url("<p>Deploy 42 failed (html)</p>")},
                    },
                    {
                        "mimeType": "text/plain",
                        "body": {
                            "data": _b64url("Deploy 42 failed on step build.\nRetry scheduled.")
                        },
                    },
                ],
            },
            {
                "mimeType": "application/pdf",
                "filename": "log.pdf",
                "body": {"attachmentId": "att1", "size": 1024},
            },
        ],
    },
}

_GMAIL_HTML_ONLY: JsonObject = {
    "id": "m2",
    "threadId": "t2",
    "labelIds": ["INBOX"],
    "internalDate": str(_INTERNAL_DATE_MS),
    "payload": {
        "mimeType": "text/html",
        "headers": [
            {"name": "Subject", "value": "Newsletter"},
            {"name": "From", "value": "news@example.com"},
            {"name": "To", "value": "kion@example.com"},
            {"name": "Date", "value": "Fri, 4 Jul 2025 11:00:00 +0000"},
        ],
        "body": {
            "data": _b64url("<html><body><p>HTML only body</p><p>Second para</p></body></html>")
        },
    },
}


def test_gmail_pull_decodes_recursive_multipart_and_falls_back_to_html() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.host == "gmail.googleapis.com"
        path = request.url.path
        if path == "/gmail/v1/users/me/messages":
            assert request.url.params["q"] == (
                "from:ops@example.com after:2026/06/01 before:2026/07/01"
            )
            return httpx.Response(
                200,
                json={
                    "messages": [{"id": "m1", "threadId": "t1"}, {"id": "m2", "threadId": "t2"}],
                    "resultSizeEstimate": 2,
                },
            )
        if path == "/gmail/v1/users/me/messages/m1":
            assert request.url.params["format"] == "full"
            return httpx.Response(200, json=_GMAIL_MULTIPART)
        if path == "/gmail/v1/users/me/messages/m2":
            return httpx.Response(200, json=_GMAIL_HTML_ONLY)
        raise AssertionError(f"unexpected request: {request.url}")

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(
        _auth(),
        PullQuery(query="from:ops@example.com", since="2026-06-01", until="2026-07-01", limit=10),
    )

    assert [item.id for item in items] == ["m1", "m2"]
    email = items[0]
    assert email.source == "gmail"
    assert email.kind is ItemKind.EMAIL
    assert email.title == "Deploy failed"
    assert email.url == "https://mail.google.com/mail/u/0/#all/m1"
    expected_created = datetime.fromtimestamp(_INTERNAL_DATE_MS / 1000, tz=UTC).isoformat(
        timespec="seconds"
    )
    assert email.created_at == expected_created
    assert email.body == (
        "From: Ops Bot <ops@example.com>\n"
        "To: kion@example.com\n"
        "Date: Fri, 4 Jul 2025 10:00:00 +0000\n"
        "\n"
        "Deploy 42 failed on step build.\n"
        "Retry scheduled."
    )
    assert email.metadata["from"] == "Ops Bot <ops@example.com>"
    assert email.metadata["to"] == "kion@example.com"
    assert email.metadata["labelIds"] == ["INBOX", "IMPORTANT"]
    assert email.metadata["threadId"] == "t1"

    fallback = items[1]
    assert fallback.title == "Newsletter"
    assert fallback.body.endswith("HTML only body\nSecond para")
    assert "<p>" not in fallback.body


def test_gmail_pull_caps_detail_fetches_at_limit() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/gmail/v1/users/me/messages":
            assert request.url.params["maxResults"] == "2"
            assert "q" not in request.url.params
            return httpx.Response(
                200,
                json={"messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]},
            )
        return httpx.Response(200, json=_GMAIL_MULTIPART)

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(limit=2))

    assert len(items) == 2
    detail_paths = [r.url.path for r in seen if r.url.path != "/gmail/v1/users/me/messages"]
    assert detail_paths == ["/gmail/v1/users/me/messages/m1", "/gmail/v1/users/me/messages/m2"]


def test_gmail_pull_paginates_the_message_list() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/gmail/v1/users/me/messages":
            if request.url.params.get("pageToken") == "p2":
                return httpx.Response(200, json={"messages": [{"id": "m2"}]})
            return httpx.Response(200, json={"messages": [{"id": "m1"}], "nextPageToken": "p2"})
        return httpx.Response(200, json=_GMAIL_MULTIPART)

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_auth(), PullQuery(limit=5))

    list_calls = [r for r in seen if r.url.path == "/gmail/v1/users/me/messages"]
    assert len(list_calls) == 2
    assert len(items) == 2


# -- refresh ------------------------------------------------------------------------------------


def _expired_auth() -> ConnectorAuth:
    return ConnectorAuth(
        kind="oauth",
        access_token="stale",
        refresh_token="rt",
        expires_at="2020-01-01T00:00:00+00:00",
    )


def test_pull_refreshes_an_expired_token_and_persists_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.setenv("WMH_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("WMH_GOOGLE_CLIENT_SECRET", "shhh")
    monkeypatch.delenv(token_env_var("gmail"), raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            form = dict(parse_qsl(request.content.decode()))
            assert form["grant_type"] == "refresh_token"
            assert form["refresh_token"] == "rt"
            return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer fresh"
        return httpx.Response(200, json={"resultSizeEstimate": 0})

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    items = connector.pull(_expired_auth(), PullQuery(limit=5))

    assert items == []
    saved = load_connector_auth("gmail")
    assert saved is not None
    assert saved.access_token == "fresh"
    assert saved.refresh_token == "rt"  # kept when the response omits one


def test_refresh_invalid_grant_tells_the_user_to_reconnect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.setenv("WMH_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "oauth2.googleapis.com"
        return httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "Token has been expired or revoked.",
            },
        )

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectError) as excinfo:
        connector.pull(_expired_auth(), PullQuery(limit=5))
    message = str(excinfo.value)
    assert "invalid_grant" in message
    assert "the connection must be reauthorized" in message


def test_pull_with_a_plain_token_needs_no_oauth_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WMH_GOOGLE_CLIENT_ID", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok"
        return httpx.Response(200, json={"items": []})

    connector = GoogleCalendarConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="tok")
    assert connector.pull(auth, PullQuery(limit=5)) == []


# -- connect ------------------------------------------------------------------------------------


def test_connect_requests_the_service_scope_and_labels_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMH_GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
    flows: list[tuple[str, list[str]]] = []

    def fake_flow(
        app: OAuthApp,
        *,
        scopes: list[str] | None = None,
        open_url: Callable[[str], None],
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> ConnectorAuth:
        del open_url, timeout, transport
        flows.append((app.name, list(scopes or [])))
        return ConnectorAuth(kind="oauth", access_token="ya29.token", refresh_token="rt")

    monkeypatch.setattr(google_mod, "run_loopback_flow", fake_flow)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/gmail/v1/users/me/profile"
        return httpx.Response(200, json={"emailAddress": "kion@example.com"})

    connector = GmailConnector(transport=httpx.MockTransport(handler))
    auth = connector.connect(_ui())

    assert flows == [("google", [GMAIL_SCOPE])]
    assert auth.account == "kion@example.com"
    assert auth.refresh_token == "rt"


def test_each_connector_requests_only_its_own_readonly_scope() -> None:
    assert GoogleCalendarConnector.scope == CALENDAR_SCOPE
    assert GoogleDriveConnector.scope == DRIVE_SCOPE
    assert GmailConnector.scope == GMAIL_SCOPE
    assert CALENDAR_SCOPE.endswith("calendar.readonly")
    assert DRIVE_SCOPE.endswith("drive.readonly")
    assert GMAIL_SCOPE.endswith("gmail.readonly")
