"""Google context connectors: Calendar, Drive, and Gmail read-only pulls.

All three connectors share one Google OAuth app (`get_app("google")`) and the browser loopback
flow, differing only in the scope they request and the account label they report. Each keeps its
own stored credential (one consent per service in v1) and refreshes it through `ensure_fresh`
before every pull so refresh tokens keep working unattended.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime, timedelta
from typing import cast
from urllib.parse import quote

import httpx

from wmh.connect.apps import get_app
from wmh.connect.connector import ConnectUI, register_connector
from wmh.connect.oauth import ensure_fresh, run_loopback_flow
from wmh.connect.types import (
    ConnectError,
    ConnectorAuth,
    ContextItem,
    ItemKind,
    PullQuery,
    capped,
    opt_str,
    strip_html,
    transport_errors,
)
from wmh.core.types import JsonObject, JsonValue

logger = logging.getLogger(__name__)

_APP_NAME = "google"
_TIMEOUT_SECONDS = 30.0

_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"
_DRIVE_BASE = "https://www.googleapis.com/drive/v3"
_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"

CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# Per-request page sizes, bounded well within each API's documented maximum.
_CALENDAR_PAGE_MAX = 250
_DRIVE_PAGE_MAX = 100
_GMAIL_PAGE_MAX = 100

# Calendar pulls default to a window around now when since/until are not given.
_DEFAULT_LOOKBACK = timedelta(days=30)
_DEFAULT_LOOKAHEAD = timedelta(days=60)

_DRIVE_LIST_FIELDS = (
    "nextPageToken,files(id,name,mimeType,modifiedTime,createdTime,webViewLink,size)"
)

# Google-native types Drive can export as text, mapped to the export mime each one supports.
# Spreadsheets have no text/plain export; text/csv (the first sheet only) is the text form.
_EXPORT_MIME_BY_TYPE: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
}


class _GoogleConnector:
    """Shared auth, refresh, and HTTP plumbing for the three Google service connectors.

    Subclasses set `name`/`label`/`scope` and implement `verify` and `pull`; everything that
    talks OAuth or carries the bearer token lives here.
    """

    name: str
    label: str
    scope: str

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        """`transport` is injected into every httpx call; None means the real network."""
        self._transport = transport

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        """Run the browser loopback OAuth flow for this service's single read-only scope.

        Args:
            ui: Presentation callbacks; only `open_url` and `info` are used.

        Returns:
            The oauth-kind credential with `account` set to the verified identity.
        """
        app = get_app(_APP_NAME)
        ui.info(f"Requesting read-only Google access for {self.label} ({self.scope}).")
        auth = run_loopback_flow(
            app, scopes=[self.scope], open_url=ui.open_url, transport=self._transport
        )
        account = self.verify(auth)
        ui.info(f"Connected {self.label} as {account}.")
        return auth.model_copy(update={"account": account})

    def verify(self, auth: ConnectorAuth) -> str:
        """Cheap identity check; implemented per service."""
        raise NotImplementedError

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Fetch normalized items; implemented per service."""
        raise NotImplementedError

    def _fresh(self, auth: ConnectorAuth) -> ConnectorAuth:
        """Refresh the credential when it is about to expire, persisting the result.

        Consults the OAuth app config only when a refresh is actually possible, so plain
        env-injected tokens work without any Google OAuth client configured.

        Raises:
            ConnectError: When the provider rejects the refresh (e.g. `invalid_grant` after
                the user revoked access); the message says the connection must be reauthorized.
        """
        if not auth.refresh_token or not auth.expires_at:
            return auth
        try:
            return ensure_fresh(get_app(_APP_NAME), self.name, auth, transport=self._transport)
        except ConnectError as exc:
            raise ConnectError(
                f"could not refresh the {self.label} credential ({exc}); "
                "the token is invalid or expired and the connection must be reauthorized"
            ) from exc

    def _client(self) -> httpx.Client:
        """A short-lived HTTP client carrying the injected transport."""
        return httpx.Client(timeout=_TIMEOUT_SECONDS, transport=self._transport)

    def _get(
        self,
        client: httpx.Client,
        auth: ConnectorAuth,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """One authorized GET; 401s and transport failures become actionable ConnectErrors."""
        with transport_errors(httpx.URL(url).host):
            response = client.get(
                url, params=params, headers={"Authorization": f"Bearer {auth.access_token}"}
            )
        if response.status_code == 401:
            raise ConnectError(
                f"{self.label} rejected the stored credential (HTTP 401); "
                "the token is invalid or expired and the connection must be reauthorized"
            )
        return response

    def _get_json(
        self,
        client: httpx.Client,
        auth: ConnectorAuth,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> JsonObject:
        """An authorized GET that must return a JSON object, else raises ConnectError."""
        response = self._get(client, auth, url, params=params)
        if response.status_code != 200:
            raise ConnectError(
                f"{self.label} request to {url} failed "
                f"(HTTP {response.status_code}): {response.text[:200]}; "
                "check the target/query and retry"
            )
        try:
            raw = response.json()
        except ValueError:
            raw = None
        if not isinstance(raw, dict):
            raise ConnectError(
                f"{self.label} returned an unexpected non-object response from {url}; retry later"
            )
        return cast(JsonObject, raw)


class GoogleCalendarConnector(_GoogleConnector):
    """Pulls calendar events, normalized as EVENT items with a human-readable detail block."""

    name = "google-calendar"
    label = "Google Calendar"
    scope = CALENDAR_SCOPE

    def verify(self, auth: ConnectorAuth) -> str:
        """The primary calendar's summary (usually the account's email address)."""
        with self._client() as client:
            payload = self._get_json(client, auth, f"{_CALENDAR_BASE}/calendars/primary")
        return opt_str(payload.get("summary")) or "primary calendar"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """List events from `query.target` (default "primary") within the query window.

        Defaults to now-30d .. now+60d when `since`/`until` are unset; expands recurring
        events (`singleEvents=true`) and paginates `nextPageToken` up to `query.limit`.
        """
        auth = self._fresh(auth)
        calendar_id = query.target or "primary"
        now = datetime.now(UTC)
        time_min = (
            _rfc3339(query.since, field="since") if query.since else _iso(now - _DEFAULT_LOOKBACK)
        )
        time_max = (
            _rfc3339(query.until, field="until") if query.until else _iso(now + _DEFAULT_LOOKAHEAD)
        )
        url = f"{_CALENDAR_BASE}/calendars/{quote(calendar_id, safe='@.')}/events"
        items: list[ContextItem] = []
        page_token: str | None = None
        with self._client() as client:
            while len(items) < query.limit:
                params: dict[str, str] = {
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "maxResults": str(min(query.limit - len(items), _CALENDAR_PAGE_MAX)),
                }
                if query.query:
                    params["q"] = query.query
                if page_token:
                    params["pageToken"] = page_token
                payload = self._get_json(client, auth, url, params=params)
                for event in _dicts(payload.get("items")):
                    item = self._event_item(event, calendar_id)
                    if item is not None:
                        items.append(item)
                    if len(items) >= query.limit:
                        break
                page_token = opt_str(payload.get("nextPageToken"))
                if not page_token:
                    break
        return items

    def _event_item(self, event: JsonObject, calendar_id: str) -> ContextItem | None:
        """Normalize one API event; events without an id are skipped."""
        event_id = opt_str(event.get("id"))
        if not event_id:
            logger.debug("skipping calendar event without an id: %r", event)
            return None
        metadata: JsonObject = {"calendar": calendar_id}
        status = opt_str(event.get("status"))
        if status:
            metadata["status"] = status
        return ContextItem(
            id=event_id,
            source=self.name,
            kind=ItemKind.EVENT,
            title=opt_str(event.get("summary")) or "(no title)",
            body=_event_body(event),
            url=opt_str(event.get("htmlLink")),
            created_at=opt_str(event.get("created")),
            updated_at=opt_str(event.get("updated")),
            metadata=metadata,
        )


class GoogleDriveConnector(_GoogleConnector):
    """Pulls Drive files: Google docs exported as text, text blobs downloaded, binaries listed."""

    name = "google-drive"
    label = "Google Drive"
    scope = DRIVE_SCOPE

    def verify(self, auth: ConnectorAuth) -> str:
        """The Drive user's display name and email address."""
        with self._client() as client:
            payload = self._get_json(
                client, auth, f"{_DRIVE_BASE}/about", params={"fields": "user"}
            )
        user = payload.get("user")
        user_obj = cast(JsonObject, user) if isinstance(user, dict) else {}
        name = opt_str(user_obj.get("displayName"))
        email = opt_str(user_obj.get("emailAddress"))
        if name and email:
            return f"{name} ({email})"
        return name or email or "Google Drive user"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Search files (newest modified first) and fetch readable content per file.

        The Drive `q` is derived from the query: `trashed=false` always, `fullText contains`
        for free text, `'<target>' in parents` for a folder id, and `modifiedTime` bounds for
        since/until. Google-native files are exported as text (Docs/Slides as plain text,
        Sheets as CSV); `text/*` and JSON blobs are downloaded (all capped at 200000
        characters); other binaries become body-less FILE items.
        """
        auth = self._fresh(auth)
        q = _drive_search_terms(query)
        items: list[ContextItem] = []
        page_token: str | None = None
        with self._client() as client:
            while len(items) < query.limit:
                params: dict[str, str] = {
                    "q": q,
                    "orderBy": "modifiedTime desc",
                    "pageSize": str(min(query.limit - len(items), _DRIVE_PAGE_MAX)),
                    "fields": _DRIVE_LIST_FIELDS,
                }
                if page_token:
                    params["pageToken"] = page_token
                payload = self._get_json(client, auth, f"{_DRIVE_BASE}/files", params=params)
                for file in _dicts(payload.get("files")):
                    item = self._file_item(client, auth, file)
                    if item is not None:
                        items.append(item)
                    if len(items) >= query.limit:
                        break
                page_token = opt_str(payload.get("nextPageToken"))
                if not page_token:
                    break
        return items

    def _file_item(
        self, client: httpx.Client, auth: ConnectorAuth, file: JsonObject
    ) -> ContextItem | None:
        """Normalize one Drive file, fetching its content when it has a readable form."""
        file_id = opt_str(file.get("id"))
        if not file_id:
            logger.debug("skipping drive file without an id: %r", file)
            return None
        mime = opt_str(file.get("mimeType")) or ""
        body, kind, fetch_error = self._file_body(client, auth, file_id, mime)
        metadata: JsonObject = {"mimeType": mime}
        size = opt_str(file.get("size"))
        if size:
            metadata["size"] = size
        if fetch_error:
            metadata["fetch_error"] = fetch_error
        return ContextItem(
            id=file_id,
            source=self.name,
            kind=kind,
            title=opt_str(file.get("name")) or file_id,
            body=body,
            url=opt_str(file.get("webViewLink")),
            created_at=opt_str(file.get("createdTime")),
            updated_at=opt_str(file.get("modifiedTime")),
            metadata=metadata,
        )

    def _file_body(
        self, client: httpx.Client, auth: ConnectorAuth, file_id: str, mime: str
    ) -> tuple[str, ItemKind, str | None]:
        """(body, kind, fetch_error) for one file; unreadable binaries stay body-less FILEs.

        A failed per-file content fetch (except 401) degrades to an empty body with the
        failure recorded in metadata instead of aborting the whole pull.
        """
        export_mime = _EXPORT_MIME_BY_TYPE.get(mime)
        if export_mime is not None:
            response = self._get(
                client,
                auth,
                f"{_DRIVE_BASE}/files/{quote(file_id, safe='')}/export",
                params={"mimeType": export_mime},
            )
        elif mime.startswith("text/") or mime == "application/json":
            response = self._get(
                client,
                auth,
                f"{_DRIVE_BASE}/files/{quote(file_id, safe='')}",
                params={"alt": "media"},
            )
        else:
            return "", ItemKind.FILE, None
        if response.status_code != 200:
            logger.warning(
                "could not fetch drive file %s content (HTTP %s); leaving its body empty",
                file_id,
                response.status_code,
            )
            return "", ItemKind.DOCUMENT, f"HTTP {response.status_code}"
        return capped(response.text), ItemKind.DOCUMENT, None


class GmailConnector(_GoogleConnector):
    """Pulls Gmail messages as EMAIL items: header block plus the decoded text body."""

    name = "gmail"
    label = "Gmail"
    scope = GMAIL_SCOPE

    def verify(self, auth: ConnectorAuth) -> str:
        """The mailbox's email address."""
        with self._client() as client:
            payload = self._get_json(client, auth, f"{_GMAIL_BASE}/users/me/profile")
        return opt_str(payload.get("emailAddress")) or "Gmail user"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Search messages, then fetch each match in full and normalize it.

        `query.query` passes through as Gmail search syntax; `since`/`until` become
        `after:`/`before:` date operators. Message ids are collected across `nextPageToken`
        pages up to `query.limit`, then each id costs one `format=full` fetch.
        """
        auth = self._fresh(auth)
        q = _gmail_search_terms(query)
        message_ids: list[str] = []
        items: list[ContextItem] = []
        with self._client() as client:
            page_token: str | None = None
            while len(message_ids) < query.limit:
                params: dict[str, str] = {
                    "maxResults": str(min(query.limit - len(message_ids), _GMAIL_PAGE_MAX))
                }
                if q:
                    params["q"] = q
                if page_token:
                    params["pageToken"] = page_token
                payload = self._get_json(
                    client, auth, f"{_GMAIL_BASE}/users/me/messages", params=params
                )
                for ref in _dicts(payload.get("messages")):
                    ref_id = opt_str(ref.get("id"))
                    if ref_id:
                        message_ids.append(ref_id)
                    if len(message_ids) >= query.limit:
                        break
                page_token = opt_str(payload.get("nextPageToken"))
                if not page_token:
                    break
            for message_id in message_ids:
                message = self._get_json(
                    client,
                    auth,
                    f"{_GMAIL_BASE}/users/me/messages/{quote(message_id, safe='')}",
                    params={"format": "full"},
                )
                items.append(self._message_item(message, message_id))
        return items

    def _message_item(self, message: JsonObject, message_id: str) -> ContextItem:
        """Normalize one full-format message into an EMAIL item."""
        payload = message.get("payload")
        payload_obj = cast(JsonObject, payload) if isinstance(payload, dict) else {}
        headers = payload_obj.get("headers")
        sender = _gmail_header(headers, "From")
        to = _gmail_header(headers, "To")
        date = _gmail_header(headers, "Date")
        text = _find_part_text(payload_obj, "text/plain")
        if text is None:
            markup = _find_part_text(payload_obj, "text/html")
            text = strip_html(markup) if markup else ""
        body = f"From: {sender}\nTo: {to}\nDate: {date}"
        if text.strip():
            body = f"{body}\n\n{text.strip()}"
        metadata: JsonObject = {"from": sender, "to": to}
        label_ids = message.get("labelIds")
        if isinstance(label_ids, list):
            metadata["labelIds"] = [label for label in label_ids if isinstance(label, str)]
        thread_id = opt_str(message.get("threadId"))
        if thread_id:
            metadata["threadId"] = thread_id
        return ContextItem(
            id=message_id,
            source=self.name,
            kind=ItemKind.EMAIL,
            title=_gmail_header(headers, "Subject") or "(no subject)",
            body=body,
            url=f"https://mail.google.com/mail/u/0/#all/{message_id}",
            created_at=_from_epoch_ms(message.get("internalDate")),
            metadata=metadata,
        )


# -- shared normalization helpers ---------------------------------------------------------------


def _dicts(value: JsonValue | None) -> list[JsonObject]:
    """The dict entries of a JSON list (non-lists and non-dict entries are dropped)."""
    if not isinstance(value, list):
        return []
    return [cast(JsonObject, entry) for entry in value if isinstance(entry, dict)]


def _parse_iso(value: str, *, field: str) -> datetime:
    """Parse an ISO-8601 date or datetime; naive values are taken as UTC.

    Raises:
        ConnectError: When the value is not ISO-8601.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ConnectError(
            f"could not parse {field}={value!r} as ISO-8601; use YYYY-MM-DD or a full timestamp"
        ) from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _rfc3339(value: str, *, field: str) -> str:
    """An ISO-8601 query bound as the RFC 3339 timestamp Google APIs expect."""
    return _parse_iso(value, field=field).isoformat()


def _iso(moment: datetime) -> str:
    """A datetime as a seconds-precision ISO-8601 string."""
    return moment.isoformat(timespec="seconds")


# -- calendar helpers ---------------------------------------------------------------------------


def _event_body(event: JsonObject) -> str:
    """A human-readable block: when, where, who, then the event description."""
    lines: list[str] = []
    start = _event_moment(event.get("start"))
    end = _event_moment(event.get("end"))
    if start or end:
        lines.append(f"When: {start or 'unknown'} to {end or 'unknown'}")
    location = opt_str(event.get("location"))
    if location:
        lines.append(f"Location: {location}")
    attendees = [label for a in _dicts(event.get("attendees")) if (label := _attendee_label(a))]
    if attendees:
        lines.append("Attendees: " + ", ".join(attendees))
    description = opt_str(event.get("description"))
    if description:
        lines.append("")
        lines.append(description)
    return "\n".join(lines)


def _event_moment(value: JsonValue | None) -> str | None:
    """An event boundary's dateTime (timed) or date (all-day), whichever is present."""
    if not isinstance(value, dict):
        return None
    return opt_str(value.get("dateTime")) or opt_str(value.get("date"))


def _attendee_label(attendee: JsonObject) -> str:
    """ "Name <email>" when both are known, else whichever one is."""
    name = opt_str(attendee.get("displayName"))
    email = opt_str(attendee.get("email"))
    if name and email:
        return f"{name} <{email}>"
    return name or email or ""


# -- drive helpers ------------------------------------------------------------------------------


def _drive_search_terms(query: PullQuery) -> str:
    """The Drive files.list `q` derived from a PullQuery (always excludes trashed files)."""
    terms = ["trashed=false"]
    if query.query:
        terms.append(f"fullText contains '{_drive_escape(query.query)}'")
    if query.target:
        terms.append(f"'{_drive_escape(query.target)}' in parents")
    if query.since:
        terms.append(f"modifiedTime > '{_rfc3339(query.since, field='since')}'")
    if query.until:
        terms.append(f"modifiedTime <= '{_rfc3339(query.until, field='until')}'")
    return " and ".join(terms)


def _drive_escape(value: str) -> str:
    """Escape backslashes and single quotes for a Drive query string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


# -- gmail helpers ------------------------------------------------------------------------------


def _gmail_search_terms(query: PullQuery) -> str:
    """The Gmail `q` string: the user's search syntax plus after:/before: date operators."""
    terms: list[str] = []
    if query.query:
        terms.append(query.query)
    if query.since:
        terms.append(f"after:{_parse_iso(query.since, field='since').strftime('%Y/%m/%d')}")
    if query.until:
        terms.append(f"before:{_parse_iso(query.until, field='until').strftime('%Y/%m/%d')}")
    return " ".join(terms)


def _gmail_header(headers: JsonValue | None, name: str) -> str:
    """The first header value with the given case-insensitive name, or ""."""
    for entry in _dicts(headers):
        if (opt_str(entry.get("name")) or "").lower() == name.lower():
            return opt_str(entry.get("value")) or ""
    return ""


def _find_part_text(part: JsonObject, mime: str) -> str | None:
    """Depth-first search of a message payload tree for the first decodable `mime` part."""
    if opt_str(part.get("mimeType")) == mime:
        body = part.get("body")
        if isinstance(body, dict):
            decoded = _decode_base64url(body.get("data"))
            if decoded:
                return decoded
    for child in _dicts(part.get("parts")):
        found = _find_part_text(child, mime)
        if found is not None:
            return found
    return None


def _decode_base64url(data: JsonValue | None) -> str | None:
    """Decode Gmail's unpadded base64url part data; undecodable data becomes None."""
    if not isinstance(data, str) or not data:
        return None
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except ValueError:
        logger.debug("undecodable base64url message part (%d chars)", len(data))
        return None


def _from_epoch_ms(value: JsonValue | None) -> str | None:
    """A Gmail internalDate (milliseconds since the epoch) as ISO-8601, or None."""
    try:
        ms = int(str(value))
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="seconds")


register_connector(GoogleCalendarConnector())
register_connector(GoogleDriveConnector())
register_connector(GmailConnector())
