"""Authorized PostHog HogQL pull transport for the focused canonical converter."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue, TypeAdapter

from wmo.common.core.artifacts import JsonObject, SourceIdentity, canonical_json_bytes
from wmo.simulation.ingest.otlp import GENAI_SEMANTIC_CONVENTION_VERSION, TraceNormalizationResult
from wmo.simulation.ingest.posthog_canonical import PostHogPullError, normalize_posthog_payload

_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


class PostHogResponse(Protocol):
    """Small response surface needed from an injected PostHog HTTP client."""

    def raise_for_status(self) -> None:
        """Raise for a non-success HTTP response.

        Raises:
            httpx.HTTPStatusError: If the response is not successful.
        """

    def json(self) -> JsonValue:
        """Return the decoded JSON response body.

        Returns:
            The response payload decoded as a JSON value.
        """


class PostHogHttpClient(Protocol):
    """Small injectable HTTP surface for authorized PostHog query calls."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JsonObject,
        timeout: float,
    ) -> PostHogResponse:
        """POST one bounded HogQL query and return its response.

        Args:
            url: Absolute PostHog query endpoint.
            headers: Request headers containing the authorized bearer token.
            json: Bounded HogQL request body.
            timeout: Maximum request duration in seconds.

        Returns:
            A response object whose status and JSON payload can be verified.
        """


@dataclass(frozen=True)
class PostHogPullRequest:
    """Authorized PostHog pull settings, excluding persisted credential values."""

    project_id: str
    limit: int = 1_000
    since: datetime | None = None
    host: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Validate bounded request parameters before any network operation."""
        if not self.project_id.strip():
            raise PostHogPullError("PostHog pull needs a non-empty project ID")
        if not 1 <= self.limit <= 10_000:
            raise PostHogPullError("PostHog pull limit must be between 1 and 10,000 events")
        if self.since is not None and self.since.tzinfo is None:
            raise PostHogPullError("PostHog pull since timestamp must include a timezone")


def pull_posthog_traces(
    request: PostHogPullRequest,
    *,
    client: PostHogHttpClient | None = None,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
) -> TraceNormalizationResult:
    """Pull authorized PostHog LLM events through the same converter as local exports.

    Args:
        request: Explicit PostHog project, bounded query, optional host, and optional API key.
        client: Optional deterministic HTTP fake. ``None`` creates one bounded ``httpx.Client``.
        semantic_convention_version: Pinned GenAI version for the returned trace source records.

    Returns:
        Canonical traces and source exclusions from the focused PostHog converter.

    Raises:
        PostHogPullError: Credentials, endpoint settings, or response data are invalid.
    """
    api_key = request.api_key or os.environ.get("POSTHOG_API_KEY")
    if not api_key:
        raise PostHogPullError("PostHog pull needs an API key or POSTHOG_API_KEY")
    host = _posthog_host(request.host or os.environ.get("POSTHOG_HOST", "https://us.posthog.com"))
    body = _hogql_body(request)
    endpoint = f"{host}/api/projects/{request.project_id}/query/"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = _query_payload(client, endpoint, headers, body)
    source = SourceIdentity(
        kind="production",
        source_id=f"posthog:{request.project_id}",
        sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )
    return normalize_posthog_payload(
        _hogql_events(payload),
        source=source,
        semantic_convention_version=semantic_convention_version,
    )


def _query_payload(
    client: PostHogHttpClient | None,
    endpoint: str,
    headers: Mapping[str, str],
    body: JsonObject,
) -> JsonValue:
    """Issue one bounded injected or owned HTTP request and validate its JSON result."""
    if client is not None:
        response = client.post(endpoint, headers=headers, json=body, timeout=60.0)
        response.raise_for_status()
        return response.json()
    with httpx.Client() as owned_client:
        response = owned_client.post(endpoint, headers=headers, json=body, timeout=60.0)
        response.raise_for_status()
        return _JSON_VALUE_ADAPTER.validate_python(response.json())


def _posthog_host(value: str) -> str:
    """Validate a region-specific PostHog HTTPS host without accepting credential-bearing URLs."""
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise PostHogPullError("POSTHOG_HOST must be an absolute HTTPS URL")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PostHogPullError("POSTHOG_HOST must not include credentials, query, or fragment")
    return value.rstrip("/")


def _hogql_body(request: PostHogPullRequest) -> JsonObject:
    """Build the bounded ordered HogQL event query used by the authorized direct pull."""
    where = "event like '$ai_%'"
    if request.since is not None:
        timestamp = request.since.astimezone(UTC).isoformat().replace("+00:00", "Z")
        where += f" and timestamp >= toDateTime('{timestamp}')"
    query = (
        "select event, properties, timestamp, uuid from events where "
        f"{where} order by timestamp asc, uuid asc limit {request.limit}"
    )
    return {"query": {"kind": "HogQLQuery", "query": query}}


def _hogql_events(payload: JsonValue) -> JsonValue:
    """Convert HogQL result rows to the event objects accepted from a local export."""
    if not isinstance(payload, dict):
        raise PostHogPullError("PostHog HogQL response must be an object")
    results = payload.get("results")
    if not isinstance(results, list):
        raise PostHogPullError("PostHog HogQL response must contain a results array")
    events: list[JsonObject] = []
    for row in results:
        if isinstance(row, dict):
            events.append(row)
            continue
        if not isinstance(row, list) or len(row) < 4:
            raise PostHogPullError(
                "PostHog HogQL rows must contain event, properties, timestamp, and uuid"
            )
        properties = row[1]
        if isinstance(properties, str):
            try:
                properties = json.loads(properties)
            except json.JSONDecodeError as exc:
                raise PostHogPullError("PostHog HogQL properties JSON is invalid") from exc
        if not isinstance(properties, dict):
            raise PostHogPullError("PostHog HogQL properties must decode to an object")
        event_name = row[0]
        timestamp = row[2]
        event_uuid = row[3]
        if (
            not isinstance(event_name, str)
            or not isinstance(timestamp, str)
            or not isinstance(event_uuid, str)
            or not event_uuid
        ):
            raise PostHogPullError(
                "PostHog HogQL event, timestamp, and uuid values must be non-empty text"
            )
        events.append(
            {
                "event": event_name,
                "properties": properties,
                "timestamp": timestamp,
                "uuid": event_uuid,
            }
        )
    return events
