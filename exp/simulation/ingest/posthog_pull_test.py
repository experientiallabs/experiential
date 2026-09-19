"""Tests for authorized PostHog pull request construction."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.posthog_pull import PostHogPullRequest, pull_posthog_traces


class _EmptyResponse:
    """Return one valid empty HogQL result."""

    def raise_for_status(self) -> None:
        """Accept the deterministic response as successful."""

    def json(self) -> JsonValue:
        """Return an empty HogQL result payload."""
        return {"results": []}


class _CapturingClient:
    """Capture the request URL without performing network traffic."""

    def __init__(self) -> None:
        """Prepare an empty request capture."""
        self.url = ""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JsonObject,
        timeout: float,
    ) -> _EmptyResponse:
        """Capture the URL and accept the remaining bounded request fields."""
        self.url = url
        del headers, json, timeout
        return _EmptyResponse()


def test_project_id_is_encoded_as_one_url_path_segment() -> None:
    """Reserved and Unicode characters cannot change the PostHog request target."""
    client = _CapturingClient()

    pull_posthog_traces(
        PostHogPullRequest(project_id="team/alpha?region=#1% café", api_key="fixture-key"),
        client=client,
    )

    assert client.url == (
        "https://us.posthog.com/api/projects/team%2Falpha%3Fregion%3D%231%25%20caf%C3%A9/query/"
    )


@pytest.mark.parametrize(("project_id", "segment"), [(".", "%2E"), ("..", "%2E%2E")])
def test_project_id_dot_segments_are_encoded(project_id: str, segment: str) -> None:
    """Complete dot segments cannot be normalized into a different PostHog path."""
    client = _CapturingClient()

    pull_posthog_traces(
        PostHogPullRequest(project_id=project_id, api_key="fixture-key"),
        client=client,
    )

    assert client.url == f"https://us.posthog.com/api/projects/{segment}/query/"
