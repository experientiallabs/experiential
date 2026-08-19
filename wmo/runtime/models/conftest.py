"""Shared deterministic transport fakes for model registry and provider tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport


class RecordedRequest(NamedTuple):
    """One request a scripted transport served, kept for wire assertions in tests.

    The payload is the JSON body a POST sent; GET reads record an empty object.
    """

    url: str
    headers: Mapping[str, str]
    payload: JsonObject


class ScriptedJsonTransport(JsonHttpTransport):
    """Deterministic sync GET transport that replays scripted answers and records requests.

    An empty script doubles as an unused-transport guard: any request raises AssertionError.
    """

    def __init__(self, responses: Sequence[JsonHttpResponse | Exception] = ()) -> None:
        """Store the answers served in order, one per expected request.

        Args:
            responses: Responses to return or exceptions to raise, consumed in order.
        """
        self._responses = list(responses)
        self.requests: list[RecordedRequest] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one GET and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers sent by the caller.
            timeout_seconds: Bounded per-attempt timeout, ignored by the fake.

        Returns:
            The next scripted response.

        Raises:
            Exception: The next scripted error, or AssertionError once the script is exhausted.
        """
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), {}))
        return self._answer()

    def _answer(self) -> JsonHttpResponse:
        """Consume and serve the next scripted answer, failing closed when exhausted."""
        if not self._responses:
            raise AssertionError("test made an unexpected provider request")
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class ScriptedAsyncJsonTransport:
    """Deterministic async transport that records requests and replays scripted answers.

    An empty script doubles as an unused-transport guard: any request raises AssertionError.
    """

    def __init__(self, responses: Sequence[JsonHttpResponse | Exception] = ()) -> None:
        """Store one answer for every expected async request.

        Args:
            responses: Ordered response objects or exceptions.
        """
        self._responses = list(responses)
        self.requests: list[RecordedRequest] = []
        self.timeouts: list[float] = []

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one GET and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers.
            timeout_seconds: Remaining attempt timeout.

        Returns:
            The next scripted response.
        """
        self.requests.append(RecordedRequest(url, dict(headers), {}))
        self.timeouts.append(timeout_seconds)
        return self._answer()

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one POST and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers.
            payload: Complete JSON request body.
            timeout_seconds: Remaining attempt timeout.

        Returns:
            The next scripted response.
        """
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        self.timeouts.append(timeout_seconds)
        return self._answer()

    def _answer(self) -> JsonHttpResponse:
        """Consume the next answer, failing closed when the script is exhausted."""
        if not self._responses:
            raise AssertionError("test made an unexpected provider request")
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer
