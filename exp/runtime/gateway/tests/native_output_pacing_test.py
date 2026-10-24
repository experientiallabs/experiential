"""End to end: an attempt's output pace holds its streamed answer to the rate.

Two engines over the waterfall harness: one with no pace (every request as
today) and one whose control plane returns 500 estimated tokens/second in
every ``start_attempt`` reply (in production only on a self-hosted rung under
a class throttle; class_throttles_test proves that gate). The primary streams
about 1,000 estimated tokens at once.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx
import pytest

from exp.runtime.gateway.tests.native_waterfall_test import (
    _chat_payload,
    _ServingEngine,
    serve_waterfall_engine,
)

_PACE = 500.0


@pytest.fixture(scope="module", name="unpaced")
def _unpaced_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """The harness with no output pace."""
    yield from serve_waterfall_engine(tmp_path_factory)


@pytest.fixture(scope="module", name="paced")
def _paced_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """The harness with a 500 tokens/second pace on every request."""
    yield from serve_waterfall_engine(tmp_path_factory, output_tokens_per_second=_PACE)


def _stream_answer(engine: _ServingEngine) -> tuple[str, float]:
    """Stream the long answer and return its text and wall time."""
    started = time.monotonic()
    text: list[str] = []
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("long-answer", stream=True),
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data: {"):
                continue
            for choice in json.loads(line.removeprefix("data: ")).get("choices", []):
                text.append(choice.get("delta", {}).get("content") or "")
    return "".join(text), time.monotonic() - started


def test_unpaced_stream_is_released_as_fast_as_it_arrives(unpaced: _ServingEngine) -> None:
    """Without a pace the 1,000-token answer streams in well under a second."""
    text, elapsed = _stream_answer(unpaced)
    assert text == "y" * 4_000
    assert elapsed < 0.8, elapsed


def test_paced_stream_holds_the_rate_and_changes_no_byte(paced: _ServingEngine) -> None:
    """At 500 tokens/second the second 500 tokens wait about a second; the text is identical."""
    text, elapsed = _stream_answer(paced)
    assert text == "y" * 4_000
    assert elapsed >= 0.95, elapsed
    assert elapsed < 5.0, elapsed
