"""End to end: a first-token stall behind keepalives fails over to the next rung.

Its own module so the stalled primary's health verdict never leaks into the
waterfall module's ordered circuit scenarios; the serving harness (fixture,
upstreams, ledger reader) is the waterfall module's.
"""

from __future__ import annotations

import time

import httpx

from exp.runtime.gateway.tests.native_waterfall_test import (
    _attempt_rows,
    _chat_payload,
    _ServingEngine,
)
from exp.runtime.gateway.tests.native_waterfall_test import _engine as _engine

__all__ = ["_engine"]


def test_first_token_stall_behind_keepalives_fails_over_to_the_second_deployment(
    engine: _ServingEngine,
) -> None:
    """Headers and keepalive comments do not satisfy the first-token bound.

    The primary answers its headers and streams SSE comments at once, then
    sends no token for ten seconds. The old bound was satisfied by the first
    body byte; the request must instead fail over to the secondary in about
    the harness's one-second bound and complete there, with the stalled
    attempt filed as a failover-eligible timeout and never redialed.
    """
    started = time.monotonic()
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_chat_payload("stall-after-headers"),
        timeout=30.0,
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "from-secondary"
    assert response.headers["x-gateway-route-depth"] == "1"
    assert elapsed < 6.0, f"the stall held the request for {elapsed:.1f}s"
    rows = _attempt_rows(engine, response.headers["x-request-id"])
    assert rows == [(0, 0, "failed"), (1, 1, "completed")]
