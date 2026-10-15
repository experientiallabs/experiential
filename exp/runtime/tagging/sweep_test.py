"""Tests for the concurrency sweep on a simulated decision server in virtual time."""

from __future__ import annotations

import asyncio
import json
import selectors
from collections.abc import Callable, Coroutine

import httpx
import pytest

from exp.common.tags.bank import DEFAULT_TAG_BANK, TagScope
from exp.common.tags.windows import TagWindow
from exp.runtime.models.providers.systemone import SystemOneClient
from exp.runtime.tagging.sweep import (
    ConcurrencySweep,
    ConcurrencySweepError,
    sweep_concurrency,
)
from exp.runtime.tagging.tagger import Tagger

_WINDOW = TagWindow(scope=TagScope.TURN, index=0, start_message=0, end_message=1, text="[x] y")
# Lets every request that starts at one instant arrive before any reads the load.
_ARRIVAL = 1e-6


class _JumpSelector(selectors.SelectSelector):
    """Selector that advances its loop's virtual clock instead of blocking."""

    def __init__(self, loop: _VirtualTimeLoop) -> None:
        """Remember the loop whose clock to advance."""
        super().__init__()
        self._loop = loop

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        """Jump the clock by the wait instead of sleeping, then poll without blocking."""
        if timeout:
            self._loop.now += timeout
        return super().select(0)


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Event loop whose clock jumps to the next timer whenever no callback is ready."""

    def __init__(self) -> None:
        """Start the virtual clock at zero."""
        self.now = 0.0
        super().__init__(_JumpSelector(self))

    def time(self) -> float:
        """Return the virtual time."""
        return self.now


def _virtual[T](make: Callable[[], Coroutine[None, None, T]]) -> T:
    """Run a coroutine to completion on a fresh virtual-time loop."""
    with asyncio.Runner(loop_factory=_VirtualTimeLoop) as runner:
        return runner.run(make())


def _server(latency: Callable[[int], float], fails: Callable[[int, int], bool]) -> SystemOneClient:
    """Simulate a decision server whose latency and failures depend on concurrent load.

    Args:
        latency: Seconds a request takes when this many are in flight.
        fails: Whether request number ``n`` (counted from 1) fails at a given load.

    Returns:
        A client allowing up to 16 requests in flight against the simulation.
    """
    active = served = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        """Wait for concurrent arrivals, then answer after the load's latency."""
        nonlocal active, served
        active += 1
        served += 1
        number = served
        await asyncio.sleep(_ARRIVAL)
        load = active
        await asyncio.sleep(latency(load))
        active -= 1
        if fails(number, load):
            return httpx.Response(503)
        questions = json.loads(request.content)["questions"]
        answers = {key: {"type": "noul", "noul": 0.5} for key in questions}
        usage = {"input_tokens": 100, "output_tokens": len(questions)}
        return httpx.Response(200, json={"model": "clef", "answers": answers, "usage": usage})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return SystemOneClient(
        "http://127.0.0.1:30000/v1", "clef", max_concurrency=16, http_client=http_client
    )


def _sweep(
    target: SystemOneClient | Tagger, levels: tuple[int, ...], max_error_rate: float = 0.0
) -> ConcurrencySweep:
    """Sweep the levels for ten virtual seconds each."""

    async def run() -> ConcurrencySweep:
        """Measure on the running virtual loop's clock."""
        clock = asyncio.get_running_loop().time
        return await sweep_concurrency(
            target, [_WINDOW], levels, 10.0, max_error_rate=max_error_rate, clock=clock
        )

    return _virtual(run)


def _saturating(load: int) -> float:
    """Take one second up to four in flight, then slow down quadratically."""
    return 1.0 if load <= 4 else (load / 4) ** 2


def test_sweep_picks_the_level_with_the_most_windows_per_second() -> None:
    """Throughput rises until the server saturates; the peak level wins."""
    sweep = _sweep(_server(_saturating, lambda n, load: False), (1, 2, 4, 8))
    assert sweep.best_in_flight == 4
    rates = [round(level.windows_per_second, 3) for level in sweep.levels]
    assert rates == [1.0, 2.0, 4.0, 2.0]
    four = sweep.levels[2]
    assert (four.completed, four.errors, four.first_error) == (40, 0, None)
    assert four.p50_latency_seconds == pytest.approx(1.0 + 1e-6)
    assert four.input_tokens_per_second == pytest.approx(400.0, rel=1e-3)
    assert sweep.levels[3].p95_latency_seconds == pytest.approx(4.0 + 1e-6)


def test_sweep_skips_faster_levels_whose_errors_exceed_the_threshold() -> None:
    """A level failing one request in five wins only when the threshold allows it."""

    def fails(number: int, load: int) -> bool:
        """Fail every fifth request at eight in flight."""
        return load == 8 and number % 5 == 0

    def flat(load: int) -> float:
        """Take one second at any load."""
        return 1.0

    strict = _sweep(_server(flat, fails), (4, 8))
    assert strict.best_in_flight == 4
    assert strict.levels[1].error_rate == pytest.approx(0.2)
    assert "HTTP 503" in (strict.levels[1].first_error or "")
    loose = _sweep(_server(flat, fails), (4, 8), max_error_rate=0.25)
    assert loose.best_in_flight == 8


def test_sweep_reports_when_no_level_is_healthy() -> None:
    """Every level failing raises an error carrying the first failure."""
    client = _server(lambda load: 1.0, lambda n, load: True)
    with pytest.raises(ConcurrencySweepError, match="HTTP 503"):
        _sweep(client, (1, 2))


def test_sweep_uses_the_tagger_bank_and_rejects_levels_the_client_cannot_hold() -> None:
    """Levels beyond the client's bound fail before any request is sent."""
    client = _server(lambda load: 1.0, lambda n, load: False)
    tagger = Tagger(client, bank=DEFAULT_TAG_BANK, max_in_flight=2)
    assert _sweep(tagger, (2,)).best_in_flight == 2
    with pytest.raises(ValueError, match="max_concurrency"):
        _sweep(client, (8, 32))
    with pytest.raises(ValueError, match="distinct"):
        _sweep(client, (2, 2))
