"""Input wait ownership preserves a ready verdict at a zero-grace boundary."""

import asyncio
import time

from exp.runtime.gateway.guardrails.input_wait import InputWait


def test_ready_result_wins_a_simultaneous_zero_grace_boundary() -> None:
    """An available decision is authoritative even when no extra waiting is allowed."""

    async def exercise() -> None:
        """Order completion and boundary notification in one event-loop turn."""
        budget = InputWait()
        started = asyncio.Event()
        released = asyncio.Event()

        async def inspect() -> str:
            """Return the ready denial only after the test releases its barrier."""
            started.set()
            await released.wait()
            return "denied"

        result = asyncio.create_task(budget.run(inspect, wait_ms=0, deadline=time.monotonic() + 2))
        await started.wait()
        released.set()
        budget.begin()
        assert await result == "denied"

    asyncio.run(exercise())
