"""Terminal metadata observation preserves the original streamed engine objects."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from exp.optimize.claas.backends.verl.completion import observe_completion


@dataclass(frozen=True)
class Completion:
    """Inert native-output metadata for testing the generic observer."""

    finish_reason: str | None


@dataclass(frozen=True)
class Output:
    """Read-only shape used by the observer, not a synthetic model response."""

    finished: bool
    outputs: tuple[Completion, ...]


def test_observer_retains_stop_and_length_without_replacing_original_outputs() -> None:
    """Observed reasons come from native terminal metadata, never token counts or text."""

    async def exercise() -> None:
        """Forward partial and terminal outputs with exact object identity."""
        for reason in ("stop", "length"):
            partial = Output(False, (Completion(None),))
            final = Output(True, (Completion(reason),))
            reasons: dict[str, Literal["stop", "length"]] = {}

            async def generate(
                *, request_id: str, partial: Output = partial, final: Output = final
            ) -> AsyncIterator[Output]:
                """Yield an inert native-shaped stream unchanged by its observer."""
                assert request_id == "r"
                yield partial
                yield final

            observed = [
                item async for item in observe_completion(generate, reasons)(request_id="r")
            ]
            assert observed[0] is partial and observed[1] is final
            assert reasons == {"r": reason}

    asyncio.run(exercise())
