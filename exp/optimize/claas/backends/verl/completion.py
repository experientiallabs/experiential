"""Observe native terminal metadata without altering sampled model output."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Literal, Protocol, cast


class CompletionPart(Protocol):
    """The native completion metadata observed without depending on an optional SDK."""

    @property
    def finish_reason(self) -> str | None:
        """Return the engine's actual terminal reason."""
        ...


class NativeOutput(Protocol):
    """The narrow read-only vLLM output surface needed for terminal provenance."""

    @property
    def finished(self) -> bool:
        """Whether this is a terminal engine output."""
        ...

    @property
    def outputs(self) -> Sequence[CompletionPart]:
        """Expose each original completion's terminal metadata."""
        ...


def observe_completion[T: NativeOutput, **P](
    generate: Callable[P, AsyncIterator[T]],
    reasons: dict[str, Literal["stop", "length"]],
) -> Callable[P, AsyncIterator[T]]:
    """Retain the actual native finish reason that veRL's TokenOutput otherwise collapses."""

    async def observed(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[T]:
        """Forward each original output unchanged while recording its terminal metadata."""
        request_id = kwargs.get("request_id")
        if not isinstance(request_id, str):
            raise ValueError("upstream generation must supply a named request_id")
        async for output in generate(*args, **kwargs):
            if output.finished and output.outputs:
                reason = output.outputs[0].finish_reason
                if reason in {"stop", "length"}:
                    reasons[request_id] = cast(Literal["stop", "length"], reason)
            yield output

    return observed
