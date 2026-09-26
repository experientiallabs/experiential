"""Conservative full-request token counting before provider dispatch."""

from typing import Protocol, runtime_checkable

from exp.common.models import ModelCapabilities, ModelRequest
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION


@runtime_checkable
class TokenCounter(Protocol):
    """Counts the full serialized request before a model client can send it."""

    def count(self, request: ModelRequest) -> int:
        """Return a conservative number of context tokens required by one request.

        Args:
            request: Complete provider-neutral request before provider conversion.

        Returns:
            A nonnegative count that includes all visible request content.
        """
        ...


class Utf8UpperBoundTokenCounter:
    """Provider-neutral byte upper bound used when no exact tokenizer is supplied."""

    def count(self, request: ModelRequest) -> int:
        """Count UTF-8 request bytes plus per-message framing as a conservative token bound.

        Args:
            request: Complete provider-neutral request to preflight.

        Returns:
            A conservative nonnegative bound that never silently shortens request content.
        """
        rendered = request.model_dump_json(exclude_none=False)
        instruction = (
            len(JSON_OBJECT_SYSTEM_INSTRUCTION.encode("utf-8")) + 4
            if request.json_object_output
            else 0
        )
        return len(rendered.encode("utf-8")) + 4 * len(request.messages) + instruction


def bound_unpublished_output(
    request: ModelRequest, capabilities: ModelCapabilities, token_counter: TokenCounter
) -> ModelRequest:
    """Fit an explicit output budget into context when no separate limit is published.

    The catalog's unknown output capability stays unknown. Only this request's token budget
    is bounded by the remaining context; messages and tools are never shortened. Invalid or
    overflowing inputs remain unchanged for the normal preflight error before dispatch.

    Args:
        request: Full request with its configured output budget.
        capabilities: Exact model metadata, including its known context window.
        token_counter: Conservative counter covering the complete request.

    Returns:
        Request with its output budget bounded by available context, or the unchanged request.
    """
    if (
        capabilities.maximum_output_tokens is not None
        or capabilities.context_window_tokens is None
        or request.maximum_output_tokens is None
    ):
        return request
    input_tokens = token_counter.count(request)
    remaining = capabilities.context_window_tokens - input_tokens
    if input_tokens < 0 or remaining <= 0:
        return request
    return request.model_copy(
        update={"maximum_output_tokens": min(request.maximum_output_tokens, remaining)}
    )
