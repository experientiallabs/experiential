"""Fields only the Responses surface defines on the canonical gateway request."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exp.runtime.gateway.contracts import GatewayRequest


def require_no_responses_only_fields(request: GatewayRequest) -> None:
    """Refuse a Responses-only field on a request of another surface.

    Args:
        request: A canonical request whose surface is not Responses.

    Raises:
        ValueError: The request carries a field only Responses defines.
    """
    present = {
        "reasoning_summary": request.reasoning_summary is not None,
        "response_store": request.response_store is not None,
        "include_output_text_logprobs": request.include_output_text_logprobs,
        "include_web_search_sources": request.include_web_search_sources,
        "include_encrypted_reasoning": request.include_encrypted_reasoning,
        "reasoning_context": request.reasoning_context is not None,
    }
    for name, set_here in present.items():
        if set_here:
            raise ValueError(f"{name} is valid only for Responses requests")
