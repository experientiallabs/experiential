"""Per-wire form of OpenAI's ``original`` image detail.

OpenAI's SDK types ``ImageDetail`` as ``low``, ``high``, ``auto`` or
``original`` on both the Responses ``input_image`` (top level and inside a
``function_call_output``) and the Chat ``image_url`` part. Codex sends
``original`` on screenshots it attaches to tool output, so the value sits in
replayed history and a decoder that refuses it fails every later turn of the
session.

``original`` is OpenAI's own value: its vision guide lists it for the gpt-5.4
and newer families and calls it unsupported on older models, and no other
origin documents it. So OpenAI's own API, and the ChatGPT Codex backend that
Codex itself sends ``original`` to, take the caller's value verbatim
(an unsupported model answers exactly as OpenAI would), and every other
detail-carrying wire receives ``high``, the nearest value every OpenAI-shaped
origin accepts and the one Codex itself substitutes when a model cannot take
``original``. Each such rung discloses the substitution. Wires without a
detail field (Anthropic, Gemini, Bedrock) never send it, so nothing changes
there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from exp.common.models.content import ImageContentPart
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.surfaces import GatewayApiSurface

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

DETAIL_CARRYING_DIALECTS: Final = frozenset({"openai_compatible", "openai_responses"})
"""The wires whose image part carries a ``detail`` field."""

_ORIGINAL_DETAIL_HOSTS: Final = frozenset({"api.openai.com", "chatgpt.com"})
"""OpenAI's own origins: the Platform API and the ChatGPT Codex backend Codex itself calls."""

RESPONSES_ORIGINAL_DETAIL_DOWNGRADED: Final = "input.input_image.detail->translated(original:high)"
"""Disclosure for a Responses request whose ``original`` detail a rung sends as ``high``."""

CHAT_ORIGINAL_DETAIL_DOWNGRADED: Final = "messages.image_url.detail->translated(original:high)"
"""Disclosure for a Chat request whose ``original`` detail a rung sends as ``high``."""


def _downgrades_original(profile: GatewayWireProfile) -> bool:
    """Whether this rung carries a detail field but not OpenAI's ``original`` value."""
    if profile.dialect not in DETAIL_CARRYING_DIALECTS:
        return False
    return (urlsplit(profile.url).hostname or "").lower() not in _ORIGINAL_DETAIL_HOSTS


def _carries_original(message: GatewayMessage) -> bool:
    """Whether one message holds an image asking for ``original`` detail."""
    return any(
        isinstance(part, ImageContentPart) and part.detail == "original"
        for part in message.content_parts
    )


def original_image_detail_disclosures(
    profile: GatewayWireProfile, request: GatewayRequest
) -> tuple[str, ...]:
    """Disclose the ``original`` to ``high`` substitution this rung sends.

    Args:
        profile: The rung's resolved wire profile.
        request: The rung's canonical request.

    Returns:
        One disclosure naming the surface's detail field, or empty when the
        rung forwards ``original`` verbatim, carries no detail field, or no
        image asked for ``original``.
    """
    if not _downgrades_original(profile) or not any(
        _carries_original(message) for message in request.messages
    ):
        return ()
    if request.surface == GatewayApiSurface.RESPONSES:
        return (RESPONSES_ORIGINAL_DETAIL_DOWNGRADED,)
    return (CHAT_ORIGINAL_DETAIL_DOWNGRADED,)


def original_image_detail_for_wire(
    profile: GatewayWireProfile, request: GatewayRequest
) -> GatewayRequest:
    """Send ``original`` image detail as ``high`` on a rung that cannot carry it.

    Args:
        profile: The rung's resolved wire profile.
        request: The rung's canonical request.

    Returns:
        The request unchanged on OpenAI's own API, on a wire without a detail
        field, or when no image asked for ``original``; otherwise a copy whose
        ``original`` images carry ``high``.
    """
    if not _downgrades_original(profile) or not any(
        _carries_original(message) for message in request.messages
    ):
        return request
    messages = tuple(
        message.model_copy(
            update={
                "content_parts": tuple(
                    part.model_copy(update={"detail": "high"})
                    if isinstance(part, ImageContentPart) and part.detail == "original"
                    else part
                    for part in message.content_parts
                )
            }
        )
        if _carries_original(message)
        else message
        for message in request.messages
    )
    return request.model_copy(update={"messages": messages})
