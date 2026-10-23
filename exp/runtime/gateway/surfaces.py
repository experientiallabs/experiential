"""The public endpoint families one canonical gateway request can belong to.

Kept apart from :mod:`exp.runtime.gateway.contracts` (which re-exports it) so
the enum has no dependencies and every surface module can import it cheaply.
"""

from __future__ import annotations

from enum import StrEnum


class GatewayApiSurface(StrEnum):
    """Public endpoint family used by one canonical request."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    EMBEDDINGS = "embeddings"
    IMAGES = "images"
    DECISIONS = "decisions"
    SPEECH = "speech"
    TRANSCRIPTION = "transcription"
    VIDEO = "video"
