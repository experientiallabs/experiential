"""The ``extra_content`` vendor block a caller echoes on a replayed tool call.

Gemini signs a function call with an opaque ``thoughtSignature`` and expects
it back with that call on the next turn. Google's own OpenAI-compatible
endpoint returns it on each Chat tool call as
``extra_content.google.thought_signature`` and reads it back from the same
place, so clients built for that endpoint already round-trip the object
(probed live 2026-10-18 on ``gemini-3-flash-preview``, streaming and not).
The gateway emits the identical shape on Chat tool calls and Responses
``function_call`` items and decodes it here on the way back in.
"""

from __future__ import annotations

import base64
import binascii

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exp.common.models.model import MAXIMUM_THOUGHT_SIGNATURE_CHARACTERS


class GoogleToolCallExtras(BaseModel):
    """Google's block inside ``extra_content``: the call's thought signature.

    Attributes:
        thought_signature: Required standard-base64 signature Gemini issued
            with the call, at most 1,048,576 characters. Google refuses a
            signature that is not base64 or was not issued by Gemini, so a
            malformed value is refused here with the field named instead.
    """

    model_config = ConfigDict(extra="forbid")

    thought_signature: str = Field(min_length=1, max_length=MAXIMUM_THOUGHT_SIGNATURE_CHARACTERS)

    @field_validator("thought_signature")
    @classmethod
    def _require_base64(cls, value: str) -> str:
        """Require the standard-base64 text Gemini emits.

        Args:
            value: The echoed signature.

        Returns:
            The unchanged signature.

        Raises:
            ValueError: The value is not standard base64.
        """
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "thought_signature must be the base64 signature Gemini returned with this call"
            ) from exc
        return value


class ToolCallExtraContent(BaseModel):
    """The ``extra_content`` object on an echoed Chat tool call or Responses item.

    Attributes:
        google: Optional Google block carrying Gemini's thought signature.
            Absent or null means the call carries no signature.
    """

    model_config = ConfigDict(extra="forbid")

    google: GoogleToolCallExtras | None = None


def thought_signature(extra_content: ToolCallExtraContent | None) -> str | None:
    """Return the Gemini thought signature an echoed call carries, if any.

    Args:
        extra_content: The call's validated ``extra_content`` object, or None.

    Returns:
        The signature to replay with the call, or None.
    """
    if extra_content is None or extra_content.google is None:
        return None
    return extra_content.google.thought_signature
