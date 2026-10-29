"""The ``extra_content`` vendor block a caller echoes on a replayed tool call.

Gemini signs a function call with an opaque ``thoughtSignature`` and expects
it back with that call on the next turn. Google's own OpenAI-compatible
endpoint returns it on each Chat tool call as
``extra_content.google.thought_signature`` and reads it back from the same
place, so clients built for that endpoint already round-trip the object
(probed live 2026-10-18 on ``gemini-3-flash-preview``, streaming and not).
The gateway emits the identical shape on Chat tool calls and Responses
``function_call`` items and decodes it here on the way back in.

Some OpenAI-compatible Gemini bridges (the Perplexity Agent API's function
calls, community Gemini-to-OpenAI proxies) instead put the signature directly
on the call as ``thought_signature``, and Node agents that keep those call
objects echo the field to whichever gateway serves the next turn. A Chat tool
call therefore accepts that spelling too: it is the same signature under
another name, validated identically and replayed exactly as the canonical
form is, and a call naming two different signatures is refused.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from exp.common.models.model import MAXIMUM_THOUGHT_SIGNATURE_CHARACTERS


def _require_base64(value: str) -> str:
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


ThoughtSignature = Annotated[
    str,
    Field(min_length=1, max_length=MAXIMUM_THOUGHT_SIGNATURE_CHARACTERS),
    AfterValidator(_require_base64),
]
"""One echoed Gemini signature: nonempty standard base64, at most 1,048,576 characters.

Google refuses a signature that is not base64 or was not issued by Gemini, so
a malformed value is refused at decoding with its field named instead."""


class GoogleToolCallExtras(BaseModel):
    """Google's block inside ``extra_content``: the call's thought signature.

    Attributes:
        thought_signature: Required signature Gemini issued with the call.
    """

    model_config = ConfigDict(extra="forbid")

    thought_signature: ThoughtSignature


class ToolCallExtraContent(BaseModel):
    """The ``extra_content`` object on an echoed Chat tool call or Responses item.

    Attributes:
        google: Optional Google block carrying Gemini's thought signature.
            Absent or null means the call carries no signature.
    """

    model_config = ConfigDict(extra="forbid")

    google: GoogleToolCallExtras | None = None


def thought_signature(
    extra_content: ToolCallExtraContent | None, direct: str | None = None
) -> str | None:
    """Return the Gemini thought signature an echoed call carries, if any.

    Args:
        extra_content: The call's validated ``extra_content`` object, or None.
        direct: The call's validated top-level ``thought_signature``, or None.
            The wire model has already refused a call whose two spellings differ.

    Returns:
        The signature to replay with the call, or None.
    """
    if extra_content is not None and extra_content.google is not None:
        return extra_content.google.thought_signature
    return direct
