"""Host-supplied additive members for a finished completion.

A host that settles requests through its own ledger can report what it settled
on the response itself, for example the billed amount as ``usage.cost`` (the
OpenRouter shape) and the provider that served at the top level. The native
data plane already awaits the finalizing ``price`` or ``settle`` callback before
it encodes a stream's terminal frames or assembles a buffered body, so the
host's members ride that callback's reply and the data plane stamps them
natively, with no second pass over the response bytes.

Members are additive: the data plane never replaces a key it already emits, so
the OpenAI and Anthropic contracts stay unchanged. Headers apply only to a
buffered answer, because a live stream's headers leave before settlement.
"""

from __future__ import annotations

import json
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exp.common.core.artifacts import JsonObject

MAXIMUM_EXTENSION_BYTES: Final = 16 * 1024
"""Serialized bound the data plane enforces; a larger reply is ignored there."""

MAXIMUM_EXTENSION_HEADERS: Final = 8
"""Header count bound the data plane enforces on one buffered answer."""

_HEADER_NAME = re.compile(r"[a-z0-9-]+")
_RESERVED_HEADERS: Final = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "retry-after",
        "transfer-encoding",
        "x-request-id",
    }
)
_RESERVED_HEADER_PREFIX: Final = "x-gateway-"


class GatewayResponseExtensions(BaseModel):
    """Additive members a host stamps on one finished completion.

    Attributes:
        usage: Members added to the terminal payload's ``usage`` object, for
            example ``{"cost": 0.0016, "is_byok": False}``. A key the data
            plane already emits is never replaced.
        body: Members added beside ``id`` and ``model`` on the terminal payload
            (the Responses envelope, the Chat body or final chunk, the Messages
            body or ``message_delta`` event, the Images API body). Never
            replaces an emitted key. The ``usage`` members also become the
            ``usage`` object of an Images answer whose lane reports none.
        headers: Lower-case ``(name, value)`` pairs added to a buffered answer;
            ignored on a live stream, whose headers precede settlement. Names
            the data plane owns (``content-*``, ``x-request-id``, ``x-gateway-*``)
            are refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    usage: JsonObject = Field(default_factory=dict)
    body: JsonObject = Field(default_factory=dict)
    headers: tuple[tuple[str, str], ...] = Field(default=(), max_length=MAXIMUM_EXTENSION_HEADERS)

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, headers: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
        """Refuse malformed and engine-owned header names and non-printable values."""
        for name, value in headers:
            if not _HEADER_NAME.fullmatch(name):
                message = f"header name {name!r} must be lower-case letters, digits or '-'"
                raise ValueError(message)
            if name in _RESERVED_HEADERS or name.startswith(_RESERVED_HEADER_PREFIX):
                message = f"header {name!r} is owned by the data plane"
                raise ValueError(message)
            if not all(0x20 <= ord(char) < 0x7F for char in value):
                message = f"header {name!r} value must be printable ASCII"
                raise ValueError(message)
        return headers

    def reply(self) -> str:
        """Render the settlement reply the data plane reads.

        Raises:
            ValueError: The serialized reply exceeds the data plane's bound,
                which would silently drop it there.
        """
        rendered = json.dumps(
            {"response_extensions": self.model_dump(mode="json")},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if len(rendered.encode()) > MAXIMUM_EXTENSION_BYTES:
            message = f"response extensions exceed {MAXIMUM_EXTENSION_BYTES} bytes"
            raise ValueError(message)
        return rendered
