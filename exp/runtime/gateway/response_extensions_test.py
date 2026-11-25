"""Tests for the host response-extension model and its settlement reply."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.response_extensions import (
    MAXIMUM_EXTENSION_BYTES,
    GatewayResponseExtensions,
)


def test_the_reply_is_the_compact_shape_the_data_plane_reads() -> None:
    """Members render under ``response_extensions`` with headers as pairs."""
    extensions = GatewayResponseExtensions(
        usage={"cost": 0.25, "is_byok": False},
        body={"provider": "example"},
        headers=(("x-example-notice", "ok"),),
    )
    assert json.loads(extensions.reply()) == {
        "response_extensions": {
            "usage": {"cost": 0.25, "is_byok": False},
            "body": {"provider": "example"},
            "headers": [["x-example-notice", "ok"]],
        }
    }
    assert " " not in extensions.reply()


@pytest.mark.parametrize(
    "header",
    [
        ("X-Upper", "v"),
        ("x-request-id", "forged"),
        ("x-gateway-provider", "forged"),
        ("content-length", "1"),
        ("x-ok", "line\r\nbreak"),
    ],
)
def test_engine_owned_and_malformed_headers_are_refused(header: tuple[str, str]) -> None:
    """A header the data plane would drop fails at the host, where it can be fixed."""
    with pytest.raises(ValidationError):
        GatewayResponseExtensions(headers=(header,))


def test_too_many_headers_are_refused() -> None:
    """The header count bound matches the data plane's."""
    with pytest.raises(ValidationError):
        GatewayResponseExtensions(headers=tuple((f"x-h{index}", "v") for index in range(9)))


def test_an_oversized_reply_is_refused_before_it_is_silently_dropped() -> None:
    """The byte bound matches the data plane's, so the host learns of an overflow."""
    extensions = GatewayResponseExtensions(body={"pad": "x" * MAXIMUM_EXTENSION_BYTES})
    with pytest.raises(ValueError, match="exceed"):
        extensions.reply()
