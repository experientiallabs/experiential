"""Shallow validation of the Messages fields forwarded byte-for-byte."""

from __future__ import annotations

import pytest

from exp.runtime.anthropic_protocol.manifest import (
    MESSAGES_BETA_TOKENS_FIELD_BOUND,
    MESSAGES_BETA_TOKENS_FORWARDED,
)
from exp.runtime.anthropic_protocol.passthrough_fields import (
    beta_tokens,
    context_management,
    diagnostics,
    safeguards,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def test_forwarded_objects_pass_verbatim_and_wrong_shapes_are_refused() -> None:
    """Each field keeps its caller value and refuses only a wrong outer shape."""
    assert context_management({}) is None
    assert context_management({"context_management": {"edits": []}}) == {"edits": []}
    assert diagnostics({"diagnostics": {"id": "x"}}) == {"id": "x"}
    assert safeguards({"safeguards": [{"kind": "auto"}]}) == ({"kind": "auto"},)
    with pytest.raises(OpenAIProtocolError, match="context_management must be a JSON object"):
        context_management({"context_management": []})
    with pytest.raises(OpenAIProtocolError, match="diagnostics must be a JSON object"):
        diagnostics({"diagnostics": "x"})
    with pytest.raises(OpenAIProtocolError, match="safeguards must be an array"):
        safeguards({"safeguards": ["x"]})


def test_beta_tokens_partition_against_the_allowlist() -> None:
    """Allowlisted tokens forward deduped, unknown ones are disclosed, field-bound ones vanish."""
    forwarded = next(iter(sorted(MESSAGES_BETA_TOKENS_FORWARDED)))
    bound = next(iter(sorted(MESSAGES_BETA_TOKENS_FIELD_BOUND)))
    header = f"{forwarded}, unknown-beta,{forwarded},{bound},,unknown-beta"
    assert beta_tokens(header) == ((forwarded,), ("anthropic-beta.unknown-beta",))
    assert beta_tokens(None) == ((), ())
    with pytest.raises(OpenAIProtocolError, match="display-safe"):
        beta_tokens("bad\nvalue")
