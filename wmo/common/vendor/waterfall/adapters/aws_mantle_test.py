"""Tests for the unimplemented AWS Mantle rung: it must fail at construction, not mid-call."""

from __future__ import annotations

import pytest

from wmo.common.vendor.waterfall.adapters.aws_mantle import AwsMantleAdapter
from wmo.common.vendor.waterfall.types import Backend


def test_constructing_the_stub_fails_and_names_the_workaround() -> None:
    # Failing here is the contract: a chain that lists this rung must break when it is BUILT, so
    # nobody discovers the gap when a live call has already failed over onto it.
    with pytest.raises(NotImplementedError, match="not implemented yet") as excinfo:
        AwsMantleAdapter(Backend(provider="bedrock", model="anthropic.claude-opus-4-8"))

    message = str(excinfo.value)
    assert "'bedrock' backend" in message
    assert "aws_mantle.py" in message


def test_the_stub_still_declares_the_whole_adapter_protocol() -> None:
    # Keeping the surface complete is what makes finishing the adapter a body-only change. The
    # protocol's own membership is pinned in base_test.py.
    missing = [
        name
        for name in ("complete", "complete_chat", "embed", "embed_model_id")
        if not hasattr(AwsMantleAdapter, name)
    ]
    assert not missing, f"AwsMantleAdapter is missing {missing}"
    assert "backend" in AwsMantleAdapter.__annotations__
